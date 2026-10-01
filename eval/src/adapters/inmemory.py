from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .base import RetrievalAdapter
from ..types import MessageEntry, SearchHit


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
# Simulation day 0.0 (eval/src/masim_loader.py writes message timestamps as
# ISO strings relative to it).
_SIM_BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)


class InMemoryAdapter(RetrievalAdapter):
    """BM25 lexical retriever with ego-scoped indexing.

    Maintains a per-ego BM25 index so that search is restricted to
    sessions the ego agent participated in (ego-centric benchmark).
    Falls back to the global index if no ego_id is provided.
    """

    def __init__(self) -> None:
        # Global index (all messages)
        self._messages: Dict[str, List[MessageEntry]] = defaultdict(list)
        self._index: Dict[str, BM25Okapi] = {}
        # Per-ego index: namespace -> ego_id -> (messages, bm25)
        self._ego_messages: Dict[str, Dict[str, List[MessageEntry]]] = defaultdict(lambda: defaultdict(list))
        self._ego_index: Dict[str, Dict[str, BM25Okapi]] = defaultdict(dict)
        # Time-scoped indexes used by the non-interleaved MASim path.
        # Key: (namespace, ego_id, cutoff_day, excluded_thread_ids)
        self._scoped_messages: Dict[Tuple[str, str, Optional[int], Tuple[str, ...]], List[MessageEntry]] = {}
        self._scoped_index: Dict[Tuple[str, str, Optional[int], Tuple[str, ...]], BM25Okapi] = {}
        # Session -> participants mapping (built during add)
        self._session_participants: Dict[str, Set[str]] = {}

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        self._messages[namespace].extend(messages)

        # Track session participants from message metadata
        for m in messages:
            sid = m.thread_id
            if sid and sid not in self._session_participants:
                # Extract participants from message text header if available
                # thread_id is the session_id, user_id might have participant info
                self._session_participants[sid] = set()

        # Build global BM25 index only if no per-ego indexing path is available.
        # When _ego_session_map is set (via set_ego_session_map), search() uses
        # per-ego indexes built lazily in _ensure_ego_index; the global index
        # would never be queried. Rebuilding it on every add() is O(N^2) over
        # the full corpus and dominates wall-clock on slower CPUs — skip it.
        if not hasattr(self, '_ego_session_map'):
            tokenized = [_tokenize(m.text) for m in self._messages[namespace]]
            self._index[namespace] = BM25Okapi(tokenized)
        else:
            # New messages invalidate any lazily built ego/time-scoped indexes
            # for this namespace. The common non-interleaved path bulk-adds
            # once, so this stays cheap while keeping interleaved fallback exact.
            self._ego_messages.pop(namespace, None)
            self._ego_index.pop(namespace, None)
            for key in [k for k in self._scoped_index if k[0] == namespace]:
                self._scoped_index.pop(key, None)
                self._scoped_messages.pop(key, None)

        return {
            "namespace": namespace,
            "indexed_messages": len(self._messages[namespace]),
        }

    def set_ego_session_map(self, ego_session_map: Dict[str, List[str]]) -> None:
        """Set the ego -> session_ids mapping for ego-scoped indexing.

        Call this after add() to build per-ego BM25 indexes.
        """
        self._ego_session_map = ego_session_map

    def _ensure_ego_index(self, namespace: str, ego_id: str) -> None:
        """Lazily build per-ego BM25 index on first search."""
        if ego_id in self._ego_index.get(namespace, {}):
            return

        ego_sids = set(getattr(self, '_ego_session_map', {}).get(ego_id, []))
        if not ego_sids:
            return

        # Filter messages to ego's sessions
        ego_msgs = [m for m in self._messages.get(namespace, [])
                     if m.thread_id in ego_sids]
        if not ego_msgs:
            return

        self._ego_messages[namespace][ego_id] = ego_msgs
        tokenized = [_tokenize(m.text) for m in ego_msgs]
        self._ego_index[namespace][ego_id] = BM25Okapi(tokenized)

    async def search(
        self,
        *,
        namespace: str,
        query: str,
        top_k: int,
        ego_id: Optional[str] = None,
        query_timestamp: Optional[object] = None,
        exclude_thread_ids: Optional[List[str]] = None,
        **_: object,
    ) -> List[SearchHit]:
        cutoff_day = _coerce_day(query_timestamp)
        excluded = tuple(sorted(str(x) for x in (exclude_thread_ids or []) if str(x)))

        if ego_id and (cutoff_day is not None or excluded):
            scoped = self._ensure_scoped_index(namespace, ego_id, cutoff_day, excluded)
            if scoped is not None:
                scoped_rows, scoped_bm25 = scoped
                return self._bm25_search(scoped_rows, scoped_bm25, query, top_k)
            if getattr(self, '_ego_session_map', {}).get(ego_id):
                # Known ego, nothing visible yet at query time: no context.
                # (Falling through to the unscoped ego index would leak
                # sessions after the cutoff.)
                return []

        # Try ego-scoped search first
        if ego_id and hasattr(self, '_ego_session_map'):
            self._ensure_ego_index(namespace, ego_id)
            ego_bm25 = self._ego_index.get(namespace, {}).get(ego_id)
            ego_rows = self._ego_messages.get(namespace, {}).get(ego_id, [])
            if ego_bm25 and ego_rows:
                return self._bm25_search(ego_rows, ego_bm25, query, top_k)

        # Fallback to global index
        rows = self._messages.get(namespace, [])
        if not rows:
            return []
        bm25 = self._index.get(namespace)
        if bm25 is None:
            return []
        return self._bm25_search(rows, bm25, query, top_k)

    def _ensure_scoped_index(
        self,
        namespace: str,
        ego_id: str,
        cutoff_day: Optional[int],
        excluded: Tuple[str, ...],
    ) -> Optional[Tuple[List[MessageEntry], BM25Okapi]]:
        """Build a BM25 index scoped to one ego and one query-time day.

        This matches the day-barrier semantics used by the memory-cache
        builders: only sessions visible to the ego and delivered no later than
        the query day are searchable. It lets RAG run non-interleaved without
        leaking future days, so answer generation can use normal batching.
        """
        key = (namespace, str(ego_id), cutoff_day, excluded)
        if key in self._scoped_index:
            return self._scoped_messages[key], self._scoped_index[key]

        ego_sids = set(getattr(self, '_ego_session_map', {}).get(ego_id, []))
        if not ego_sids:
            return None
        excluded_sids = set(excluded)
        rows: List[MessageEntry] = [
            m for m in self._messages.get(namespace, [])
            if m.thread_id in ego_sids and visible_at(m, cutoff_day, excluded_sids)
        ]
        if not rows:
            return None

        self._scoped_messages[key] = rows
        self._scoped_index[key] = BM25Okapi([_tokenize(m.text) for m in rows])
        return rows, self._scoped_index[key]

    @staticmethod
    def _bm25_search(
        rows: List[MessageEntry],
        bm25: BM25Okapi,
        query: str,
        top_k: int,
    ) -> List[SearchHit]:
        q_tokens = _tokenize(query)
        scores = bm25.get_scores(q_tokens)

        k = max(1, int(top_k))
        if len(scores) <= k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s <= 0 and results:
                break
            m = rows[idx]
            results.append(SearchHit(
                msg_id=m.msg_id,
                score=s,
                text=m.text,
                occur_ts=m.occur_ts,
                thread_id=m.thread_id,
                user_id=m.user_id,
            ))

        # Fallback to recency if no BM25 matches
        if not results:
            tail = rows[max(0, len(rows) - k):]
            return [
                SearchHit(
                    msg_id=m.msg_id,
                    score=0.0,
                    text=m.text,
                    occur_ts=m.occur_ts,
                    thread_id=m.thread_id,
                    user_id=m.user_id,
                )
                for m in reversed(tail)
            ]

        return results


def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def _coerce_day(value: object) -> Optional[int]:
    """Simulation day of a timestamp: a float day (``7.53``) or an ISO string
    relative to day 0 = 2025-01-01 (how masim_loader writes message times)."""
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((dt - _SIM_BASE).total_seconds() // 86400)


def message_day(m: MessageEntry) -> Optional[int]:
    return _coerce_day(m.deliver_ts) if m.deliver_ts else _coerce_day(m.occur_ts)


def visible_at(m: MessageEntry, cutoff_day: Optional[int], excluded_sids: Iterable[str] = ()) -> bool:
    """Whether a message is searchable for a query: not in an excluded
    session, and delivered no later than the query day. Shared by the BM25,
    dense and hybrid retrievers so they cut identically."""
    if m.thread_id in excluded_sids:
        return False
    if cutoff_day is not None:
        day = message_day(m)
        if day is not None and day > cutoff_day:
            return False
    return True
