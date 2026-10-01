from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import numpy as np
from rank_bm25 import BM25Okapi

from .base import RetrievalAdapter
from ..types import MessageEntry, SearchHit


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def _to_epoch(ts: Union[str, float, int]) -> float:
    """Convert a timestamp (ISO string or numeric) to epoch seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return float(ts)
    except (ValueError, TypeError):
        pass
    # Parse ISO 8601 string
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


class TemporalAdapter(RetrievalAdapter):
    """Structured temporal memory: BM25 + exponential recency decay.

    Combines lexical relevance (BM25) with temporal proximity weighting.
    More recent messages receive higher scores via exponential decay,
    simulating a structured memory that prioritizes recent information.

    Final score = bm25_score * temporal_weight
    where temporal_weight = exp(-lambda * age_days)

    Parameters:
        half_life_days: Number of days for the temporal weight to halve (default 30).
        temporal_alpha: Blend factor [0,1] between pure BM25 (0) and
            temporal-weighted BM25 (1). Default 0.5 for balanced weighting.
    """

    def __init__(
        self,
        *,
        half_life_days: float = 30.0,
        temporal_alpha: float = 0.5,
        window_days: Optional[float] = None,
    ) -> None:
        self._messages: Dict[str, List[MessageEntry]] = defaultdict(list)
        self._index: Dict[str, BM25Okapi] = {}
        self._corpus: Dict[str, List[List[str]]] = defaultdict(list)
        self._half_life_days = max(1.0, float(half_life_days))
        self._decay_lambda = math.log(2) / self._half_life_days
        self._temporal_alpha = max(0.0, min(1.0, float(temporal_alpha)))
        self._window_days: Optional[float] = float(window_days) if window_days is not None else None

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        # Bulk-store messages and INVALIDATE any cached BM25 index — the index
        # is rebuilt lazily on first search() call. Rebuilding on every add()
        # was O(N^2) over the full corpus and dominated wall-clock during
        # interleaved ingestion of memarena-L's 137k messages (~7h vs ~5min).
        self._messages[namespace].extend(messages)
        self._index.pop(namespace, None)
        self._corpus.pop(namespace, None)
        return {
            "namespace": namespace,
            "indexed_messages": len(self._messages[namespace]),
            "mode": "temporal",
            "half_life_days": self._half_life_days,
            "temporal_alpha": self._temporal_alpha,
            "window_days": self._window_days,
        }

    async def search(self, *, namespace: str, query: str, top_k: int, **kwargs) -> List[SearchHit]:
        rows = self._messages.get(namespace, [])
        if not rows:
            return []

        # Apply temporal window: discard messages older than window_days before
        # the most recent message in this namespace.
        if self._window_days is not None:
            all_ts = [_to_epoch(m.occur_ts) for m in rows]
            max_ts_all = max(all_ts) if all_ts else 0.0
            cutoff_ts = max_ts_all - self._window_days * 86400.0
            rows = [m for m in rows if _to_epoch(m.occur_ts) >= cutoff_ts]
            if not rows:
                return []

        bm25 = self._index.get(namespace)
        if bm25 is None:
            tokenized = [_tokenize(m.text) for m in rows]
            self._corpus[namespace] = tokenized
            self._index[namespace] = BM25Okapi(tokenized)
            bm25 = self._index[namespace]

        q_tokens = _tokenize(query)
        bm25_scores = bm25.get_scores(q_tokens)

        # Compute temporal weights based on recency relative to the latest message
        timestamps = np.array([_to_epoch(m.occur_ts) for m in rows], dtype=np.float64)
        max_ts = timestamps.max() if len(timestamps) > 0 else 0.0
        age_seconds = max_ts - timestamps
        age_days = age_seconds / 86400.0
        temporal_weights = np.exp(-self._decay_lambda * age_days)

        # Blend: final = (1-alpha)*bm25_norm + alpha*bm25*temporal
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max
        final_scores = (
            (1 - self._temporal_alpha) * bm25_norm
            + self._temporal_alpha * bm25_norm * temporal_weights
        )

        k = max(1, int(top_k))
        if len(final_scores) <= k:
            top_indices = np.argsort(final_scores)[::-1]
        else:
            top_indices = np.argpartition(final_scores, -k)[-k:]
            top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            s = float(final_scores[idx])
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
            if len(results) >= k:
                break

        # Fallback to recency if no matches
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
