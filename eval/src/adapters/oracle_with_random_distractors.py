from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .base import PromptStyle, RetrievalAdapter
from ..types import MessageEntry, SearchHit

logger = logging.getLogger(__name__)

# Token budget for returned context (matches answering.py heuristic).
_MAX_CONTEXT_TOKENS = 8192
_CHARS_PER_TOKEN = 3


class OracleWithRandomDistractorsAdapter(RetrievalAdapter):
    """§G5 control: Oracle evidence + TEMPORALLY-UNIFORM distractor sampling.

    Same as OracleWithDistractorsAdapter EXCEPT distractor sessions are
    sampled uniformly across the ego's full session history (not biased
    toward the recent tail).

    Sampling is per-query deterministic via a hash of (ego_id, query) so
    repeated calls return the same distractors.

    Reviewer #5 (0426 round) flagged that recency-biased distractors
    confound distractor presence with temporal recency. This adapter
    isolates whether d8_temporal degradation persists when distractors
    are temporally uniform.
    """

    # The answer prompt is built from this adapter's hits (evidence +
    # distractor turns), not from the gold evidence session ids.
    text_context_from_hits = True

    @property
    def prompt_style(self) -> PromptStyle:
        return PromptStyle.TEXT_SESSIONS

    def __init__(self, *, max_messages: int = 2000) -> None:
        self.max_messages = max(1, int(max_messages))
        self._messages: Dict[str, List[MessageEntry]] = defaultdict(list)
        self._ego_session_map: Optional[Dict[str, List[str]]] = None
        self._corpus_sessions: Optional[Dict[str, dict]] = None

    # ── Injection setters (called by EvalPipeline.__init__) ────────────

    def set_ego_session_map(self, ego_session_map: Dict[str, List[str]]) -> None:
        self._ego_session_map = ego_session_map

    def set_corpus_sessions(self, corpus_sessions: Dict[str, dict]) -> None:
        self._corpus_sessions = corpus_sessions

    # ── RetrievalAdapter interface ─────────────────────────────────────

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        self._messages[namespace].extend(messages)
        return {
            "namespace": namespace,
            "indexed_messages": len(self._messages[namespace]),
            "mode": "oracle_with_distractors",
        }

    async def search(
        self,
        *,
        namespace: str,
        query: str,
        top_k: int,
        ego_id: Optional[str] = None,
        evidence_session_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[SearchHit]:
        rows = self._messages.get(namespace, [])
        if not rows:
            return []

        evidence_sids: set = set(evidence_session_ids or [])

        # ── Build index of stored messages by thread_id (= session_id) ──
        msgs_by_session: Dict[str, List[MessageEntry]] = defaultdict(list)
        for m in rows:
            msgs_by_session[m.thread_id].append(m)

        # ── Identify distractor sessions ────────────────────────────────
        distractor_sids: List[str] = []
        if ego_id and self._ego_session_map:
            ego_sids = self._ego_session_map.get(ego_id, [])
            # ego_session_map lists sessions in chronological order;
            # pick the most recent ones that are NOT evidence.
            distractor_sids = [
                sid for sid in ego_sids
                if sid not in evidence_sids and sid in msgs_by_session
            ]
            # Take the tail (most recent) — we'll trim to budget later.
            # No fixed count cap here; token budget controls total size.

        # ── Merge evidence + distractors ────────────────────────────────
        selected_sids_set = set(evidence_sids) | set(distractor_sids)
        # Gather all messages belonging to selected sessions.
        selected: List[MessageEntry] = []
        for sid in selected_sids_set:
            selected.extend(msgs_by_session.get(sid, []))

        # Sort by timestamp (chronological).
        selected.sort(key=lambda m: m.occur_ts)

        # ── Truncate to token budget ────────────────────────────────────
        # Evidence messages are always kept; distractors fill remaining budget.
        evidence_msgs = [m for m in selected if m.thread_id in evidence_sids]
        distractor_msgs = [m for m in selected if m.thread_id not in evidence_sids]

        budget_chars = _MAX_CONTEXT_TOKENS * _CHARS_PER_TOKEN
        evidence_chars = sum(len(m.text) for m in evidence_msgs)
        remaining_chars = max(0, budget_chars - evidence_chars)

        # §G5 change: shuffle distractors with a per-query deterministic
        # seed (hash of ego_id + query), then fill the budget. This
        # samples distractor messages uniformly across the ego's history
        # instead of taking the recent tail, removing the recency
        # confound flagged by reviewer #5.
        seed_key = f"{ego_id or ''}\x00{query}".encode("utf-8")
        seed = int.from_bytes(hashlib.blake2b(seed_key, digest_size=8).digest(), "big")
        rng = random.Random(seed)
        distractor_pool = list(distractor_msgs)
        rng.shuffle(distractor_pool)

        kept_distractors: List[MessageEntry] = []
        running = 0
        for m in distractor_pool:
            c = len(m.text)
            if running + c > remaining_chars:
                break
            kept_distractors.append(m)
            running += c

        final = evidence_msgs + kept_distractors
        final.sort(key=lambda m: m.occur_ts)

        # Apply overall message count cap.
        final = final[-self.max_messages:]

        return [
            SearchHit(
                msg_id=m.msg_id,
                score=1.0 if m.thread_id in evidence_sids else 0.5,
                text=m.text,
                occur_ts=m.occur_ts,
                thread_id=m.thread_id,
                user_id=m.user_id,
            )
            for m in final
        ]
