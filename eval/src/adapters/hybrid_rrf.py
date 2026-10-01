"""Hybrid BM25 + dense BGE-M3 retriever combined via Reciprocal Rank Fusion.

Each candidate's RRF score is the sum over retrievers of 1 / (k + rank).
We use the standard RRF k=60 from the original paper. Final ranking takes
the top-K by RRF score.

Used for the b2_rag_strong ablation (pre-registered wave-1 ablations) — tests whether a
"strong" retriever combining lexical + dense recall narrows the gap to
oracle / structured-memory.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import RetrievalAdapter, normalize_score
from .dense_bge_m3 import DenseBgeM3Adapter
from .inmemory import InMemoryAdapter
from ..types import MessageEntry, SearchHit


_RRF_K = 60
_FIRST_STAGE_K = 30  # candidates per retriever before fusion


class HybridRrfAdapter(RetrievalAdapter):
    """BM25 ∪ BGE-M3 → Reciprocal Rank Fusion → top-K."""

    def __init__(self) -> None:
        self._bm25 = InMemoryAdapter()
        self._dense = DenseBgeM3Adapter()

    def set_ego_session_map(self, ego_session_map: Dict[str, List[str]]) -> None:
        self._bm25.set_ego_session_map(ego_session_map)
        self._dense.set_ego_session_map(ego_session_map)

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        await self._bm25.add(namespace=namespace, messages=messages)
        await self._dense.add(namespace=namespace, messages=messages)
        return {
            "namespace": namespace,
            "indexed_messages": len(messages),
            "mode": "hybrid_rrf",
        }

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
        first_stage = max(_FIRST_STAGE_K, top_k)
        scope = {"ego_id": ego_id, "query_timestamp": query_timestamp, "exclude_thread_ids": exclude_thread_ids}
        bm25_hits = await self._bm25.search(
            namespace=namespace, query=query, top_k=first_stage, **scope,
        )
        dense_hits = await self._dense.search(
            namespace=namespace, query=query, top_k=first_stage, **scope,
        )

        # RRF: fuse by 1/(k + rank). Identify candidates by msg_id.
        rrf: Dict[str, float] = {}
        store: Dict[str, SearchHit] = {}

        for rank, h in enumerate(bm25_hits, start=1):
            rrf[h.msg_id] = rrf.get(h.msg_id, 0.0) + 1.0 / (_RRF_K + rank)
            store.setdefault(h.msg_id, h)

        for rank, h in enumerate(dense_hits, start=1):
            rrf[h.msg_id] = rrf.get(h.msg_id, 0.0) + 1.0 / (_RRF_K + rank)
            store.setdefault(h.msg_id, h)

        if not rrf:
            return []

        # Sort by fused score descending, take top_k
        ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]
        max_s = ordered[0][1]
        min_s = ordered[-1][1] if len(ordered) > 1 else max_s

        results = []
        for mid, score in ordered:
            h = store[mid]
            results.append(SearchHit(
                msg_id=h.msg_id,
                score=normalize_score(score, min_value=min_s, max_value=max_s),
                text=h.text,
                occur_ts=h.occur_ts,
                thread_id=h.thread_id,
                user_id=h.user_id,
            ))
        return results

    async def reset(self, *, namespace: str) -> None:
        await self._bm25.reset(namespace=namespace)
        await self._dense.reset(namespace=namespace)

    async def close(self) -> None:
        await self._dense.close()
        await self._bm25.close()
