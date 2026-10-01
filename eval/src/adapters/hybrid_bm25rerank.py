from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .base import RetrievalAdapter, normalize_score
from .inmemory import InMemoryAdapter
from ..types import MessageEntry, SearchHit

logger = logging.getLogger(__name__)

# Number of BM25 candidates to fetch before reranking
_BM25_FIRST_STAGE_K = 20


class HybridBm25RerankAdapter(RetrievalAdapter):
    """BM25 first-stage recall (top-20) followed by cross-encoder reranking.

    Delegates BM25 retrieval to InMemoryAdapter, then reranks candidates
    using BAAI/bge-reranker-v2-m3 to return the final top-k results.
    """

    def __init__(self) -> None:
        self._bm25 = InMemoryAdapter()
        self._reranker = None  # lazy-loaded

    def _ensure_reranker(self):
        """Lazy-load cross-encoder reranker on first use."""
        if self._reranker is not None:
            return
        from FlagEmbedding import FlagReranker
        logger.info("Loading BAAI/bge-reranker-v2-m3 model...")
        self._reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        logger.info("BAAI/bge-reranker-v2-m3 model loaded.")

    def set_ego_session_map(self, ego_session_map: Dict[str, List[str]]) -> None:
        """Proxy through to the underlying BM25 adapter."""
        self._bm25.set_ego_session_map(ego_session_map)

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        return await self._bm25.add(namespace=namespace, messages=messages)

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
        # Stage 1: BM25 recall — fetch more candidates than needed. The BM25
        # stage applies the ego / query-day / excluded-thread scope.
        first_stage_k = max(_BM25_FIRST_STAGE_K, top_k)
        candidates = await self._bm25.search(
            namespace=namespace,
            query=query,
            top_k=first_stage_k,
            ego_id=ego_id,
            query_timestamp=query_timestamp,
            exclude_thread_ids=exclude_thread_ids,
        )

        if not candidates:
            return []

        # If we have fewer candidates than top_k, skip reranking
        if len(candidates) <= top_k:
            return candidates

        # Stage 2: Cross-encoder rerank
        self._ensure_reranker()

        pairs = [[query, hit.text] for hit in candidates]
        rerank_scores = self._reranker.compute_score(pairs, normalize=True)

        # compute_score returns a single float when given one pair
        if isinstance(rerank_scores, (int, float)):
            rerank_scores = [rerank_scores]

        # Pair candidates with rerank scores and sort descending
        scored = list(zip(candidates, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top-k
        top = scored[:max(1, int(top_k))]

        if not top:
            return []

        max_s = top[0][1]
        min_s = top[-1][1] if len(top) > 1 else max_s

        results = []
        for hit, score in top:
            results.append(SearchHit(
                msg_id=hit.msg_id,
                score=normalize_score(score, min_value=min_s, max_value=max_s),
                text=hit.text,
                occur_ts=hit.occur_ts,
                thread_id=hit.thread_id,
                user_id=hit.user_id,
            ))

        return results

    async def reset(self, *, namespace: str) -> None:
        await self._bm25.reset(namespace=namespace)

    async def close(self) -> None:
        self._reranker = None
        await self._bm25.close()
