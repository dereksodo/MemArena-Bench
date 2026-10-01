from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np

from .base import RetrievalAdapter, normalize_score
from .inmemory import _coerce_day, visible_at
from ..types import MessageEntry, SearchHit

logger = logging.getLogger(__name__)


class DenseE5Adapter(RetrievalAdapter):
    """Pure dense retriever using intfloat/e5-large-v2 embeddings with cosine similarity.

    Maintains per-ego embedding indexes so that search is restricted to
    sessions the ego agent participated in (ego-centric benchmark).
    Falls back to the global index if no ego_id is provided.

    E5 requires prefixing queries with "query: " and passages with "passage: ".
    """

    def __init__(self) -> None:
        self._model = None  # lazy-loaded

        # Global index: namespace -> list of messages
        self._messages: Dict[str, List[MessageEntry]] = defaultdict(list)
        # Global embeddings: namespace -> np.ndarray (N x 1024), built lazily
        self._embeddings: Dict[str, np.ndarray] = {}
        self._embeddings_dirty: Dict[str, bool] = defaultdict(lambda: True)

        # Per-ego structures
        self._ego_messages: Dict[str, Dict[str, List[MessageEntry]]] = defaultdict(lambda: defaultdict(list))
        self._ego_embeddings: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

        # Session -> participants mapping
        self._session_participants: Dict[str, Set[str]] = {}

    def _ensure_model(self):
        """Lazy-load e5-large-v2 model on first use."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading intfloat/e5-large-v2 model...")
        self._model = SentenceTransformer("intfloat/e5-large-v2")
        self._model.max_seq_length = 512
        logger.info("intfloat/e5-large-v2 model loaded.")

    def _encode_passages(self, texts: List[str]) -> np.ndarray:
        """Encode passage texts into dense embeddings. Returns (N, 1024) float32 array."""
        self._ensure_model()
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self._model.encode(prefixed, batch_size=256, normalize_embeddings=True)
        if not isinstance(vecs, np.ndarray):
            vecs = np.array(vecs, dtype=np.float32)
        return vecs.astype(np.float32)

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a single query into a dense embedding. Returns (1, 1024) float32 array."""
        self._ensure_model()
        prefixed = [f"query: {text}"]
        vecs = self._model.encode(prefixed, batch_size=1, normalize_embeddings=True)
        if not isinstance(vecs, np.ndarray):
            vecs = np.array(vecs, dtype=np.float32)
        return vecs.astype(np.float32)

    def set_ego_session_map(self, ego_session_map: Dict[str, List[str]]) -> None:
        """Set the ego -> session_ids mapping for ego-scoped indexing."""
        self._ego_session_map = ego_session_map

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        self._messages[namespace].extend(messages)
        self._embeddings_dirty[namespace] = True

        # Track session participants
        for m in messages:
            sid = m.thread_id
            if sid and sid not in self._session_participants:
                self._session_participants[sid] = set()

        return {
            "namespace": namespace,
            "indexed_messages": len(self._messages[namespace]),
        }

    def _ensure_global_embeddings(self, namespace: str) -> None:
        """Build global embedding index if dirty."""
        if not self._embeddings_dirty.get(namespace, True):
            return
        rows = self._messages.get(namespace, [])
        if not rows:
            return
        texts = [m.text or "" for m in rows]
        self._embeddings[namespace] = self._encode_passages(texts)
        self._embeddings_dirty[namespace] = False

    def _ensure_ego_index(self, namespace: str, ego_id: str) -> None:
        """Lazily build per-ego embedding index on first search."""
        if ego_id in self._ego_embeddings.get(namespace, {}):
            return

        ego_sids = set(getattr(self, "_ego_session_map", {}).get(ego_id, []))
        if not ego_sids:
            return

        ego_msgs = [m for m in self._messages.get(namespace, []) if m.thread_id in ego_sids]
        if not ego_msgs:
            return

        self._ego_messages[namespace][ego_id] = ego_msgs
        texts = [m.text or "" for m in ego_msgs]
        self._ego_embeddings[namespace][ego_id] = self._encode_passages(texts)

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
        # Same scope as the BM25 InMemoryAdapter: the ego's sessions, minus
        # excluded threads, delivered no later than the query day.
        cutoff_day = _coerce_day(query_timestamp)
        excluded = {str(x) for x in (exclude_thread_ids or []) if str(x)}

        # Try ego-scoped search first
        if ego_id and hasattr(self, "_ego_session_map"):
            self._ensure_ego_index(namespace, ego_id)
            ego_embs = self._ego_embeddings.get(namespace, {}).get(ego_id)
            ego_rows = self._ego_messages.get(namespace, {}).get(ego_id, [])
            if ego_embs is not None and ego_rows:
                if cutoff_day is None and not excluded:
                    return self._dense_search(ego_rows, ego_embs, query, top_k)
                keep = [i for i, m in enumerate(ego_rows) if visible_at(m, cutoff_day, excluded)]
                if not keep:
                    return []
                return self._dense_search([ego_rows[i] for i in keep], ego_embs[keep], query, top_k)

        # Fallback to global index
        rows = self._messages.get(namespace, [])
        if not rows:
            return []
        self._ensure_global_embeddings(namespace)
        embs = self._embeddings.get(namespace)
        if embs is None:
            return []
        return self._dense_search(rows, embs, query, top_k)

    def _dense_search(
        self,
        rows: List[MessageEntry],
        embeddings: np.ndarray,
        query: str,
        top_k: int,
    ) -> List[SearchHit]:
        query_emb = self._encode_query(query)  # (1, 1024)
        # Cosine similarity: embeddings are already L2-normalized
        scores = (embeddings @ query_emb.T).squeeze(-1)  # (N,)

        k = min(max(1, int(top_k)), len(scores))
        if len(scores) <= k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        if len(scores) == 0:
            return []

        max_s = float(scores[top_indices[0]])
        min_s = float(scores[top_indices[-1]]) if len(top_indices) > 1 else max_s

        results = []
        for idx in top_indices[:k]:
            s = float(scores[idx])
            m = rows[idx]
            results.append(SearchHit(
                msg_id=m.msg_id,
                score=normalize_score(s, min_value=min_s, max_value=max_s),
                text=m.text,
                occur_ts=m.occur_ts,
                thread_id=m.thread_id,
                user_id=m.user_id,
            ))

        # Fallback to recency if no meaningful results
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

    async def reset(self, *, namespace: str) -> None:
        self._messages.pop(namespace, None)
        self._embeddings.pop(namespace, None)
        self._embeddings_dirty.pop(namespace, None)
        self._ego_messages.pop(namespace, None)
        self._ego_embeddings.pop(namespace, None)

    async def close(self) -> None:
        self._model = None
