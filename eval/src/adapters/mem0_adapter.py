from __future__ import annotations

import os
import time
from typing import Any, List, Optional

from .base import normalize_score
from .http_base import HttpRetrievalAdapter
from ..types import MessageEntry, SearchHit


class Mem0Adapter(HttpRetrievalAdapter):
    """HTTP-compatible Mem0 adapter (endpoint paths configurable)."""

    def __init__(self, *, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        base_url = str(cfg.get("base_url") or os.getenv("MEM0_BASE_URL") or "https://api.mem0.ai").strip()
        api_key = str(cfg.get("api_key") or os.getenv("MEM0_API_KEY") or "").strip() or None
        if not api_key:
            raise ValueError("Mem0Adapter requires api_key (cfg.mem0.api_key or MEM0_API_KEY)")
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_s=int(cfg.get("timeout_s", 30)),
            max_retries=int(cfg.get("max_retries", 3)),
        )
        self.add_path = str(cfg.get("add_path", "/v1/memories"))
        self.search_path = str(cfg.get("search_path", "/v1/memories/search"))

    def _build_add_payload(self, namespace: str, message: MessageEntry) -> dict:
        return {
            "run_id": namespace,
            "messages": [
                {
                    "role": "user",
                    "name": str(message.user_id) if message.user_id is not None else "unknown",
                    "content": message.text,
                }
            ],
            "timestamp": message.occur_ts,
        }

    def _build_search_payload(self, namespace: str, query: str, top_k: int) -> dict:
        return {
            "query": query,
            "top_k": int(top_k),
            "filters": {"run_id": namespace},
            "version": "v2",
        }

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        t0 = time.time()
        for m in messages:
            await self._request_with_retry("POST", self.add_path, json_body=self._build_add_payload(namespace, m))
        return {
            "namespace": namespace,
            "indexed_messages": len(messages),
            "latency_ms": int((time.time() - t0) * 1000),
            "backend_meta": {"adapter": "mem0"},
        }

    async def search(self, *, namespace: str, query: str, top_k: int, **_: Any) -> List[SearchHit]:
        # Scope kwargs (ego_id, query_timestamp, ...) are ignored: the store
        # is namespaced per ego at ingest and holds what was ingested.
        data = await self._request_with_retry("POST", self.search_path, json_body=self._build_search_payload(namespace, query, top_k))
        rows = data.get("results") or data.get("hits") or []
        rows = rows[: max(1, int(top_k))]
        scores = [float(r.get("score", 0.0) or 0.0) for r in rows] or [0.0]
        lo, hi = min(scores), max(scores)

        out: List[SearchHit] = []
        for i, r in enumerate(rows):
            out.append(
                SearchHit(
                    msg_id=str(r.get("id") or r.get("msg_id") or f"mem0_{i}"),
                    score=normalize_score(float(r.get("score", 0.0) or 0.0), min_value=lo, max_value=hi),
                    text=str(r.get("memory") or r.get("text") or ""),
                    occur_ts=str(r.get("created_at") or r.get("occur_ts") or ""),
                    thread_id=str(r.get("run_id") or namespace),
                    user_id=None,
                )
            )
        return out
