from __future__ import annotations

from typing import Any

from .base import RetrievalAdapter
from ..types import MessageEntry, SearchHit


class NoopAdapter(RetrievalAdapter):
    """No-op adapter: keeps stage contracts but performs no retrieval."""

    async def add(self, *, namespace: str, messages: list[MessageEntry]) -> dict:
        return {
            "namespace": namespace,
            "indexed_messages": len(messages),
            "mode": "noop",
        }

    async def search(self, *, namespace: str, query: str, top_k: int, **_: Any) -> list[SearchHit]:
        # Scope kwargs (ego_id, query_timestamp, ...) are ignored: the
        # TEXT_SESSIONS answer prompt is built from session ids, not hits.
        return []
