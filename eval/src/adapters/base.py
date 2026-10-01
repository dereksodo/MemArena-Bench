from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List

from ..types import MessageEntry, SearchHit


class PromptStyle(str, Enum):
    """How the answer engine should build prompts for this adapter."""
    JSON_CONTEXT = "json_context"       # pipeline default: JSON context prefix + JSON question
    TEXT_SESSIONS = "text_sessions"     # direct eval: text session headers, system+user messages


class RetrievalAdapter(ABC):
    # TEXT_SESSIONS adapters set this when the answer prompt must be built
    # from their search hits rather than from the gold evidence session ids.
    text_context_from_hits: bool = False

    @abstractmethod
    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        raise NotImplementedError

    @abstractmethod
    async def search(self, *, namespace: str, query: str, top_k: int, **kwargs: Any) -> List[SearchHit]:
        """Return the top_k hits for query. The pipeline also passes the
        item's scope (ego_id, query_timestamp, exclude_thread_ids, ...) as
        keyword arguments; an adapter must accept them and either apply them
        or document that it ignores them."""
        raise NotImplementedError

    @property
    def prompt_style(self) -> PromptStyle:
        return PromptStyle.JSON_CONTEXT

    async def health(self) -> dict:
        return {"ok": True, "detail": "not_implemented", "latency_ms": 0}

    async def reset(self, *, namespace: str) -> None:
        """Drop all data for a single namespace. Adapters that don't support
        this can leave the default no-op (sanity gates will skip the relevant
        check)."""
        return None

    async def close(self) -> None:
        return None


def normalize_score(score: float, *, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    v = (float(score) - float(min_value)) / (float(max_value) - float(min_value))
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)
