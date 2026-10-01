from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .base import PromptStyle, RetrievalAdapter
from ..types import MessageEntry, SearchHit


class VanillaAdapter(RetrievalAdapter):
    """Vanilla adapter: returns only the most recent messages up to a token cap.

    No retrieval or memory management — a lower-bound baseline representing
    what a context-window-only model can do. The model sees only the tail
    of the conversation history that fits within max_tokens.
    """

    @property
    def prompt_style(self) -> PromptStyle:
        return PromptStyle.TEXT_SESSIONS

    def __init__(self, *, max_tokens: int = 8192, max_messages: int = 2000) -> None:
        self.max_tokens = max(1, int(max_tokens))
        self.max_messages = max(1, int(max_messages))
        self._messages: Dict[str, List[MessageEntry]] = defaultdict(list)

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        self._messages[namespace].extend(messages)
        return {
            "namespace": namespace,
            "indexed_messages": len(self._messages[namespace]),
            "mode": "vanilla",
        }

    async def search(self, *, namespace: str, query: str, top_k: int, **_: Any) -> List[SearchHit]:
        # Scope kwargs (ego_id, query_timestamp, ...) are ignored: the
        # TEXT_SESSIONS answer prompt is built from session ids, not hits.
        rows = self._messages.get(namespace, [])
        if not rows:
            return []

        # Walk backward from most recent, accumulating until token budget exhausted.
        selected: list[MessageEntry] = []
        token_budget = self.max_tokens
        for m in reversed(rows):
            # Rough token estimate: 1 token ≈ 4 chars
            msg_tokens = len(m.text) // 4 + 1
            if token_budget - msg_tokens < 0 and selected:
                break
            selected.append(m)
            token_budget -= msg_tokens
            if len(selected) >= self.max_messages:
                break

        # Restore chronological order
        selected.reverse()
        return [
            SearchHit(
                msg_id=m.msg_id,
                score=1.0,
                text=m.text,
                occur_ts=m.occur_ts,
                thread_id=m.thread_id,
                user_id=m.user_id,
            )
            for m in selected
        ]
