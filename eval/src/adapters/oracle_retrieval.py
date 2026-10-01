from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .base import PromptStyle, RetrievalAdapter
from ..types import MessageEntry, SearchHit


class OracleRetrievalAdapter(RetrievalAdapter):
    """Oracle retrieval adapter: returns all provided evidence sessions.

    The pipeline feeds only the ground-truth evidence sessions for each query,
    so this adapter simply stores and returns everything it receives.
    This represents an upper bound — perfect retrieval — isolating reasoning
    ability from retrieval quality.
    """

    @property
    def prompt_style(self) -> PromptStyle:
        return PromptStyle.TEXT_SESSIONS

    def __init__(self, *, max_messages: int = 2000) -> None:
        self.max_messages = max(1, int(max_messages))
        self._messages: Dict[str, List[MessageEntry]] = defaultdict(list)

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        self._messages[namespace].extend(messages)
        return {
            "namespace": namespace,
            "indexed_messages": len(self._messages[namespace]),
            "mode": "oracle_retrieval",
        }

    async def search(self, *, namespace: str, query: str, top_k: int, **_: Any) -> List[SearchHit]:
        # Scope kwargs (ego_id, query_timestamp, ...) are ignored: the
        # TEXT_SESSIONS answer prompt is built from session ids, not hits.
        rows = self._messages.get(namespace, [])
        if not rows:
            return []

        # Return all evidence context (bounded) in chronological order.
        selected = rows[-self.max_messages :]
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
