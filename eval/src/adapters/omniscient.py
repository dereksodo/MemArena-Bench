from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .base import PromptStyle, RetrievalAdapter
from ..types import MessageEntry, SearchHit


class OmniscientAdapter(RetrievalAdapter):
    """Omniscient adapter: provides ALL corpus sessions as context.

    Used for the ego-centric vs omniscient ablation (TODO #12/#23).
    Unlike oracle (which returns only ground-truth evidence sessions) or
    vanilla (which returns only recent ego messages), this adapter returns
    the entire corpus — giving the model a god's-eye view of all dialogues.

    The pipeline detects 'omniscient' adapter_name and injects all session
    IDs into each instance's evidence_session_ids, bypassing ego filtering.
    """

    @property
    def prompt_style(self) -> PromptStyle:
        return PromptStyle.TEXT_SESSIONS

    def __init__(self, *, max_messages: int = 50000) -> None:
        self.max_messages = max(1, int(max_messages))
        self._messages: Dict[str, List[MessageEntry]] = defaultdict(list)

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        self._messages[namespace].extend(messages)
        return {
            "namespace": namespace,
            "indexed_messages": len(self._messages[namespace]),
            "mode": "omniscient",
        }

    async def search(self, *, namespace: str, query: str, top_k: int, **_: Any) -> List[SearchHit]:
        # Scope kwargs (ego_id, query_timestamp, ...) are ignored: the
        # TEXT_SESSIONS answer prompt is built from session ids, not hits.
        rows = self._messages.get(namespace, [])
        if not rows:
            return []
        selected = rows[-self.max_messages:]
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
