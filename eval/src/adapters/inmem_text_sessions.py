"""InMemory (BM25) adapter with TEXT_SESSIONS prompt style.

Used for §27 P3: prompt-matched RAG × D5 control experiment.
Identical to InMemoryAdapter except prompt_style returns TEXT_SESSIONS
instead of JSON_CONTEXT, matching Oracle/Vanilla prompt formatting.
"""
from .inmemory import InMemoryAdapter
from .base import PromptStyle


class InMemTextSessionsAdapter(InMemoryAdapter):
    """BM25 retrieval with TEXT_SESSIONS prompt format (same as Oracle/Vanilla).

    The answer prompt renders the BM25 hits (with their session headers) in
    the TEXT_SESSIONS format; it never falls back to the gold evidence.
    """

    text_context_from_hits = True

    @property
    def prompt_style(self) -> PromptStyle:
        return PromptStyle.TEXT_SESSIONS
