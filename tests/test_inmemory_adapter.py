"""Unit coverage for the BM25 in-memory adapter."""
from __future__ import annotations

import asyncio

from eval.src.adapters.inmemory import InMemoryAdapter
from eval.src.types import MessageEntry


def _msg(msg_id: str, text: str, ts: float, thread_id: str) -> MessageEntry:
    return MessageEntry(
        msg_id=msg_id,
        occur_ts=str(ts),
        deliver_ts=str(ts),
        user_id=None,
        thread_id=thread_id,
        text=text,
    )


def test_inmemory_time_scoped_search_does_not_retrieve_future_days() -> None:
    adapter = InMemoryAdapter()
    adapter.set_ego_session_map({"alice": ["day0", "day1"]})

    async def run() -> list[str]:
        await adapter.add(
            namespace="rag",
            messages=[
                _msg("m0", "Alice keeps a blue notebook.", 0.2, "day0"),
                _msg("m1", "Alice will hide the red notebook tomorrow.", 1.2, "day1"),
            ],
        )
        hits = await adapter.search(
            namespace="rag",
            query="red notebook",
            top_k=3,
            ego_id="alice",
            query_timestamp=0.9,
        )
        return [h.text for h in hits]

    texts = asyncio.run(run())

    assert texts
    assert all("red notebook" not in text for text in texts)
    assert any("blue notebook" in text for text in texts)


def test_inmemory_scoped_search_honors_ego_and_excluded_threads() -> None:
    adapter = InMemoryAdapter()
    adapter.set_ego_session_map({"alice": ["a1", "a2"], "bob": ["b1"]})

    async def run() -> list[str]:
        await adapter.add(
            namespace="rag",
            messages=[
                _msg("a1m", "Alice knows the garden code.", 0.1, "a1"),
                _msg("a2m", "Alice knows the vault code.", 0.2, "a2"),
                _msg("b1m", "Bob knows the garden code.", 0.1, "b1"),
            ],
        )
        hits = await adapter.search(
            namespace="rag",
            query="garden code",
            top_k=5,
            ego_id="alice",
            query_timestamp=0.5,
            exclude_thread_ids=["a1"],
        )
        return [h.text for h in hits]

    texts = asyncio.run(run())

    assert all("Bob knows" not in text for text in texts)
    assert all("garden code" not in text for text in texts)
    assert any("vault code" in text for text in texts)

