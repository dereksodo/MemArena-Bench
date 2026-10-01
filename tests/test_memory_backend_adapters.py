from __future__ import annotations

import asyncio

from eval.src.adapters.memobase_adapter import MemobaseAdapter
from eval.src.adapters.memos_adapter import MemosAdapter
from eval.src.types import MessageEntry


def _msg(*, speaker: str, text: str = "kept a blue notebook") -> MessageEntry:
    return MessageEntry(
        msg_id=f"{speaker}_1",
        occur_ts="1.25",
        deliver_ts="1.25",
        user_id=None,
        thread_id="session_1",
        text=text,
        meta={"speaker": speaker},
    )


def test_memos_payload_minimal_h200_shape() -> None:
    """MemOS payload follows the shape used for the paper runs: minimal fields,
    [Group: <tid>][Speaker: <user_id>] content prefix, no top_k in search."""
    adapter = MemosAdapter(cfg={"base_url": "http://localhost:8020"})

    add_payload = adapter._build_batch_payload(
        "abigail_ross",
        [_msg(speaker="min_jun_lee")],
    )
    search_payload = adapter._build_search_payload(
        "abigail_ross",
        "What notebook was mentioned?",
        7,
    )

    assert set(add_payload.keys()) == {"messages", "user_id", "async_mode"}
    assert add_payload["user_id"] == "abigail_ross"
    assert add_payload["async_mode"] == "sync"
    msg = add_payload["messages"][0]
    assert msg["role"] == "user"
    assert msg["chat_time"] == "1.25"
    assert msg["content"] == "[Group: session_1][Speaker: None] kept a blue notebook"

    assert set(search_payload.keys()) == {"query", "user_id"}
    assert search_payload["user_id"] == "abigail_ross"
    assert search_payload["query"] == "What notebook was mentioned?"


def test_memobase_blob_uses_per_speaker_alias_and_no_content_prefix() -> None:
    """Memobase blob payload uses meta['speaker'] as alias per message and
    sends raw text in content (no [Speaker: ...] prefix)."""
    adapter = MemobaseAdapter(
        cfg={"base_url": "http://localhost:8019", "api_key": "secret"}
    )

    captured: list = []

    async def fake_request(method, path, *, json_body=None, headers=None, timeout_s=None):
        captured.append({"method": method, "path": path, "json_body": json_body})
        if method == "POST" and path == "/api/v1/users":
            return {"data": {"id": "fake-uid-1"}}
        return {"data": {}}

    adapter._request_with_retry = fake_request  # type: ignore[assignment]

    asyncio.run(
        adapter.add(
            namespace="abigail_ross",
            messages=[
                _msg(speaker="abigail_ross", text="I keep a blue notebook."),
                _msg(speaker="min_jun_lee", text="Min-Jun borrowed a charger."),
            ],
        )
    )

    blob_calls = [c for c in captured if c["path"].startswith("/api/v1/blobs/insert/")]
    assert len(blob_calls) == 1
    msgs = blob_calls[0]["json_body"]["blob_data"]["messages"]
    assert [m["alias"] for m in msgs] == ["abigail_ross", "min_jun_lee"]
    assert msgs[0]["content"] == "I keep a blue notebook."
    assert msgs[1]["content"] == "Min-Jun borrowed a charger."


def test_memobase_reset_calls_delete_user() -> None:
    """The MemArena-specific reset() method DELETEs the memobase user (not
    inherited from the pre-release pipeline and preserved since)."""
    adapter = MemobaseAdapter(
        cfg={"base_url": "http://localhost:8019", "api_key": "secret"}
    )

    captured: list = []

    async def fake_request(method, path, *, json_body=None, headers=None, timeout_s=None):
        captured.append({"method": method, "path": path})
        if method == "POST" and path == "/api/v1/users":
            return {"data": {"id": "fake-uid-9"}}
        return {"data": {}}

    adapter._request_with_retry = fake_request  # type: ignore[assignment]

    async def _exercise() -> None:
        await adapter._ensure_user("abigail_ross")
        await adapter.reset(namespace="abigail_ross")

    asyncio.run(_exercise())

    delete_calls = [c for c in captured if c["method"] == "DELETE"]
    assert len(delete_calls) == 1
    assert delete_calls[0]["path"] == "/api/v1/users/fake-uid-9"
    assert "abigail_ross" not in adapter._user_ids
