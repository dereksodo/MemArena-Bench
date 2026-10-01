"""Tests for frozen memory-cache construction."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from eval.src.types import SearchHit
from scripts.reproduce import build_memory_cache as bmc


class _FakeMemoryAdapter:
    def __init__(self) -> None:
        self.reset_namespaces: list[str] = []
        self.add_namespaces: list[str] = []
        self.search_namespaces: list[str] = []

    async def reset(self, *, namespace: str) -> None:
        self.reset_namespaces.append(namespace)
        return None

    async def add(self, *, namespace: str, messages: list) -> dict:
        self.add_namespaces.append(namespace)
        return {"n_batches": 1}

    async def search(self, *, namespace: str, query: str, top_k: int) -> list[SearchHit]:
        self.search_namespaces.append(namespace)
        return [
            SearchHit(
                msg_id="m1",
                score=1.0,
                text=f"cached for {query}",
                occur_ts="0.0",
                thread_id="s1",
                user_id=None,
            )
        ]

    async def close(self) -> None:
        return None


def _write_minimal_run(run_dir: Path) -> None:
    (run_dir / "eval_instances").mkdir(parents=True)
    (run_dir / "corpus_sessions.jsonl").write_text(
        json.dumps({
            "session_id": "s1",
            "participants": ["alice", "bob"],
            "turns": [
                {
                    "turn_id": "s1_t1",
                    "speaker_id": "alice",
                    "listener_id": "bob",
                    "text": "Alice remembers the blue notebook.",
                    "timestamp": 0.0,
                }
            ],
        }) + "\n",
        encoding="utf-8",
    )
    (run_dir / "ego_session_map.json").write_text(
        json.dumps({"alice": ["s1"]}),
        encoding="utf-8",
    )
    rows = [
        {
            "instance_id": "qa_answerer_fallback",
            "query": "What does Alice remember?",
            "answerer_agent_id": "alice",
            "metadata": {"query_timestamp": 0.2},
        },
        {
            "instance_id": "qa_missing_ego",
            "query": "Who can answer this?",
            "metadata": {"query_timestamp": 0.2},
        },
    ]
    (run_dir / "eval_instances" / "d1_test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_build_cache_writes_error_rows_for_unscheduled_instances(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    out_path = tmp_path / "cache.jsonl"
    adapter = _FakeMemoryAdapter()

    monkeypatch.setattr(
        bmc,
        "build_adapter_for_extraction",
        lambda **_: adapter,
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output=str(out_path),
        qa_limit=0,
        user_limit=0,
        namespace_prefix=None,
        qdrant_path=str(tmp_path / "qdrant"),
        vanilla=False,
        system="memobase",
        extractor_model="reader",
        extractor_endpoint="http://localhost:16000",
        extractor_api_key="EMPTY",
        extractor_config="A_paired",
        concurrency=4,
        max_tokens=128,
        top_k=10,
        trial_seed=1002,
        hw_marker_file=None,
        verbose=False,
    )

    asyncio.run(bmc.build_cache(args))

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["instance_id"]: row for row in rows}

    assert set(by_id) == {"qa_answerer_fallback", "qa_missing_ego"}
    assert by_id["qa_answerer_fallback"]["error"] is None
    assert by_id["qa_answerer_fallback"]["namespace"] == "alice"
    assert "storage_namespace" not in by_id["qa_answerer_fallback"]
    assert by_id["qa_missing_ego"]["memories"] is None
    assert "missing ego/query-agent identity" in by_id["qa_missing_ego"]["error"]
    assert adapter.reset_namespaces == ["alice"]
    assert adapter.add_namespaces == ["alice"]
    assert adapter.search_namespaces == ["alice"]
