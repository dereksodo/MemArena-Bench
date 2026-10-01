"""Dry-run test for `scripts/run_accuracy.py`.

Exercises the accuracy pipeline against the mini_eval_instances fixture
with both reader and judge mocked. Verifies that the two output files
exist, carry the expected keys, and that per-dimension accuracy is
computed over the input dimensions.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_accuracy import (
    _build_eval_cli_cmd,
    _judge_passes,
    _normalise_openai_base,
    _redact_cmd,
    build_arg_parser,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "mini_eval_instances.jsonl"


def _run(out_dir: Path, extra: list[str] | None = None) -> None:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "run_accuracy.py"),
        "--dry-run",
        "--instances", str(FIXTURE),
        "--out-dir", str(out_dir),
        "--n", "10",
    ]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def _latest_subdir(root: Path) -> Path:
    subs = [p for p in root.iterdir() if p.is_dir()]
    assert subs, f"no subdir under {root}"
    return max(subs, key=lambda p: p.stat().st_mtime)


def test_accuracy_dryrun_vanilla(tmp_path: Path) -> None:
    out_root = tmp_path / "acc_out"
    _run(out_root, extra=["--backend", "vanilla"])
    run_dir = _latest_subdir(out_root)

    results = json.loads((run_dir / "accuracy_results.json").read_text())
    summary = json.loads((run_dir / "accuracy_summary.json").read_text())

    assert isinstance(results, list) and results, "accuracy_results.json is empty"
    for r in results:
        assert {"instance_id", "dimension", "question", "gold",
                "model_answer", "score"}.issubset(r)
        assert r["model_answer"].startswith("[dry-run]")
        assert r["score"] in (0, 1)

    assert summary["dry_run"] is True
    assert summary["backend"] == "vanilla"
    assert summary["n"] == len(results)
    assert 0.0 <= summary["overall_accuracy"] <= 1.0
    assert summary["per_dimension_accuracy"], "per_dim dict should not be empty"
    for dim, acc in summary["per_dimension_accuracy"].items():
        assert 0.0 <= acc <= 1.0, (dim, acc)


def test_accuracy_dryrun_oracle(tmp_path: Path) -> None:
    out_root = tmp_path / "acc_out_oracle"
    _run(out_root, extra=["--backend", "oracle"])
    run_dir = _latest_subdir(out_root)

    summary = json.loads((run_dir / "accuracy_summary.json").read_text())
    assert summary["backend"] == "oracle"
    assert summary["dry_run"] is True


def test_accuracy_dryrun_is_deterministic(tmp_path: Path) -> None:
    """Two identical dry-run invocations must yield identical scores."""
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    _run(a, extra=["--backend", "vanilla", "--seed", "7"])
    _run(b, extra=["--backend", "vanilla", "--seed", "7"])

    def scores(root: Path) -> list[int]:
        sub = _latest_subdir(root)
        return [r["score"] for r in json.loads(
            (sub / "accuracy_results.json").read_text())]

    assert scores(a) == scores(b), "dry-run judge output depends on seed"


def test_production_endpoint_normalisation() -> None:
    assert _normalise_openai_base("http://localhost:8000") == "http://localhost:8000/v1"
    assert _normalise_openai_base("http://localhost:8000/v1") == "http://localhost:8000/v1"


def test_production_both_judge_passes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    args = build_arg_parser().parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--judge-preset", "both",
        "--sglang-url", "http://127.0.0.1:8000",
        "--model-name", "qwen3",
        "--out-dir", str(tmp_path / "out"),
    ])

    passes = _judge_passes(args)
    assert [p.tag for p in passes] == ["remote", "local"]

    cmd, env, namespace = _build_eval_cli_cmd(args, passes[1])
    assert namespace.endswith("_judge_local")
    assert cmd[cmd.index("--namespace") + 1] == namespace
    assert namespace not in cmd[cmd.index("--namespace") + 2:]
    assert "--judge-model" in cmd
    assert "qwen3" in cmd
    assert "http://127.0.0.1:8000/v1" in cmd
    assert env == {}


def test_local_judge_does_not_reuse_remote_key(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--judge-preset", "both",
        "--judge-api-key", "sk-openai-secret",
        "--sglang-url", "http://127.0.0.1:8000",
        "--model-name", "qwen3",
        "--out-dir", str(tmp_path / "out"),
    ])

    passes = _judge_passes(args)
    cmd, env, _namespace = _build_eval_cli_cmd(args, passes[1])

    assert passes[1].tag == "local"
    assert "sk-openai-secret" not in cmd
    assert cmd[cmd.index("--judge-api-key") + 1] == "EMPTY"
    assert env == {}


def test_production_test_mode_preserves_retrieval_stages_without_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = build_arg_parser().parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--system", "inmem",
        "--judge-preset", "remote",
        "--model-name", "qwen3",
        "--model-tag", "8b",
        "--test",
        "--out-dir", str(tmp_path / "out"),
    ])

    passes = _judge_passes(args)
    cmd, env, namespace = _build_eval_cli_cmd(args, passes[0])

    assert namespace == "inmem_8b_judge_remote"
    assert cmd[cmd.index("--stages") + 1:cmd.index("--model")] == [
        "add", "search", "answer", "evaluate",
    ]
    assert "--test" in cmd
    assert env == {}


def test_production_namespace_includes_trial_name_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = build_arg_parser().parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--system", "memos",
        "--judge-preset", "remote",
        "--model-name", "Qwen/Qwen3-8B",
        "--model-tag", "8b",
        "--trial-name", "s3",
        "--test",
        "--out-dir", str(tmp_path / "out"),
    ])

    passes = _judge_passes(args)
    cmd, env, namespace = _build_eval_cli_cmd(args, passes[0])

    assert namespace == "memos_8b_s3_judge_remote"
    assert cmd[cmd.index("--namespace") + 1] == namespace
    assert env == {}


def test_memory_cache_production_passes_cache_args(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cache_path = tmp_path / "cache.jsonl"
    args = build_arg_parser().parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--system", "memory_cache",
        "--cache-path", str(cache_path),
        "--expected-extractor", "Qwen/Qwen3-8B",
        "--expected-memory-system", "memos",
        "--expected-config", "A_paired",
        "--judge-preset", "remote",
        "--model-name", "Qwen/Qwen3-8B",
        "--model-tag", "8b",
        "--trial-name", "s3",
        "--test",
        "--out-dir", str(tmp_path / "out"),
    ])

    passes = _judge_passes(args)
    cmd, env, namespace = _build_eval_cli_cmd(args, passes[0])

    assert namespace == "memory_cache_8b_s3_judge_remote"
    assert cmd[cmd.index("--stages") + 1:cmd.index("--model")] == [
        "search", "answer", "evaluate",
    ]
    assert cmd[cmd.index("--cache-path") + 1] == str(cache_path)
    assert cmd[cmd.index("--expected-memory-system") + 1] == "memos"
    assert env == {}


def test_answer_only_memory_cache_keeps_search_stage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cache_path = tmp_path / "cache.jsonl"
    args = build_arg_parser().parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--system", "memory_cache",
        "--cache-path", str(cache_path),
        "--expected-extractor", "Qwen/Qwen3-8B",
        "--expected-memory-system", "memos",
        "--expected-config", "A_paired",
        "--judge-preset", "remote",
        "--model-name", "Qwen/Qwen3-8B",
        "--model-tag", "8b",
        "--trial-name", "s3",
        "--out-dir", str(tmp_path / "out"),
    ])

    cmd, env, namespace = _build_eval_cli_cmd(args, None)

    assert namespace == "memory_cache_8b_s3"
    assert cmd[cmd.index("--stages") + 1:cmd.index("--model")] == ["search", "answer"]
    assert cmd[cmd.index("--cache-path") + 1] == str(cache_path)
    assert env == {}


def test_answer_only_inmem_keeps_ingest_and_search_stages(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--system", "inmem",
        "--model-name", "Qwen/Qwen3-8B",
        "--model-tag", "8b",
        "--trial-name", "s2",
        "--out-dir", str(tmp_path / "out"),
    ])

    cmd, _env, namespace = _build_eval_cli_cmd(args, None)

    assert namespace == "inmem_8b_s2"
    assert cmd[cmd.index("--stages") + 1:cmd.index("--model")] == ["add", "search", "answer"]


def test_memory_cache_replay_uses_instance_id_lookup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_dir = tmp_path / "masim"
    eval_dir = run_dir / "eval_instances"
    eval_dir.mkdir(parents=True)
    (run_dir / "corpus_sessions.jsonl").write_text(
        json.dumps({
            "session_id": "sess_1",
            "participants": ["alice", "bob"],
            "turns": [
                {
                    "turn_id": "sess_1_t1",
                    "session_id": "sess_1",
                    "speaker_id": "alice",
                    "listener_id": "bob",
                    "text": "Alice keeps the blue notebook.",
                    "timestamp": 0.0,
                    "metadata": {},
                }
            ],
            "modality": "text_message",
        }) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "d7_qa.jsonl").write_text(
        json.dumps({
            "instance_id": "qa_1",
            "dimension": "d7_qa",
            "query": "What notebook does Alice keep?",
            "ground_truth": {"answer": "The blue notebook."},
            "evidence_session_ids": ["sess_1"],
            "ego_agent_id": "alice",
            "metadata": {"query_agent": "alice"},
        }) + "\n",
        encoding="utf-8",
    )
    cache_path = tmp_path / "memcache.jsonl"
    cache_path.write_text(
        json.dumps({
            "instance_id": "qa_1",
            "namespace": "alice",
            "storage_namespace": "memos_8b_s3_alice",
            "query": "What notebook does Alice keep?",
            "system": "memos",
            "config": "A_paired",
            "extractor_model": "Qwen/Qwen3-8B",
            "memories": [
                {
                    "id": "m1",
                    "text": "Alice keeps the blue notebook.",
                    "score": 1.0,
                    "occur_ts": "0.0",
                    "thread_id": "sess_1",
                }
            ],
            "memory_count": 1,
            "error": None,
        }) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_accuracy.py"),
            "--run-dir", str(run_dir),
            "--system", "memory_cache",
            "--cache-path", str(cache_path),
            "--expected-extractor", "Qwen/Qwen3-8B",
            "--expected-memory-system", "memos",
            "--expected-config", "A_paired",
            "--judge-preset", "remote",
            "--judge-inline",
            "--model-name", "Qwen/Qwen3-8B",
            "--model-tag", "8b",
            "--trial-name", "s3",
            "--out-dir", str(out_dir),
            "--test",
            "--no-progress",
        ],
        check=True,
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )

    search_path = (
        out_dir / "eval_results_s3" / "memory_cache"
        / "search_results_memory_cache_8b_s3_judge_remote.json"
    )
    rows = json.loads(search_path.read_text(encoding="utf-8"))
    assert rows[0]["hits"][0]["text"] == "Alice keeps the blue notebook."


def test_production_test_mode_writes_artifacts_without_llm(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_dir = tmp_path / "masim"
    eval_dir = run_dir / "eval_instances"
    eval_dir.mkdir(parents=True)
    (run_dir / "corpus_sessions.jsonl").write_text(
        json.dumps({
            "session_id": "sess_1",
            "participants": ["alice", "bob"],
            "turns": [
                {
                    "turn_id": "sess_1_t1",
                    "session_id": "sess_1",
                    "speaker_id": "alice",
                    "listener_id": "bob",
                    "text": "Alice said hello.",
                    "timestamp": 0.0,
                    "metadata": {},
                }
            ],
            "modality": "text_message",
        }) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "d7_qa.jsonl").write_text(
        json.dumps({
            "instance_id": "qa_1",
            "dimension": "d7_qa",
            "query": "What did Alice say?",
            "ground_truth": {"answer": "Alice said hello."},
            "evidence_session_ids": ["sess_1"],
            "ego_agent_id": "alice",
            "metadata": {"query_agent": "alice"},
        }) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_accuracy.py"),
            "--run-dir", str(run_dir),
            "--system", "memobase",
            "--judge-preset", "remote",
            "--judge-inline",
            "--model-name", "Qwen/Qwen3-8B",
            "--model-tag", "8b",
            "--trial-name", "s2",
            "--out-dir", str(out_dir),
            "--test",
            "--no-progress",
        ],
        check=True,
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )

    answer_path = out_dir / "eval_results_s2" / "memobase" / "answer_results_memobase_8b_s2_judge_remote.json"
    eval_path = out_dir / "eval_results_s2" / "memobase" / "evaluation_results_memobase_8b_s2_judge_remote.json"
    args_path = out_dir / "eval_results_s2" / "memobase" / "runs" / "latest_memobase_8b_s2_judge_remote.txt"
    assert answer_path.exists()
    assert eval_path.exists()
    assert args_path.exists()

    answers = json.loads(answer_path.read_text(encoding="utf-8"))
    assert len(answers) == 1
    assert answers[0]["prediction"] == "I don't know"
    assert answers[0]["raw_response"] == "I don't know"
    assert answers[0]["model"].endswith(":test-stub")


def test_inmem_production_uses_non_interleaved_batched_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_dir = tmp_path / "masim"
    eval_dir = run_dir / "eval_instances"
    eval_dir.mkdir(parents=True)
    (run_dir / "corpus_sessions.jsonl").write_text(
        json.dumps({
            "session_id": "sess_1",
            "participants": ["alice", "bob"],
            "turns": [
                {
                    "turn_id": "sess_1_t1",
                    "session_id": "sess_1",
                    "speaker_id": "alice",
                    "listener_id": "bob",
                    "text": "Alice said hello.",
                    "timestamp": 0.0,
                    "metadata": {},
                }
            ],
            "modality": "text_message",
        }) + "\n",
        encoding="utf-8",
    )
    (run_dir / "ego_session_map.json").write_text(
        json.dumps({"alice": ["sess_1"], "bob": ["sess_1"]}),
        encoding="utf-8",
    )
    (eval_dir / "d7_qa.jsonl").write_text(
        json.dumps({
            "instance_id": "qa_1",
            "dimension": "d7_qa",
            "query": "What did Alice say?",
            "ground_truth": {"answer": "Alice said hello."},
            "evidence_session_ids": ["sess_1"],
            "ego_agent_id": "alice",
            "metadata": {"query_agent": "alice", "query_timestamp": 0.0},
        }) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_accuracy.py"),
            "--run-dir", str(run_dir),
            "--system", "inmem",
            "--judge-preset", "remote",
            "--judge-inline",
            "--model-name", "Qwen/Qwen3-8B",
            "--model-tag", "8b",
            "--trial-name", "s2",
            "--out-dir", str(out_dir),
            "--test",
            "--no-progress",
        ],
        check=True,
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )

    result_dir = out_dir / "eval_results_s2" / "inmem"
    io_path = result_dir / "input_output_inmem_8b_s2_judge_remote.json"
    search_path = result_dir / "search_results_inmem_8b_s2_judge_remote.json"
    assert io_path.exists()
    assert search_path.exists()

    io_payload = json.loads(io_path.read_text(encoding="utf-8"))
    search_rows = json.loads(search_path.read_text(encoding="utf-8"))
    assert io_payload["mode"] == "non_interleaved"
    assert "Alice said hello." in search_rows[0]["hits"][0]["text"]


def test_legacy_judge_aliases_are_accepted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    remote_args = build_arg_parser().parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--judge-preset", "gpt4omini",
    ])
    local_args = build_arg_parser().parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--judge-preset", "qwen3_8b",
    ])

    assert _judge_passes(remote_args)[0].tag == "remote"
    assert _judge_passes(local_args)[0].tag == "local"


def test_command_redaction_hides_keys() -> None:
    cmd = ["python", "-m", "eval.cli", "--api-key", "secret", "--judge-api-key", "judge-secret"]
    assert "secret" not in _redact_cmd(cmd)
    assert "judge-secret" not in _redact_cmd(cmd)
    assert _redact_cmd(cmd).count("<redacted>") == 2
