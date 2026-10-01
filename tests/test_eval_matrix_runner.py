"""Dry-run coverage for the backend x model x trial matrix launcher."""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_eval_matrix.sh"


def test_eval_matrix_help() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "model_tag|model_name|endpoint" in result.stdout
    assert "memobase_url|memos_url" in result.stdout
    assert "--print-only" in result.stdout
    assert "--model-parallelism" in result.stdout


def test_eval_matrix_print_only_expands_cells(tmp_path: Path) -> None:
    run_dir = tmp_path / "masim_run"
    run_dir.mkdir()
    models_file = tmp_path / "models.tsv"
    models_file.write_text(
        "# model_tag|model_name|endpoint\n"
        "qwen3|qwen3|http://localhost:8000\n"
        "mini|mini-model|http://localhost:8001\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(SCRIPT),
            "--run-dir", str(run_dir),
            "--models-file", str(models_file),
            "--backends", "oracle,vanilla",
            "--trials", "s1,s2",
            "--judge-preset", "local",
            "--answer-concurrency", "2",
            "--eval-concurrency", "3",
            "--model-parallelism", "2",
            "--out-dir", str(tmp_path / "accuracy"),
            "--print-only",
        ],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
    )

    out = result.stdout
    assert "2 backends x 2 models x 2 trials = 8" in out
    assert "model parallelism: 2" in out
    assert out.count("[run-eval-matrix] [") == 8
    assert "--system oracle" in out
    assert "--model-tag qwen3" in out
    assert "--sglang-url http://localhost:8001" in out
    assert "--trial-name s1" in out
    assert "--trial-name s2" in out
    assert "--namespace oracle_qwen3_s1" in out
    assert "--namespace vanilla_mini_s2" in out
    assert "--seed 1001" in out
    assert "--seed 1002" in out
    assert "--answer-concurrency 2" in out
    assert "--eval-concurrency 3" in out


def test_eval_matrix_paper_full_test_preset_expands_75_cells(tmp_path: Path) -> None:
    run_dir = tmp_path / "masim_run"
    run_dir.mkdir()

    result = subprocess.run(
        [
            str(SCRIPT),
            "--paper-full",
            "--test",
            "--run-dir", str(run_dir),
            "--out-dir", str(tmp_path / "accuracy"),
            "--print-only",
        ],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
    )

    out = result.stdout
    assert "5 backends x 5 models x 3 trials = 75" in out
    assert "model parallelism: 5" in out
    assert out.count("[run-eval-matrix] [") == 75
    assert "--system memobase" in out
    assert "--system memos" in out
    assert "--model-tag 0_6b" in out
    assert "--model-tag llama3b" in out
    assert "--model-tag 32b" in out
    assert "--trial-name s2" in out
    assert "--trial-name s4" in out
    assert "--judge-preset remote" in out
    assert "--test" in out
    assert "--namespace memos_32b_s4" in out
    assert "MEMOBASE_BASE_URL=http://localhost:18108" in out
    assert "MEMOS_BASE_URL=http://localhost:18109" in out

    manifest = tmp_path / "accuracy" / "run_eval_matrix_manifest.tsv"
    rows = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 76  # header + 75 cells


def test_eval_matrix_qwen_0_6b_quick_preset_expands_two_cells(tmp_path: Path) -> None:
    run_dir = tmp_path / "masim_run"
    run_dir.mkdir()

    result = subprocess.run(
        [
            str(SCRIPT),
            "--run-dir", str(run_dir),
            "--models-file", "config/eval_matrix_qwen3_0_6b_models.tsv",
            "--backends", "vanilla,oracle",
            "--trials", "s2",
            "--judge-preset", "remote",
            "--out-dir", str(tmp_path / "accuracy"),
            "--print-only",
        ],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
    )

    out = result.stdout
    assert "2 backends x 1 models x 1 trials = 2" in out
    assert out.count("[run-eval-matrix] [") == 2
    assert "--model-tag 0_6b" in out
    assert "--model-name Qwen/Qwen3-0.6B" in out
    assert "--sglang-url http://localhost:16000" in out
    assert "--system vanilla" in out
    assert "--system oracle" in out
    assert "--answer-concurrency 32" in out
    assert "--eval-concurrency 512" in out


def test_eval_matrix_accepts_process_substitution_models_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "masim_run"
    run_dir.mkdir()

    command = (
        f"{shlex.quote(str(SCRIPT))} "
        f"--run-dir {shlex.quote(str(run_dir))} "
        "--models-file <(printf '%s\\n' "
        "'qwen3|qwen3|http://localhost:8000|http://localhost:18019|http://localhost:18020') "
        "--backends vanilla "
        "--trials s2 "
        "--judge-preset remote "
        f"--out-dir {shlex.quote(str(tmp_path / 'accuracy'))} "
        "--print-only"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "1 backends x 1 models x 1 trials = 1" in result.stdout
    assert "--model-tag qwen3" in result.stdout


def test_eval_matrix_routes_structured_memory_urls_per_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "masim_run"
    run_dir.mkdir()
    models_file = tmp_path / "models.tsv"
    models_file.write_text(
        "# model_tag|model_name|endpoint|memobase_url|memos_url\n"
        "qwen3|qwen3|http://localhost:8000|http://localhost:18019|http://localhost:18020\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(SCRIPT),
            "--run-dir", str(run_dir),
            "--models-file", str(models_file),
            "--backends", "memobase,memos",
            "--trials", "s2,s3",
            "--judge-preset", "remote",
            "--out-dir", str(tmp_path / "accuracy"),
            "--print-only",
        ],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
    )

    out = result.stdout
    assert "build_memory_cache.py --system memobase" in out
    assert "build_memory_cache.py --system memos" in out
    assert "--concurrency 32" in out
    assert "env MEMOBASE_BASE_URL=http://localhost:18019 MEMOBASE_API_TOKEN=secret" in out
    assert "env MEMOS_BASE_URL=http://localhost:18020 MEMOS_API_KEY=EMPTY" in out
    assert "--system memory_cache" in out
    assert "--cache-path" in out
    assert "--expected-memory-system memobase" in out
    assert "--expected-memory-system memos" in out
    assert "--namespace-prefix" not in out
    assert "--namespace memobase_qwen3_s2" in out
    assert "--namespace memobase_qwen3_s3" in out
    assert "--namespace memos_qwen3_s2" in out
    assert "--namespace memos_qwen3_s3" in out


def test_eval_matrix_parallel_test_mode_writes_manifest_and_logs(tmp_path: Path) -> None:
    run_dir = tmp_path / "masim_run"
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
    models_file = tmp_path / "models.tsv"
    models_file.write_text(
        "qwen3|qwen3|http://localhost:8000\n"
        "mini|mini-model|http://localhost:8001\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "accuracy"

    result = subprocess.run(
        [
            str(SCRIPT),
            "--run-dir", str(run_dir),
            "--models-file", str(models_file),
            "--backends", "oracle,vanilla",
            "--trials", "s1",
            "--judge-preset", "remote",
            "--model-parallelism", "2",
            "--qa-limit", "1",
            "--out-dir", str(out_dir),
            "--test",
        ],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "active-models=" in result.stdout
    assert "qwen3" in result.stdout
    assert "mini" in result.stdout
    manifest = out_dir / "run_eval_matrix_manifest.tsv"
    rows = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 5  # header + 2 models x 2 backends x 1 trial
    assert all(row.endswith("\t0") for row in rows[1:])
    assert (out_dir / "matrix_logs" / "qwen3" / "oracle_s1.log").exists()
    assert (out_dir / "matrix_logs" / "mini" / "vanilla_s1.log").exists()
