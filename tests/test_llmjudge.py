from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts.llmjudge import (
    _jobs_from_progress,
    _load_env_file,
    _output_namespace,
    _resolve_judge,
    _run_jobs,
    build_arg_parser,
)


def _write_progress(progress_path: Path, log_path: Path) -> None:
    progress_path.write_text(
        json.dumps({
            "trial_paths": {
                "8b|memos|s2": [
                    {
                        "source_type": "matrix_log",
                        "status": "complete",
                        "path": str(log_path),
                        "seed": "1002",
                        "namespace": "memos_8b_s2",
                    }
                ]
            }
        }),
        encoding="utf-8",
    )


def test_progress_mode_finds_completed_answer_without_judge(tmp_path: Path) -> None:
    root = tmp_path / "accuracy"
    answer_dir = root / "eval_results_s2" / "memory_cache"
    answer_dir.mkdir(parents=True)
    answer_path = answer_dir / "answer_results_memos_8b_s2_judge_remote.json"
    answer_path.write_text("[]", encoding="utf-8")
    log_path = root / "matrix_logs" / "8b" / "memos_s2.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "[run-eval-matrix] [1/1] backend=memos model=8b trial=s2 seed=1002 namespace=memos_8b_s2\n"
        "[run-accuracy] completed 1 pass(es)\n",
        encoding="utf-8",
    )
    progress = tmp_path / "eval_progress_trial_paths.json"
    _write_progress(progress, log_path)

    jobs = _jobs_from_progress(
        progress_json=progress,
        judge_tag="remote",
        models=["8b"],
        backends=["memos"],
        trials=["s2"],
        only_missing_judge=True,
    )

    assert jobs == [answer_path.resolve()]


def test_progress_mode_finds_answer_only_namespace(tmp_path: Path) -> None:
    root = tmp_path / "accuracy"
    answer_dir = root / "eval_results_s2" / "memory_cache"
    answer_dir.mkdir(parents=True)
    answer_path = answer_dir / "answer_results_memos_8b_s2.json"
    answer_path.write_text("[]", encoding="utf-8")
    log_path = root / "matrix_logs" / "8b" / "memos_s2.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("[run-accuracy] completed 1 pass(es)\n", encoding="utf-8")
    progress = tmp_path / "eval_progress_trial_paths.json"
    _write_progress(progress, log_path)

    jobs = _jobs_from_progress(
        progress_json=progress,
        judge_tag="remote",
        models=["8b"],
        backends=["memos"],
        trials=["s2"],
        only_missing_judge=True,
    )

    assert jobs == [answer_path.resolve()]


def test_progress_mode_skips_existing_llm_judged_result(tmp_path: Path) -> None:
    root = tmp_path / "accuracy"
    answer_dir = root / "eval_results_s2" / "memory_cache"
    answer_dir.mkdir(parents=True)
    answer_path = answer_dir / "answer_results_memos_8b_s2_judge_remote.json"
    answer_path.write_text("[]", encoding="utf-8")
    eval_path = answer_dir / "evaluation_results_memos_8b_s2_judge_remote.json"
    eval_path.write_text(
        json.dumps({
            "details": [
                {
                    "answer_scored": True,
                    "reason": "evidence_judge: correct",
                }
            ]
        }),
        encoding="utf-8",
    )
    log_path = root / "matrix_logs" / "8b" / "memos_s2.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("[run-accuracy] completed 1 pass(es)\n", encoding="utf-8")
    progress = tmp_path / "eval_progress_trial_paths.json"
    _write_progress(progress, log_path)

    jobs = _jobs_from_progress(
        progress_json=progress,
        judge_tag="remote",
        models=["8b"],
        backends=["memos"],
        trials=["s2"],
        only_missing_judge=True,
    )

    assert jobs == []


def test_openrouter_default_concurrency_is_512() -> None:
    args = build_arg_parser().parse_args(["--answer-path", "answer_results_x.json"])
    assert args.concurrency == 512


def test_openrouter_requires_real_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # Stub out the .env loader so a developer's local .env (which often holds
    # a real OPENROUTER_API_KEY) does not silently overwrite the empty values
    # we just monkeypatched.
    import scripts.llmjudge as _mod
    monkeypatch.setattr(_mod, "_load_env_file", lambda *a, **kw: None)

    args = build_arg_parser().parse_args([
        "--answer-path", "answer_results_x.json",
        "--judge-preset", "openrouter",
    ])

    with pytest.raises(SystemExit, match="judge API key missing"):
        asyncio.run(_run_jobs(args))


def test_dotenv_loader_sets_openrouter_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY='sk-or-test'\n", encoding="utf-8")

    _load_env_file(env_file)
    args = build_arg_parser().parse_args(["--answer-path", "answer_results_x.json"])
    _tag, _model, _endpoint, api_key = _resolve_judge(args)

    assert api_key == "sk-or-test"


def test_non_remote_judge_gets_own_output_namespace() -> None:
    assert (
        _output_namespace("memos_32b_s2_judge_remote", "qwen235b", None)
        == "memos_32b_s2_judge_qwen235b"
    )
