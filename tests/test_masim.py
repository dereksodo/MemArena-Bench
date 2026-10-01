"""Dry-run smoke test for `scripts/run_masim.py --smoke`.

Runs the MASim mini-pipeline in dry-run mode (no OpenAI calls), then
verifies that each stage emitted schema-valid JSONL output.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(out_dir: Path, extra: list[str] | None = None) -> None:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "run_masim.py"),
        "--smoke",
        "--dry-run",
        "--agents", "2",
        "--days", "1",
        "--output", str(out_dir),
    ]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def _read_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"missing output: {path}"
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_masim_smoke_dryrun(tmp_path: Path) -> None:
    run_dir = tmp_path / "masim_out"
    _run(run_dir)

    personas  = _read_jsonl(run_dir / "agents_personas.jsonl")
    schedules = _read_jsonl(run_dir / "agent_schedules.jsonl")
    sessions  = _read_jsonl(run_dir / "corpus_sessions.jsonl")
    evals     = _read_jsonl(run_dir / "eval_instances" / "d7_qa.jsonl")

    assert len(personas) == 2, personas
    assert len(schedules) == 2, schedules  # 2 agents × 1 day
    assert len(sessions) == 2, sessions    # 1 session per agent-day pair
    assert len(evals) >= 1, evals

    # Schema checks.
    for p in personas:
        assert {"agent_id", "name", "persona"}.issubset(p)
        assert p["persona"].startswith("[dry-run]"), p
    for s in sessions:
        assert {"session_id", "day", "participants", "turns"}.issubset(s)
        assert len(s["turns"]) >= 1
    for e in evals:
        assert {"instance_id", "dimension", "question", "gold_answer"}.issubset(e)

    report = json.loads((run_dir / "pipeline_report.json").read_text())
    assert report["dry_run"] is True
    assert report["counts"]["personas"] == 2


def test_masim_smoke_dryrun_scales(tmp_path: Path) -> None:
    """Scaling the agents/days flags proportionally scales output counts."""
    run_dir = tmp_path / "masim_out_big"
    _run(run_dir, extra=["--agents", "4", "--days", "2"])

    personas  = _read_jsonl(run_dir / "agents_personas.jsonl")
    schedules = _read_jsonl(run_dir / "agent_schedules.jsonl")
    sessions  = _read_jsonl(run_dir / "corpus_sessions.jsonl")

    assert len(personas) == 4
    assert len(schedules) == 4 * 2
    assert len(sessions) == 4 * 2
