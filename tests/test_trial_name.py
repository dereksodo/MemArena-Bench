"""Tests for user-configurable trial/seed naming.

External users who run the benchmark multiple times want to organise each
pass under an arbitrary trial name (e.g. s1, s2, s3). Figure scripts then
aggregate across those trials via the MEMARENA_SEEDS env var.

Two things to verify:
1. `run_accuracy.py --trial-name s1` writes to the canonical
   `<out-dir>/eval_results_s1/<backend>/` layout, not a timestamp directory.
2. `paper_data.SEEDS` honours `MEMARENA_SEEDS`; the default stays
   `["s2","s3","s4"]` to keep paper reproductions numerically identical.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "mini_eval_instances.jsonl"
PAPER_DATA = REPO / "memarena" / "figures" / "paper_data.py"


def test_run_accuracy_trial_name_layout(tmp_path: Path) -> None:
    out_dir = tmp_path / "acc"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_accuracy.py"),
         "--dry-run", "--trial-name", "s1",
         "--backend", "vanilla", "--n", "5",
         "--instances", str(FIXTURE),
         "--out-dir", str(out_dir)],
        check=True, cwd=str(REPO),
    )

    canonical = out_dir / "eval_results_s1" / "vanilla" / "accuracy_summary.json"
    assert canonical.exists(), f"expected canonical layout at {canonical}"

    summary = json.loads(canonical.read_text())
    assert summary["trial_name"] == "s1"
    assert summary["backend"] == "vanilla"
    assert summary["n"] == 5


@pytest.mark.skipif(
    not PAPER_DATA.exists(),
    reason="paper_data.py is gitignored; copy from MemArena to exercise",
)
def test_paper_data_seeds_default() -> None:
    """Default must stay s2,s3,s4 for paper reproducibility."""
    env = {k: v for k, v in os.environ.items() if k != "MEMARENA_SEEDS"}
    result = subprocess.run(
        [sys.executable, "-c",
         "from memarena.figures import paper_data; print(','.join(paper_data.SEEDS))"],
        check=True, cwd=str(REPO), env=env, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "s2,s3,s4", (
        f"paper default must match committed paper numbers; got {result.stdout!r}"
    )


@pytest.mark.skipif(
    not PAPER_DATA.exists(),
    reason="paper_data.py is gitignored; copy from MemArena to exercise",
)
def test_paper_data_seeds_env_override() -> None:
    """MEMARENA_SEEDS must replace the default for external-user aggregation."""
    env = dict(os.environ)
    env["MEMARENA_SEEDS"] = "s1,s2,s3"
    result = subprocess.run(
        [sys.executable, "-c",
         "from memarena.figures import paper_data; print(','.join(paper_data.SEEDS))"],
        check=True, cwd=str(REPO), env=env, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "s1,s2,s3"
