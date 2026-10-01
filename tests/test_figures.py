"""Smoke test for `scripts/reproduce_figures.py`.

The paper-figure generators under `memarena/figures/` need the released
CSV artifacts from the Hugging Face dataset repo. When those aren't on
disk the test is skipped — CI still runs, while a local developer who has
symlinked or downloaded the artifacts exercises the pipeline.

We use the main table as the smoke artifact: it exercises the full
main-grid loader end-to-end on the canonical MemArena-L run.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CANONICAL_RUN = REPO / "MASim" / "runs" / "l_20260408_111046"
LOADER = REPO / "memarena" / "figures" / "paper_data.py"


pytestmark = pytest.mark.skipif(
    not (CANONICAL_RUN.exists() and LOADER.exists()),
    reason=(
        "baseline run + paper_data loader not present; "
        "symlink MASim/runs/l_20260408_111046 and copy paper_data.py to exercise"
    ),
)


def test_gen_tables_smoke(tmp_path: Path) -> None:
    tables_out = tmp_path / "tables" / "main_SML_body.tex"
    env = {**os.environ, "MEMARENA_FIGURE_OUT_DIR": str(tmp_path)}

    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "reproduce_figures.py"),
         "--name", "main_table"],
        check=True, cwd=str(REPO), env=env,
    )

    assert tables_out.exists(), f"{tables_out} was not written"
    body = tables_out.read_text()
    assert len(body) > 500, f"{tables_out} suspiciously short ({len(body)} chars)"
    assert "\\multirow" in body and "TTFT/ms" in body, "expected the reader blocks of the main table"
