"""Smoke test for `python -m eval.cli --dry-run`.

Verifies the end-to-end eval pipeline runs offline (no sglang, no API keys)
by routing through the local vanilla adapter and forcing the
heuristic-fallback answerer via an empty api_key.

Skipped when the MemArena-L MASim run (required for corpus + eval
instances) is not available — keeps CI green on fresh checkouts.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "MASim" / "runs" / "l_20260408_111046"
RUN_READY = (
    RUN_DIR.exists()
    and (RUN_DIR / "corpus_sessions.jsonl").exists()
    and (RUN_DIR / "eval_instances").is_dir()
)


pytestmark = pytest.mark.skipif(
    not RUN_READY,
    reason=(
        "eval dry-run needs a full MASim run directory with corpus_sessions.jsonl + "
        "eval_instances/; symlink MASim/runs/l_20260408_111046 to exercise"
    ),
)


def test_eval_cli_dryrun(tmp_path: Path) -> None:
    out_dir = tmp_path / "eval_out"

    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli",
            "--run-dir", str(RUN_DIR),
            "--system", "vanilla",
            "--stages", "answer",
            "--dry-run",
            "--output-dir", str(out_dir),
            "--namespace", "smoke",
        ],
        check=True, cwd=str(REPO), capture_output=True, text=True,
    )

    answer_results = out_dir / "vanilla" / "answer_results_smoke.json"
    assert answer_results.exists(), f"missing {answer_results}\nstdout={result.stdout}"

    records = json.loads(answer_results.read_text())
    assert isinstance(records, list) and len(records) == 10, (
        f"expected 10 records (dry-run qa_limit), got {len(records)}"
    )
    for r in records:
        assert {"question_id", "question", "prediction", "model"}.issubset(r)
        # Heuristic fallback marks its outputs so we can verify no HTTP was made.
        assert r["model"] in {"no-client", "heuristic-offline"}, r["model"]
