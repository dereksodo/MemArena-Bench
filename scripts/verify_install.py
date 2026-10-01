#!/usr/bin/env python3
"""Post-install health check for MemArena.

Run this immediately after activating ``.venv`` and running
``python -m pip install -e ".[dev]"`` to confirm your checkout is wired up
correctly. It:

1.  Runs the dry-run pytest suite (no GPU / no API keys / no dataset
    download). All non-optional tests must pass on any fresh clone.
2.  Reports which optional paths are SKIPPED because they require assets
    the repo deliberately does not ship:
      * paper-figure reproduction (needs ``memarena/figures/paper_data.py``
        from the paper-dev workflow and a MASim run directory)
      * ``python -m eval.cli --dry-run`` smoke (needs a MASim run directory)

Exit code:
    0  — the core tests passed. You're good to follow the README.
    1  — at least one core test failed; your install is inconsistent.

Output is intentionally colour-free so it pipes cleanly into CI logs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memarena.runtime import configure_live_output

PAPER_DATA = REPO / "memarena" / "figures" / "paper_data.py"
CANONICAL_RUN = REPO / "MASim" / "runs" / "l_20260408_111046"
SMOKE_RUN = REPO / "MASim" / "runs" / "l_20260408_111046"
RUN_READY = (
    SMOKE_RUN.exists()
    and (SMOKE_RUN / "corpus_sessions.jsonl").exists()
    and (SMOKE_RUN / "eval_instances").is_dir()
)
RESULTS_READY = (
    CANONICAL_RUN.exists()
    and (CANONICAL_RUN / "eval_results").is_dir()
    and (CANONICAL_RUN / "spark_results").is_dir()
)


def _h(title: str) -> None:
    bar = "=" * 68
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> int:
    configure_live_output()
    _h("MemArena install verification")

    # --- Part 1: core pytest suite (must pass) -----------------------------
    print("\n[1/2] Running dry-run pytest suite (optional dataset tests may skip)...\n")
    rc = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "tests"), "-v", "--tb=short"],
        cwd=str(REPO),
    ).returncode
    if rc != 0:
        print("\n[verify] FAIL: at least one core test failed. Your install is "
              "inconsistent with the reference CI environment.")
        return 1

    # --- Part 2: optional-asset matrix -------------------------------------
    _h("What you CAN run next")

    print("\nSmoke path (no external setup required):")
    print("  python run_masim.py --smoke --output out/smoke/masim --overwrite")
    print("  python scripts/run_accuracy.py --dry-run --backend vanilla --n 10 "
          "--out-dir out/smoke/accuracy_vanilla")
    print("  python scripts/run_accuracy.py --dry-run --backend oracle --n 10 "
          "--out-dir out/smoke/accuracy_oracle")
    print("  python scripts/run_latency.py --dry-run --backend vanilla --n 10 "
          "--out-dir out/smoke/latency_vanilla")
    print("\nReal path:")
    print("  Follow README.md -> 'After verify_install.py: choose a path'.")

    _h("What you need to PROVIDE to unlock more")

    optional = [
        ("Paper figure generation (reproduces paper tables/PDFs)",
         PAPER_DATA.exists() and RESULTS_READY,
         "Generate local eval_results*/ + spark_results*/ under "
         "MASim/runs/l_20260408_111046/. The hosted dataset does not include "
         "baseline result files."),
        ("`python -m eval.cli --dry-run` end-to-end smoke",
         RUN_READY,
         "Populate a MASim run directory at MASim/runs/l_20260408_111046/ "
         "(corpus_sessions.jsonl + eval_instances/) — your own generated run "
         "works, or symlink from an existing MemArena dataset download."),
        ("Canonical MemArena-L figures",
         RESULTS_READY,
         "Generate eval_results*/ + spark_results*/ locally by running the "
         "reproduce/evaluation pipeline on your own infrastructure."),
    ]
    for title, present, hint in optional:
        status = "present"    if present else "absent "
        print(f"\n  [{status}]  {title}")
        if not present:
            print(f"           → {hint}")

    print("\n[verify] OK: core dry-run suite passes. Follow README for real runs.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
