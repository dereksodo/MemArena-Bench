#!/usr/bin/env python3
"""Post-install health check for MemArena.

Run this right after ``scripts/setup_venv.sh`` and
``source scripts/activate_venv.sh`` to confirm your checkout is wired up
correctly. It:

1.  Runs the dry-run pytest suite (no GPU / no API keys / no dataset
    download). All non-optional tests must pass on any fresh clone.
2.  Reports which optional paths are SKIPPED because they require assets
    the repo deliberately does not ship:
      * real runs (need the MemArena-L dataset from ``scripts/download_dataset.py``)
      * paper tables and figures (need your own grid outputs under ``out/runs``)

Exit code:
    0  — the core tests passed. You're good to follow the README.
    1  — at least one core test failed; your install is inconsistent.

Output is intentionally colour-free so it pipes cleanly into CI logs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memarena.runtime import configure_live_output

DATASET = Path(os.getenv("MEMARENA_DATASET_DIR") or REPO / "data" / "benchmark")
DATASET_READY = (DATASET / "eval_instances").is_dir()
RESULTS_READY = (REPO / "out" / "runs").is_dir()


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
    print("  Follow README.md -> 'Reproducing the paper' and docs/REPRODUCE.md.")

    _h("What you need to PROVIDE to unlock more")

    optional = [
        (f"MemArena-L dataset ({DATASET})",
         DATASET_READY,
         "python scripts/download_dataset.py --out data/   (or set MEMARENA_DATASET_DIR)"),
        ("Paper tables and figures from your own runs",
         RESULTS_READY,
         "Run the grid (README -> 'Reproducing the paper'); the result files behind the "
         "paper are not distributed. Then: python scripts/reproduce_figures.py --runs-dir out/runs --all"),
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
