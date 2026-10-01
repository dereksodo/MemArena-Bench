#!/usr/bin/env python3
"""One-shot entry point that regenerates every paper table and figure.

Runs each specialised generator script in dependency order and prints a
short status line per step. Nothing here touches data: the generators
themselves read ``MASim/runs/l_20260408_111046/`` via
:mod:`paper.paper_data` and write into ``paper/tables/`` or
``paper/figures/``. This script is just the convenience wrapper so "rerun
the paper" is a single command::

    python generate_tables_figures.py

Usage notes:

* Each generator is invoked as a fresh ``python3`` subprocess so that an
  exception in one (e.g. missing data for a pending cell) does not abort
  the others. Failures are printed and the runner continues.
* Scripts are grouped: table generators first (cheap; output is text),
  figure generators second (slower; matplotlib backends). Within each
  group order does not matter — scripts are independent.
* If you need to skip one, pass its label as ``--skip`` (can be repeated).
  Example: ``python generate_tables_figures.py --skip fig_d6``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DIR = Path(__file__).resolve().parent

# (label, script_path, one-line description) triples. Label is what the
# user passes to ``--skip``; description shows in the status line.
TABLE_GENERATORS = [
    ("main_tables",   PAPER_DIR / "gen_tables.py",
     "Table 3 (main_L) + Tab 7 (appendix_L) + appendix_L_subdim"),
    ("tokenf1",       PAPER_DIR / "gen_tokenf1.py",
     "Table 8 — Token F1 per open-ended dim (appendix_L_f1)"),
    ("privacy_table", PAPER_DIR / "gen_privacy_table.py",
     "Table 9 — D6 withhold/false-refuse (appendix_L_privacy)"),
]

FIGURE_GENERATORS = [
    ("fig_rec_rea",   PAPER_DIR / "analyze_rec_rea_scatter.py",
     "Figure — Recall vs Reasoning scatter + per-backend ellipses"),
    ("fig_d6",        PAPER_DIR / "gen_fig_d6_v2.py",
     "Figure 3 (fig:d6) — content / social / privacy-utility 3-panel"),
    ("fig_d6_split",  PAPER_DIR / "analyze_d6_split.py",
     "Figure — D6 three-panel split (disclose / refuse social / refuse auto)"),
    ("fig_finding3",  PAPER_DIR / "analyze_finding3_figure.py",
     "Figure 4 (fig:finding3) — Pareto + lift + winner sweep"),
    ("fig_extractor", PAPER_DIR / "analyze_extractor_tradeoff.py",
     "Figure — Config A vs Config B extractor trade-off scatter"),
    ("fig_deployment_triptych", PAPER_DIR / "analyze_deployment_triptych.py",
     "Figure — Deployment triptych (TTFT / Decode / Total) on Spark"),
    ("fig_ingest_amortization", PAPER_DIR / "analyze_ingest_amortization.py",
     "Figure — Memcache ingest amortization (J / s break-even)"),
]

DIAGNOSTIC_SCRIPTS = [
    # Run-once diagnostic: prints a table to stdout, writes no paper artefact.
    ("refusal_audit", PAPER_DIR / "analyze_refusal_provenance.py",
     "Diagnostic — classify Vanilla/RAG D6 refusals (info_absent vs privacy)"),
]


def _run_one(label: str, script: Path, desc: str, extra_args: list[str]) -> bool:
    """Invoke one generator; return True on exit 0, False otherwise.

    Captures stdout/stderr so the per-script output doesn't flood the
    driver's console; we just surface the final status line. If something
    fails the captured output is printed in full to aid debugging.
    """
    if not script.exists():
        print(f"[skip] {label:15s} — missing script {script}")
        return False
    print(f"[run ] {label:15s} — {desc}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script), *extra_args],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL] {label:15s} — exit {result.returncode} in {dt:.1f}s")
        if result.stdout.strip():
            print("  --- stdout ---")
            print("  " + result.stdout.replace("\n", "\n  "))
        if result.stderr.strip():
            print("  --- stderr ---")
            print("  " + result.stderr.replace("\n", "\n  "))
        return False
    tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    print(f"[ok  ] {label:15s} — {dt:.1f}s  {tail}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip", action="append", default=[],
        metavar="LABEL",
        help="label of a generator to skip (repeatable). Valid labels: "
             + ", ".join(l for l, *_ in TABLE_GENERATORS + FIGURE_GENERATORS + DIAGNOSTIC_SCRIPTS),
    )
    parser.add_argument(
        "--tables-only", action="store_true",
        help="skip all figure generators (faster sanity check)",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="skip all table generators",
    )
    parser.add_argument(
        "--with-diagnostics", action="store_true",
        help="also run diagnostic scripts (analyze_refusal_provenance etc.) "
             "that print to stdout but write no paper artefact",
    )
    args, extra_args = parser.parse_known_args()

    skip = set(args.skip)

    # Decide which groups run this invocation.
    scripts: list[tuple[str, Path, str]] = []
    if not args.figures_only:
        scripts.extend(TABLE_GENERATORS)
    if not args.tables_only:
        scripts.extend(FIGURE_GENERATORS)
    if args.with_diagnostics:
        scripts.extend(DIAGNOSTIC_SCRIPTS)

    ok_count = fail_count = 0
    for label, script, desc in scripts:
        if label in skip:
            print(f"[skip] {label:15s} — explicitly skipped via --skip")
            continue
        if _run_one(label, script, desc, extra_args):
            ok_count += 1
        else:
            fail_count += 1

    print()
    print(f"Done. {ok_count} ok, {fail_count} failed, "
          f"{len(scripts) - ok_count - fail_count} skipped.")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
