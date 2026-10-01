#!/usr/bin/env python3
"""Regenerate every table and figure of the MemArena paper from result files.

One command rebuilds all paper artifacts from the per-item judge outputs
(``evaluation_results_*.json`` and the latency/serving logs). Result files are
not part of the repository; point ``MEMARENA_RESULTS_DIR`` at the directory
holding the ``out/`` (and ``MASim/runs/``) tree, and ``MEMARENA_DATASET_DIR``
at the released MemArena-L dataset.

Usage
-----
  python scripts/reproduce_figures.py --list
  python scripts/reproduce_figures.py --all --out-dir paper_rev0719/generated   # what the paper inputs
  python scripts/reproduce_figures.py --check paper_rev0719                      # regenerate and diff
  python scripts/reproduce_figures.py --name main_table --out-dir /tmp/x
  python scripts/reproduce_figures.py --all --runs-dir out/runs --judge gpt4omini --out-dir /tmp/x

Results source. By default the main-grid cells come from the paper's
``experiments_index.csv`` layout (``MEMARENA_RESULTS_DIR``). ``--runs-dir``
(or ``MEMARENA_RUNS_DIR``) reads the grid-runner tree instead,
``<runs>/<backend>/<model>/<seed>/<system>/evaluation_results_*_judge_<tag>.json``,
for the judge ``--judge`` / ``MEMARENA_JUDGE``: ``gpt4omini`` (default, the
primary judge, ``*_judge_remote.json``) or ``deepseek`` (the secondary judge,
``*_judge_deepseek.json``). This switches every
generator that goes through ``paper_data.load_all_cells`` (main table, grids,
D6 tables and figures); artifacts built from ablation rows, latency logs or the
dataset keep their own inputs.

The paper inputs generated files from ``<paper>/generated/{tables,figures}/``;
hand-written captions and headers stay in ``<paper>/tables/``. ``--check``
regenerates everything into a temporary directory and compares: generated
bodies must equal ``<paper>/generated/tables/<name>`` byte for byte; full tables
whose caption is maintained by hand in ``<paper>/tables/<name>`` are compared
number by number (``memarena.tools.tex_numeric_diff``). Figures are regenerated
but not diffed.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from memarena.judges import JUDGES  # noqa: E402
from memarena.runtime import configure_live_output  # noqa: E402
from memarena.tools.tex_numeric_diff import diff_numbers, extract_numbers  # noqa: E402

EXACT = "exact"      # generated file is included verbatim by the paper
NUMBERS = "numbers"  # paper keeps its own caption/header; compare numbers after the header


@dataclass
class Artifact:
    name: str
    module: str
    description: str
    tables: List[Tuple[str, str]] = field(default_factory=list)  # (file under tables/, check mode)
    figures: List[str] = field(default_factory=list)


# In the order the artifacts appear in the paper.
ARTIFACTS: List[Artifact] = [
    Artifact("main_table", "memarena.figures.gen_main_table", "Table: main results (tab:results-L)",
             [("main_SML_body.tex", EXACT)]),
    Artifact("core_findings", "memarena.figures.gen_fig_core_findings", "Figure: D6 two modes + TTFT decomposition",
             figures=["fig_core_findings.pdf"]),
    Artifact("dataset_summary", "memarena.figures.gen_dataset_summary", "Table: dataset statistics",
             [("dataset_summary_body.tex", EXACT)]),
    Artifact("rec_rea_scatter", "memarena.figures.analyze_rec_rea_scatter", "Figure: Recall vs Reasoning",
             figures=["fig_rec_rea_scatter.pdf"]),
    Artifact("d6_heatmap", "memarena.figures.gen_fig_d6_finding1", "Figure: per-cell F1_PU heatmap",
             figures=["fig_d6_f1pu_heatmap.pdf"]),
    Artifact("d6_deny_composition", "memarena.figures.gen_fig_d6_ablation_buckets",
             "Table + figure: DENY-response composition (tab:d6-ablation-buckets)",
             [("d6_ablation_buckets_body.tex", EXACT)], ["fig_d6_ablation_buckets.pdf"]),
    Artifact("d6_decomposition", "memarena.figures.gen_permission_decomposition",
             "Tables: D6 DENY composition, Oracle leak by family, BM25 evidence audit",
             [("d6_deny_composition_body.tex", EXACT), ("d6_family_leak_body.tex", EXACT), ("d6_bm25_audit_body.tex", EXACT)]),
    Artifact("d6_ablation_sweep", "memarena.figures.gen_d6_ablation_sweep", "Table: D6 ablation sweep",
             [("d6_ablation_sweep_body.tex", EXACT)]),
    Artifact("oracle_distractors", "memarena.figures.gen_oracle_distractor_tables",
             "Table: Oracle with distractors",
             [("review4_oracle_dist_body.tex", EXACT)]),
    Artifact("access_markers", "memarena.figures.gen_access_marker_table", "Table: reader-side access markers",
             [("appendix_access_markers_body.tex", EXACT)]),
    Artifact("prompt_format", "memarena.figures.gen_prompt_format_table", "Table: prompt-format control on D5",
             [("prompt_format_body.tex", EXACT)]),
    Artifact("writer32", "memarena.figures.gen_writer32_table", "Table: Qwen3-32B memory writer",
             [("appendix_cross_extractor_writer32_body.tex", EXACT)]),
    Artifact("d6_identity", "memarena.figures.gen_d6_identity_table", "Table: D6 requester-identity strength",
             [("d6_identity_body.tex", EXACT)]),
    Artifact("impersonation", "memarena.figures.gen_impersonation_table", "Table: spoofed-authorisation slice",
             [("impersonation_body.tex", EXACT)]),
    Artifact("appendix_grid", "memarena.figures.gen_appendix_grid_tables", "Tables: per-dimension main grid + cross-family",
             [("appendix_L_body.tex", EXACT), ("appendix_cross_family_body.tex", EXACT)]),
    Artifact("privacy_utility", "memarena.figures.gen_d6_privacy_utility_table", "Table: D6 privacy / utility halves",
             [("appendix_L_privacy_body.tex", EXACT)]),
    Artifact("realism", "memarena.figures.gen_realism_tables",
             "Tables: realism audit vs. REALTALK / DailyDialog / PERSONA-CHAT / LoCoMo",
             [("realism_structure_body.tex", EXACT), ("realism_lexical_body.tex", EXACT)]),
    Artifact("identity_continuity", "memarena.figures.gen_identity_continuity_table",
             "Table: cross-session identity-continuity ablation (break vs. sham)",
             [("identity_continuity_body.tex", EXACT)]),
    Artifact("judge_robustness", "memarena.figures.gen_judge_robustness_table",
             "Tables: cross-family judge ranking stability + agreement",
             [("judge_robustness_rho_body.tex", EXACT), ("judge_robustness_agreement_body.tex", EXACT)]),
    Artifact("latency", "scripts.gen_latency_appendix_tables",
             "Tables: cache-off prefill/decode curves, end-to-end latency, answering efficiency",
             [("appendix_latency_fit_body.tex", EXACT), ("appendix_latency_predicted_body.tex", EXACT),
              ("efficiency_body.tex", EXACT)]),
    Artifact("ingest_overhead", "memarena.figures.analyze_serving_costs", "Table: Memobase ingest overhead",
             [("appendix_ingest_overhead_body.tex", EXACT)]),
]
BY_NAME = {a.name: a for a in ARTIFACTS}


def _print_listing() -> None:
    w = max(len(a.name) for a in ARTIFACTS)
    print("Paper artifacts (run in this order by --all):")
    for a in ARTIFACTS:
        print(f"  {a.name:<{w}}   {a.description}")


def _run(art: Artifact, extra: List[str]) -> int:
    orig_argv = sys.argv
    try:
        mod = importlib.import_module(art.module)
        sys.argv = [art.module, *extra]
        rc = mod.main() or 0
    except SystemExit as ex:
        rc = int(ex.code or 0)
    except Exception as ex:  # report and keep going with the other artifacts
        print(f"[reproduce] {art.name}: error -> {type(ex).__name__}: {ex}", file=sys.stderr)
        rc = 2
    finally:
        sys.argv = orig_argv
    print(f"[reproduce] {art.name}: {'ok' if rc == 0 else f'exit {rc}'}")
    return rc


def _check(generated_root: Path, paper_dir: Path, arts: List[Artifact]) -> int:
    failures = 0
    for art in arts:
        for name, mode in art.tables:
            gen = generated_root / "tables" / name
            ref = paper_dir / ("generated/tables" if mode == EXACT else "tables") / name
            if not ref.exists():
                print(f"MISSING  {ref} (not yet in the paper)")
                failures += 1
                continue
            if mode == EXACT:
                a, b = gen.read_text(encoding="utf-8"), ref.read_text(encoding="utf-8")
                bad = [] if a == b else diff_numbers(extract_numbers(a), extract_numbers(b))
                same = a == b
            else:
                bad = diff_numbers(
                    extract_numbers(gen.read_text(encoding="utf-8"), skip_header=True),
                    extract_numbers(ref.read_text(encoding="utf-8"), skip_header=True),
                )
                same = not bad
            if same:
                print(f"OK       {name}")
            else:
                failures += 1
                detail = "; ".join(f"#{i}: {x} vs {y}" for i, x, y in bad[:3]) or "text differs, numbers equal"
                print(f"DIFF     {name}: {detail}")
    print(f"[reproduce] check: {failures} artifact(s) differ from {paper_dir}")
    return 1 if failures else 0


def main() -> int:
    configure_live_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="list the paper artifacts and exit")
    g.add_argument("--name", choices=sorted(BY_NAME), help="regenerate a single artifact")
    g.add_argument("--all", action="store_true", help="regenerate every artifact")
    g.add_argument("--check", type=Path, metavar="PAPER_DIR",
                   help="regenerate into a temp dir and compare with PAPER_DIR/tables")
    ap.add_argument("--out-dir", type=Path,
                    help="artifact root (tables/ and figures/ go inside); default $MEMARENA_FIGURE_OUT_DIR")
    ap.add_argument("--runs-dir", type=Path,
                    help="read main-grid cells from this grid-runner tree (sets MEMARENA_RUNS_DIR)")
    ap.add_argument("--judge", choices=sorted(JUDGES),
                    help="judge of the runner tree (sets MEMARENA_JUDGE; default gpt4omini)")
    ap.add_argument("remaining", nargs=argparse.REMAINDER, help="extra args forwarded to a single --name run")
    args = ap.parse_args()

    if args.list or not (args.name or args.all or args.check):
        _print_listing()
        return 0

    if args.runs_dir:
        runs = args.runs_dir.expanduser()
        os.environ["MEMARENA_RUNS_DIR"] = str(runs if runs.is_absolute() else (REPO_ROOT / runs).resolve())
    if args.judge:
        os.environ["MEMARENA_JUDGE"] = args.judge
    from memarena.figures.paper_data import results_source_label
    print(f"[reproduce] results: {results_source_label()}")

    tmp = None
    if args.check:
        tmp = tempfile.TemporaryDirectory(prefix="memarena_check_")
        os.environ["MEMARENA_FIGURE_OUT_DIR"] = tmp.name
    elif args.out_dir:
        out = args.out_dir.expanduser()
        os.environ["MEMARENA_FIGURE_OUT_DIR"] = str(out if out.is_absolute() else (REPO_ROOT / out).resolve())
    print(f"[reproduce] artifact root: {os.environ.get('MEMARENA_FIGURE_OUT_DIR', '(default)')}")

    arts = [BY_NAME[args.name]] if args.name else ARTIFACTS
    rc_all = 0
    for art in arts:
        rc_all = _run(art, args.remaining if args.name else []) or rc_all
    if args.check:
        rc_all = _check(Path(tmp.name), args.check.resolve(), arts) or rc_all
        tmp.cleanup()
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
