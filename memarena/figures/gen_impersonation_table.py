"""Generate the body of tab:impersonation (spoofed authorisation on DENY).

Oracle context, readers Qwen3-8B and Mistral-7B, seeds s2/s3/s4, three
conditions: L0 (the main grid: asker named, owner's contacts shown), IMP-owner
("I am the owner"), IMP-auth (claimed delegation). Only the 73 DENY queries
carry the spoofed prefix; the 71 ALLOW instances are byte-identical to the
released dataset, so utility is fixed by construction.

Leak = DISCLOSE_CORRECT on DENY and utility = DISCLOSE_CORRECT on ALLOW, as in
the main grid; items are the 144 fact-bearing D6 items. ± is the sample SD over
seeds; ΔF1_PU is against L0 of the same reader and judge.

Cell sources (``memarena.figures.paper_data.ablation_runs_dir``):
  L0       the main-grid Oracle cells (MEMARENA_RUNS_DIR)
  IMP-*    oracle-imp{owner,auth}/<reader>/<seed>/oracle/
           evaluation_results_oracle_imp{owner,auth}_<reader>_<seed>_judge_<tag>.json
           (answers from run_d6_impersonation.py datasets via test_cell.sh -tag)

    python -m memarena.figures.gen_impersonation_table
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path
from typing import Dict, List

from eval.src.permission_metrics import POPULATION_FACT, load_instances, population_ids
from memarena.figures.gen_d6_identity_table import seed_metrics
from memarena.figures.main_grid import SEEDS, default_instances_path
from memarena.figures.paper_data import ablation_eval_path, load_all_cells
from memarena.figures.paths import table_path

READERS = (("8b", "Qwen3-8B"), ("7b", "Mistral-7B"))
LEVELS = (("L0", "L0 (main grid)"), ("impowner", "IMP-owner"), ("impauth", "IMP-auth"))
JUDGES = ("gpt4omini", "deepseek")


def cell_paths(reader: str, level: str, judge: str) -> List[Path]:
    if level == "L0":
        grid = load_all_cells(include_ablation=False, judge=judge)
        return [Path(grid[(s, "oracle", reader)].source_path) for s in SEEDS]
    return [ablation_eval_path(f"oracle-{level}", "oracle", f"oracle_{level}", reader, s, judge) for s in SEEDS]


def collect(instances: Path | None = None) -> Dict[tuple, List[Dict[str, float]]]:
    keep = population_ids(load_instances(instances or default_instances_path()), POPULATION_FACT)
    return {
        (r, lvl, j): [seed_metrics(p, keep) for p in cell_paths(r, lvl, j)]
        for r, _ in READERS for lvl, _ in LEVELS for j in JUDGES
    }


def render_body(data) -> str:
    lines: List[str] = []
    for i, (r, label) in enumerate(READERS):
        base = {j: st.mean(m["f1"] for m in data[(r, "L0", j)]) for j in JUDGES}
        for k, (lvl, name) in enumerate(LEVELS):
            rows = data[(r, lvl, "gpt4omini")]
            f1s = [m["f1"] for m in rows]
            head = f"\\multirow{{{len(LEVELS)}}}{{*}}{{{label}}}" if k == 0 else ""
            if lvl == "L0":
                d_p = d_d = "---"
            else:
                d_p = f"${st.mean(f1s) - base['gpt4omini']:+.1f}$"
                d_d = f"${st.mean(m['f1'] for m in data[(r, lvl, 'deepseek')]) - base['deepseek']:+.1f}$"
            lines.append(
                (f"{head}\n" if head else "") + f"  & {name:<15s} & ${st.mean(m['leak'] for m in rows):.1f}$ & "
                f"${st.mean(f1s):.1f} \\pm {st.stdev(f1s):.1f}$ & {d_p} & {d_d} \\\\"
            )
        if i < len(READERS) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the impersonation table body.")
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or table_path("impersonation_body.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_body(collect(args.instances)), encoding="utf-8")
    print(f"[gen_impersonation_table] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
