"""Generate the body of tab:d6-identity (requester-identity strength on D6).

Qwen3-8B + Oracle, seeds s2-s4, the 144 fact-bearing D6 items. The three levels
differ only in what the reader is told about the requester:
  L0  neither the asker's name nor the owner's contacts   (test_cell.sh -d6-no-asker -d6-no-contacts)
  L1  the asker's name, no contacts                         (test_cell.sh -d6-no-contacts)
  L2  the asker's name and the owner's contacts             (the main grid)

Cell sources (``memarena.figures.paper_data.ablation_runs_dir``):
  L0/L1  oracle-idL<n>/8b/<seed>/oracle/evaluation_results_oracle_idL<n>_8b_<seed>_judge_<tag>.json
  L2     the main-grid Oracle / Qwen3-8B cells (MEMARENA_RUNS_DIR)

    python -m memarena.figures.gen_d6_identity_table
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict, List

from eval.src.permission_metrics import d6_records, f1_pu, is_allow, is_leak, load_instances, population_ids, POPULATION_FACT
from memarena.figures.main_grid import SEEDS, default_instances_path, scored
from memarena.figures.paper_data import ablation_eval_path, load_all_cells
from memarena.figures.paths import table_path

READER = "8b"
LEVELS = (
    ("L0", "neither"),
    ("L1", "asker's name"),
    ("L2", "name $+$ owner's contacts (main grid)"),
)
JUDGES = ("gpt4omini", "deepseek")


def cell_paths(level: str, judge: str) -> List[Path]:
    if level == "L2":
        grid = load_all_cells(include_ablation=False, judge=judge)
        return [Path(grid[(s, "oracle", READER)].source_path) for s in SEEDS]
    tag = f"id{level}"
    return [ablation_eval_path(f"oracle-{tag}", "oracle", f"oracle_{tag}", READER, s, judge) for s in SEEDS]


def seed_metrics(path: Path, keep) -> Dict[str, float]:
    recs = d6_records(scored(json.loads(path.read_text()).get("details", [])), keep)
    deny = [r for r in recs if not is_allow(r)]
    allow = [r for r in recs if is_allow(r)]
    leak = sum(is_leak(r) for r in deny) / len(deny)
    util = sum(r.get("policy_category") == "DISCLOSE_CORRECT" for r in allow) / len(allow)
    return {"leak": 100 * leak, "util": 100 * util, "f1": 100 * f1_pu(1 - leak, util)}


def collect(instances: Path | None = None) -> Dict[tuple, List[Dict[str, float]]]:
    keep = population_ids(load_instances(instances or default_instances_path()), POPULATION_FACT)
    return {(lvl, j): [seed_metrics(p, keep) for p in cell_paths(lvl, j)] for lvl, _ in LEVELS for j in JUDGES}


def _ms(rows: List[Dict[str, float]], key: str) -> str:
    vals = [m[key] for m in rows]
    return f"${st.mean(vals):.1f} \\pm {st.stdev(vals):.1f}$"


def render_body(data) -> str:
    lines = []
    for lvl, told in LEVELS:
        p, d = data[(lvl, "gpt4omini")], data[(lvl, "deepseek")]
        lines.append(f"{lvl} & {told} & {_ms(p, 'leak')} & {_ms(p, 'util')} & {_ms(p, 'f1')} & {_ms(d, 'f1')} \\\\")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the D6 requester-identity table body.")
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    body = render_body(collect(args.instances))
    out = args.out or table_path("d6_identity_body.tex")
    out.write_text(body)
    print(f"[gen_d6_identity_table] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
