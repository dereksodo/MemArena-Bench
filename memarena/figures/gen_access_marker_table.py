"""Generate the body of tab:appendix-access-markers.

Per reader and seed, D6 F1_PU of the Oracle main-grid cell ("base") and of the
same Oracle context with the item's gold access label ("Access policy:
[access:DENY]" / "[access:ALLOW]") prepended to the reader prompt
(grid-runner cells ``oracle-access/<reader>/<seed>/oracle/``, see
``memarena.figures.paper_data.ablation_runs_dir``). Reported per reader: seed means of base and marker F1_PU,
their difference Δ, and the [min, max] of the per-seed Δ; the last row pools
all 15 (reader, seed) cells.

    python -m memarena.figures.gen_access_marker_table
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple

from eval.src.permission_metrics import (
    LEAK_FLAG,
    LEAK_LABEL,
    POPULATION_ALL,
    POPULATION_FACT,
    cell_scalars,
    load_instances,
    population_ids,
)
from memarena.figures.main_grid import READERS, SEEDS, default_instances_path, scored
from memarena.figures.paper_data import ablation_eval_path, load_all_cells, selected_judge
from memarena.figures.paths import table_path

READER_LABEL = {"0_6b": "Qwen3-0.6B", "llama3b": "Llama-3.2-3B", "7b": "Mistral-7B", "8b": "Qwen3-8B", "32b": "Qwen3-32B-AWQ"}


def _f1(path: Path, keep, leak) -> float:
    details = json.loads(path.read_text()).get("details", [])
    return 100 * cell_scalars(scored(details), keep, leak=leak).f1_pu


def _marker_paths() -> Dict[Tuple[str, str], Path]:
    judge = selected_judge()
    return {(s, r): ablation_eval_path("oracle-access", "oracle", "oracle_access", r, s, judge) for r in READERS for s in SEEDS}


def collect(population: str, leak: str, instances: Path | None):
    keep = population_ids(load_instances(instances or default_instances_path()), population)
    grid = load_all_cells(include_ablation=False)
    markers = _marker_paths()
    rows = {}
    for r in READERS:
        base = [_f1(Path(grid[(s, "oracle", r)].source_path), keep, leak) for s in SEEDS]
        mark = [_f1(markers[(s, r)], keep, leak) for s in SEEDS]
        rows[r] = (base, mark)
    return rows


def _signed(x: float) -> str:
    return f"{x:+.1f}" if round(x, 1) != 0 else "0.0"


def _pp(x: float) -> str:
    return f"${_signed(x)}$"


def render_body(rows) -> str:
    lines = []
    all_base, all_mark, all_delta = [], [], []
    for r in READERS:
        base, mark = rows[r]
        delta = [m - b for b, m in zip(base, mark)]
        all_base += base
        all_mark += mark
        all_delta += delta
        lines.append(
            f"{READER_LABEL[r]:<13s} & ${st.mean(base):.1f}$ & ${st.mean(mark):.1f}$ & {_pp(st.mean(delta))} & "
            f"$[{_signed(min(delta))},\\ {_signed(max(delta))}]$ \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        f"Mean (${len(all_delta)}$ cells) & ${st.mean(all_base):.1f}$ & ${st.mean(all_mark):.1f}$ & {_pp(st.mean(all_delta))} & "
        f"$[{_signed(min(all_delta))},\\ {_signed(max(all_delta))}]$ \\\\"
    )
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the access-marker table body.")
    ap.add_argument("--population", choices=(POPULATION_FACT, POPULATION_ALL), default=POPULATION_FACT)
    ap.add_argument("--leak", choices=(LEAK_LABEL, LEAK_FLAG), default=LEAK_LABEL)
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or table_path("appendix_access_markers_body.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_body(collect(args.population, args.leak, args.instances)), encoding="utf-8")
    print(f"[gen_access_marker_table] population={args.population} leak={args.leak} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
