"""Generate the body of tab:results-L-privacy: both halves of F1_PU per cell.

For each (backend, reader), mean ± sample SD over seeds s2/s3/s4 of
  Privacy P    = 1 - leak rate on DENY items (leak = DISCLOSE_CORRECT)
  Utility U    = DISCLOSE_CORRECT rate on ALLOW items
  No answer    = DONT_KNOW + REFUSE rate on ALLOW items
  F1_PU        = harmonic mean of P and U
on the D6 population of ``eval.src.permission_metrics`` (default: the 144
fact-bearing items). Output: <artifact root>/tables/appendix_L_privacy_body.tex

    python -m memarena.figures.gen_d6_privacy_utility_table
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
from memarena.figures.main_grid import BACKENDS, READER_LABEL, READERS, SEEDS, default_instances_path, scored
from memarena.figures.paper_data import load_all_cells
from memarena.figures.paths import table_path

BACKEND_LABEL = {"vanilla": "Vanilla", "rag": "RAG", "memobase": "Memobase", "memsearch": "MemSearch", "oracle": "Oracle"}
COLUMNS = ("privacy", "utility", "no_answer", "f1_pu")


def _seed_values(details, keep, leak) -> Dict[str, float]:
    s = cell_scalars(scored(details), keep, leak=leak)
    return {
        "privacy": 100 * s.privacy,
        "utility": 100 * s.utility,
        "no_answer": 100 * (s.allow_labels["DONT_KNOW"] + s.allow_labels["REFUSE"]),
        "f1_pu": 100 * s.f1_pu,
    }


def collect(population: str, leak: str, instances: Path | None) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]:
    keep = population_ids(load_instances(instances or default_instances_path()), population)
    grid = load_all_cells(include_ablation=False)
    out = {}
    for b in BACKENDS:
        for r in READERS:
            rows = [
                _seed_values(json.loads(Path(grid[(s, b, r)].source_path).read_text()).get("details", []), keep, leak)
                for s in SEEDS
            ]
            out[(b, r)] = {c: (st.mean(x[c] for x in rows), st.stdev([x[c] for x in rows])) for c in COLUMNS}
    return out


def render_body(stats) -> str:
    lines: List[str] = []
    for i, b in enumerate(BACKENDS):
        lines.append(f"\\multirow{{{len(READERS)}}}{{*}}{{{BACKEND_LABEL[b]}}}")
        rows = []
        for r in READERS:
            cells = [f"{round(m, 1):.1f}{{\\scriptsize$\\pm${round(sd, 1):.1f}}}" for m, sd in (stats[(b, r)][c] for c in COLUMNS)]
            rows.append(f"  & {READER_LABEL[r]:<14s} & " + " & ".join(cells) + r" \\")
        lines.append("\n\n".join(rows))
        if i < len(BACKENDS) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the D6 privacy/utility table body.")
    ap.add_argument("--population", choices=(POPULATION_FACT, POPULATION_ALL), default=POPULATION_FACT)
    ap.add_argument("--leak", choices=(LEAK_LABEL, LEAK_FLAG), default=LEAK_LABEL)
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out = args.out or table_path("appendix_L_privacy_body.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_body(collect(args.population, args.leak, args.instances)), encoding="utf-8")
    print(f"[gen_d6_privacy_utility_table] population={args.population} leak={args.leak} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
