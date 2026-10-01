"""Generate the body of tab:appendix-cross-extractor-writer32.

Memobase rows are the main-grid Memobase cells; Memobase-writer32 rows pin the
memory writer/extractor to Qwen3-32B-AWQ while keeping the reader (grid-runner
cells ``memobase-writer32/<reader>/<seed>/memory_cache/``, see
``memarena.figures.paper_data.ablation_runs_dir``). Columns D1..D6 and Avg follow
``memarena.figures.main_grid`` (seed mean ± SD).

    python -m memarena.figures.gen_writer32_table
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict, List

from eval.src.permission_metrics import LEAK_FLAG, LEAK_LABEL, POPULATION_ALL, POPULATION_FACT, load_instances, population_ids
from memarena.figures.main_grid import SEEDS, Stat, default_instances_path, main_grid_stats, seed_scores
from memarena.figures.paper_data import ablation_eval_path, selected_judge
from memarena.figures.paths import table_path

READERS = ("0_6b", "llama3b", "7b", "8b")
READER_LABEL = {"0_6b": "Qwen3-0.6B", "llama3b": "Llama-3.2-3B", "7b": "Mistral-7B", "8b": "Qwen3-8B"}
COLUMNS = ("D1", "D2", "D3", "D4", "D5", "D6", "Avg")


def writer32_stats(population: str, leak: str, instances: Path | None) -> Dict[str, Dict[str, Stat]]:
    keep = population_ids(load_instances(instances or default_instances_path()), population)
    judge = selected_judge()
    out = {}
    for r in READERS:
        paths = [ablation_eval_path("memobase-writer32", "memory_cache", "memobase_writer32", r, s, judge) for s in SEEDS]
        rows = [seed_scores(json.loads(p.read_text())["details"], keep, leak) for p in paths]
        out[r] = {c: Stat(st.mean(x[c] for x in rows), st.stdev([x[c] for x in rows]), len(rows)) for c in COLUMNS}
    return out


def _cell(stat: Stat) -> str:
    m = round(stat.mean, 1)
    pad = r"\phantom{0}" if m < 10 else ""
    return f"${pad}{m:.1f}${{\\scriptsize$\\pm{round(stat.sd, 1):.1f}$}}"


def render_body(base: Dict, w32: Dict) -> str:
    lines: List[str] = []
    for label, stats in (("Memobase", {r: base[("memobase", r)] for r in READERS}), ("Memobase-writer32", w32)):
        lines.append(f"\\multirow{{{len(READERS)}}}{{*}}{{{label}}}")
        for r in READERS:
            lines.append(f"  & {READER_LABEL[r]:<12s} & " + " & ".join(_cell(stats[r][c]) for c in COLUMNS) + r" \\")
        if label == "Memobase":
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the writer32 ablation table body.")
    ap.add_argument("--population", choices=(POPULATION_FACT, POPULATION_ALL), default=POPULATION_FACT)
    ap.add_argument("--leak", choices=(LEAK_LABEL, LEAK_FLAG), default=LEAK_LABEL)
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    base = main_grid_stats(args.population, args.leak, args.instances)
    w32 = writer32_stats(args.population, args.leak, args.instances)
    out = args.out or table_path("appendix_cross_extractor_writer32_body.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_body(base, w32), encoding="utf-8")
    print(f"[gen_writer32_table] population={args.population} leak={args.leak} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
