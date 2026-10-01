"""Generate the bodies of the per-dimension main-grid tables.

  appendix_L_body.tex             tab:results-L-full        (all five readers)
  appendix_cross_family_body.tex  tab:cross-family-readers  (Llama-3.2-3B, Mistral-7B)

Both carry D1..D6 and Avg per (backend, reader) from ``memarena.figures.main_grid``.

    python -m memarena.figures.gen_appendix_grid_tables
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from eval.src.permission_metrics import LEAK_FLAG, LEAK_LABEL, POPULATION_ALL, POPULATION_FACT
from memarena.figures.main_grid import BACKENDS, READER_LABEL, READERS, Stat, main_grid_stats
from memarena.figures.paths import table_path

BACKEND_LABEL = {"vanilla": "Vanilla", "rag": "RAG", "memobase": "Memobase", "memsearch": "MemSearch", "oracle": "Oracle"}
COLUMNS = ("D1", "D2", "D3", "D4", "D5", "D6", "Avg")
CROSS_FAMILY_READERS = ("llama3b", "7b")


def _cell(stat: Stat) -> str:
    return f"{round(stat.mean, 1):.1f}{{\\scriptsize$\\pm${round(stat.sd, 1):.1f}}}"


def render_body(stats: Dict[Tuple[str, str], Dict[str, Stat]], readers: Sequence[str]) -> str:
    lines: List[str] = []
    for i, b in enumerate(BACKENDS):
        lines.append(f"\\multirow{{{len(readers)}}}{{*}}{{{BACKEND_LABEL[b]}}}")
        rows = [f"  & {READER_LABEL[r]:<14s} & " + " & ".join(_cell(stats[(b, r)][c]) for c in COLUMNS) + r" \\" for r in readers]
        lines.append("\n\n".join(rows))
        if i < len(BACKENDS) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the per-dimension main-grid table bodies.")
    ap.add_argument("--population", choices=(POPULATION_FACT, POPULATION_ALL), default=POPULATION_FACT)
    ap.add_argument("--leak", choices=(LEAK_LABEL, LEAK_FLAG), default=LEAK_LABEL)
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <artifact root>/tables")
    args = ap.parse_args(argv)

    stats = main_grid_stats(args.population, args.leak, args.instances)
    for name, readers in (("appendix_L_body.tex", READERS), ("appendix_cross_family_body.tex", CROSS_FAMILY_READERS)):
        out = (args.out_dir / name) if args.out_dir else table_path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_body(stats, readers), encoding="utf-8")
        print(f"[gen_appendix_grid_tables] population={args.population} leak={args.leak} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
