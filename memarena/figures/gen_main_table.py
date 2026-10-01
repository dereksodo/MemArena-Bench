"""Generate the body of the main results table (tab:results-L).

Writes ``main_SML_body.tex`` -- everything between the header ``\\midrule``
and ``\\bottomrule`` -- which ``paper_rev0719/tables/main_SML.tex`` inputs.
Accuracy rows come from ``memarena.figures.main_grid``; the retrieval-overhead
and TTFT rows come from the cache-off latency model
(``memarena.figures.latency_model``: per-reader cold-prefill curve applied to
each cell's measured prompt lengths, plus measured search time).

    python -m memarena.figures.gen_main_table                 # camera-ready: 144-item D6, label leak
    python -m memarena.figures.gen_main_table --population all --leak flag   # May-2026 table
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from eval.src.permission_metrics import LEAK_FLAG, LEAK_LABEL, POPULATION_ALL, POPULATION_FACT
from memarena.figures.main_grid import BACKENDS, READER_LABEL, READERS, Stat, main_grid_stats
from memarena.figures.latency_model import latency_model
from memarena.figures.paths import table_path

ROWS = (("Rec", "Rec."), ("Rea", "Rea."), ("Trust", "Trust."), ("Avg", "Avg"))
LATENCY_KEY = {"vanilla": "vanilla", "rag": "inmem", "memobase": "memobase", "memsearch": "memsearch", "oracle": "oracle"}
TTFT_OMITTED = {"7b"}  # Mistral-7B was not calibrated on the Spark rig (App. lat:caveats)
UP = r"\,{{\tiny\textcolor{{green!55!black}}{{$\uparrow${:.1f}}}}}"
DOWN = r"\,{{\tiny\textcolor{{red!70!black}}{{$\downarrow${:.1f}}}}}"


def _fmt_cell(stat: Stat, vanilla: Stat | None, bold: bool) -> str:
    mean, sd = round(stat.mean, 1), round(stat.sd, 1)
    text = f"{mean:.1f}{{\\scriptsize$\\pm${sd:.1f}}}"
    if bold:
        text = f"\\textbf{{{text}}}"
    if vanilla is not None:
        delta = round(stat.mean - vanilla.mean, 1)
        if abs(delta) >= 0.1:
            text += (UP if delta > 0 else DOWN).format(abs(delta))
    return text


def ms(x: float) -> str:
    """Whole milliseconds, halves rounded up (as everywhere in the paper)."""
    return f"{int(x + 0.5)}"


def render_body(stats: Dict[Tuple[str, str], Dict[str, Stat]]) -> str:
    lat = latency_model()["cells"]
    overhead = [ms(lat[f"{LATENCY_KEY[b]}_0_6b"]["t_search_p50_ms"]) for b in BACKENDS]
    lines = [r"\multicolumn{2}{l}{Mem.\ retrieval overhead/ms} & " + " & ".join(overhead) + r" \\", r"\midrule"]
    for i, r in enumerate(READERS):
        n_rows = len(ROWS) + (0 if r in TTFT_OMITTED else 1)
        for j, (metric, label) in enumerate(ROWS):
            best = max(BACKENDS, key=lambda b: round(stats[(b, r)][metric].mean, 1))
            cells = [
                _fmt_cell(stats[(b, r)][metric], None if b == "vanilla" else stats[("vanilla", r)][metric], b == best)
                for b in BACKENDS
            ]
            head = f"\\multirow{{{n_rows}}}{{*}}{{{READER_LABEL[r]}}}" if j == 0 else ""
            lines.append(f"{head} & {label} & " + " & ".join(cells) + r" \\")
        if r not in TTFT_OMITTED:
            ttft = [ms(lat[f"{LATENCY_KEY[b]}_{r}"]["ttft_p50_ms"]) for b in BACKENDS]
            lines.append(r" & TTFT/ms & " + " & ".join(ttft) + r" \\")
        if i < len(READERS) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the main results table body.")
    ap.add_argument("--population", choices=(POPULATION_FACT, POPULATION_ALL), default=POPULATION_FACT)
    ap.add_argument("--leak", choices=(LEAK_LABEL, LEAK_FLAG), default=LEAK_LABEL)
    ap.add_argument("--instances", type=Path, default=None, help="d4_permission.jsonl (default: $MEMARENA_DATASET_DIR)")
    ap.add_argument("--out", type=Path, default=None, help="output .tex (default: <artifact root>/tables/main_SML_body.tex)")
    args = ap.parse_args(argv)

    stats = main_grid_stats(args.population, args.leak, args.instances)
    out = args.out or table_path("main_SML_body.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_body(stats), encoding="utf-8")
    print(f"[gen_main_table] population={args.population} leak={args.leak} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
