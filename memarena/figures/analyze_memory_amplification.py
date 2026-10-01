#!/usr/bin/env python3
"""Generate Figure (``fig:memory-amplification``): memory-system lift vs model tier.

Finding 2 of MemArena is rewritten around two complementary claims:

* **Pareto substitute (Claim A).** On the Avg metric (macro-mean over
  D1..D5), Memobase paired with Qwen3-0.6B beats Vanilla paired with
  Qwen3-32B despite the 53x parameter gap.
* **Compute amplification (Claim B).** The Memobase-minus-Vanilla Avg lift
  *grows* with model size rather than shrinks, contradicting the common
  "memory democratizes small models" narrative.

This script reproduces the relevant numbers from ``paper_data.py`` and emits a
two-panel figure:

* Panel (a) -- Avg accuracy vs model tier for Vanilla, Memobase, MemOS. The
  widening gap visualises Claim B.
* Panel (b) -- Memobase minus Vanilla Avg lift per model tier. A monotone
  increasing bar-chart is the direct visual of Claim B.

Output
------
``paper/figures/fig_memory_amplification.pdf`` (and ``.png``).
The stdout delta table prints the Memobase/MemOS lift per tier plus the 53x
Pareto comparison, for use in the paper text.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python paper/analyze_memory_amplification.py`` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

from paper_data import (  # noqa: E402
    CAT_RECALL, CAT_REASONING, CAT_TRUST,
    MODEL_TEX,
    BACKEND_TEX,
    DEFAULT_RUN,
    Stat,
    aggregate_seeds, category_accuracy, load_all_cells,
)
from paths import figure_path  # noqa: E402

OUT_PDF = figure_path("fig_memory_amplification.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")

# We amplify-compare Vanilla against the two structured backends that actually
# cover the full model grid. Mem0 is partial; RAG is not a structured-memory
# system in the sense of Finding 2, and Oracle sets a different oracle-ceiling
# story (reported separately in the main table).
BACKENDS = ["vanilla", "memobase", "memos"]

# Four tiers where all three backends exist.  Mistral-7B is excluded here
# because memobase/memos cells are system-incompatible (dashes in the main
# table); including it would produce missing points on two of three lines and
# is misleading in the amplification plot.
TIERS = ["0_6b", "llama3b", "8b", "32b"]

# Consistent, readable colours across the two panels.  Gray for the
# no-memory baseline, red/orange for the structured families to match the
# scatter plot's conventions (analyze_rec_rea_scatter.py).
BACKEND_COLORS = {
    "vanilla":  "#7f7f7f",
    "memobase": "#d62728",
    "memos":    "#ff7f0e",
}

# Panel (b): single bar per tier of (Memobase - Vanilla) lift.
LIFT_COLOR = "#d62728"


def _cell_overall_avg(cell):
    """Macro-mean over D1..D5, equal weight per dimension.

    Duplicated locally (rather than imported from gen_tables.py, whose module
    name starts with a non-package pattern) to keep this script self-contained;
    the definition is identical and must stay in lock-step.
    """
    parts = [category_accuracy(cell, [i]) for i in [0, 1, 2, 3, 4]]
    parts = [p for p in parts if p is not None]
    return sum(parts) / len(parts) if parts else None


def _cell_rec(cell):
    return category_accuracy(cell, CAT_RECALL)


def _cell_rea(cell):
    return category_accuracy(cell, CAT_REASONING)


def _cell_conf(cell):
    # Conf = D5 Confabulation only, per the main-table convention.
    return category_accuracy(cell, CAT_TRUST)


METRICS = {
    "Rec":  _cell_rec,
    "Rea":  _cell_rea,
    "Conf": _cell_conf,
    "Avg":  _cell_overall_avg,
}


def _compute_stats(grid):
    """Return ``stats[backend][model][metric] = Stat``."""
    out = {}
    for backend in BACKENDS:
        out[backend] = {}
        for model in TIERS:
            out[backend][model] = {}
            for metric_name, sel in METRICS.items():
                out[backend][model][metric_name] = aggregate_seeds(
                    grid, backend, model, sel
                )
    return out


def _print_table(stats):
    """Emit a plain-text table matching the paper's rounding (one decimal pp)."""
    header = f"{'Backend':<10} {'Model':<14} " + " ".join(
        f"{m:>8}" for m in ["Rec", "Rea", "Conf", "Avg"]
    )
    print(header)
    print("-" * len(header))
    for backend in BACKENDS:
        for model in TIERS:
            row = stats[backend][model]
            cells = []
            for m in ["Rec", "Rea", "Conf", "Avg"]:
                st: Stat = row[m]
                cells.append(f"{st.mean*100:>8.1f}" if st.n else f"{'--':>8}")
            print(f"{BACKEND_TEX[backend]:<10} {MODEL_TEX[model]:<14} " + " ".join(cells))
        print()

    # Delta / amplification block.
    print("Amplification (Avg pp lift over Vanilla):")
    print(f"{'Model':<14} {'Memobase-Vanilla':>18} {'MemOS-Vanilla':>16}")
    for model in TIERS:
        van = stats["vanilla"][model]["Avg"].mean
        mb  = stats["memobase"][model]["Avg"].mean
        mo  = stats["memos"][model]["Avg"].mean
        d_mb = (mb - van) * 100 if (van and mb) else float("nan")
        d_mo = (mo - van) * 100 if (van and mo) else float("nan")
        print(f"{MODEL_TEX[model]:<14} {d_mb:>18.1f} {d_mo:>16.1f}")
    print()

    # Claim A Pareto check.
    mb_small = stats["memobase"]["0_6b"]["Avg"].mean * 100
    van_big  = stats["vanilla"]["32b"]["Avg"].mean * 100
    print("Pareto (Claim A):")
    print(f"  Memobase x Qwen3-0.6B  Avg = {mb_small:.1f}")
    print(f"  Vanilla  x Qwen3-32B   Avg = {van_big:.1f}")
    print(f"  gap = {mb_small - van_big:+.1f} pp  (positive means small+memory wins)")


def _render(stats, out_pdf: Path) -> None:
    """Two-panel figure: (a) Avg vs tier per backend, (b) (Memobase-Vanilla) lift."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # ----- Panel (a): Avg vs tier, one line per backend -----
    xs = list(range(len(TIERS)))
    xticklabels = [MODEL_TEX[m] for m in TIERS]
    for backend in BACKENDS:
        means = []
        stds = []
        for model in TIERS:
            st = stats[backend][model]["Avg"]
            means.append(st.mean * 100 if st.n else None)
            stds.append(st.std * 100 if st.n >= 2 else 0.0)
        axA.errorbar(
            xs, means, yerr=stds,
            marker="o", linewidth=2.0, markersize=7,
            color=BACKEND_COLORS[backend],
            label=BACKEND_TEX[backend],
            capsize=3,
        )

    axA.set_xticks(xs)
    axA.set_xticklabels(xticklabels, rotation=15, ha="right")
    axA.set_xlabel("Extractor / reader model")
    axA.set_ylabel("Avg = mean(D1..D5)  (pp)")
    axA.set_title("(a) Memory widens the gap at larger scale")
    axA.grid(True, alpha=0.3)
    axA.legend(loc="lower right", fontsize=9, frameon=False)
    axA.set_ylim(0, 100)

    # Annotate the 53x-Pareto comparison: horizontal dashed guideline from
    # Memobase x 0.6B across to meet Vanilla x 32B for reader eye-tracking.
    mb_small = stats["memobase"]["0_6b"]["Avg"].mean * 100
    van_big  = stats["vanilla"]["32b"]["Avg"].mean * 100
    axA.axhline(mb_small, color=BACKEND_COLORS["memobase"], lw=0.7, ls=":", alpha=0.6)
    axA.annotate(
        f"Memobase x 0.6B = {mb_small:.1f}\nVanilla x 32B = {van_big:.1f}",
        xy=(xs[-1], van_big),
        xytext=(0.02, 0.58), textcoords="axes fraction",
        fontsize=8, color="#333333",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#888888"),
    )

    # ----- Panel (b): Memobase - Vanilla Avg lift per tier -----
    lifts_mb = []
    lifts_mo = []
    for model in TIERS:
        van = stats["vanilla"][model]["Avg"].mean
        mb  = stats["memobase"][model]["Avg"].mean
        mo  = stats["memos"][model]["Avg"].mean
        lifts_mb.append((mb - van) * 100)
        lifts_mo.append((mo - van) * 100)

    width = 0.38
    xs_arr = [i for i in range(len(TIERS))]
    xs_mb  = [i - width/2 for i in xs_arr]
    xs_mo  = [i + width/2 for i in xs_arr]

    axB.bar(xs_mb, lifts_mb, width,
            color=BACKEND_COLORS["memobase"], edgecolor="black", linewidth=0.6,
            label="Memobase - Vanilla")
    axB.bar(xs_mo, lifts_mo, width,
            color=BACKEND_COLORS["memos"], edgecolor="black", linewidth=0.6,
            label="MemOS - Vanilla")

    for x, v in zip(xs_mb, lifts_mb):
        axB.text(x, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=8, color="#333333")
    for x, v in zip(xs_mo, lifts_mo):
        axB.text(x, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=8, color="#333333")

    axB.set_xticks(xs_arr)
    axB.set_xticklabels(xticklabels, rotation=15, ha="right")
    axB.set_xlabel("Extractor / reader model")
    axB.set_ylabel("Avg lift over Vanilla  (pp)")
    axB.set_title("(b) Memory lift grows with model size")
    axB.grid(True, axis="y", alpha=0.3)
    axB.legend(loc="upper left", fontsize=9, frameon=False)
    axB.set_ylim(0, max(max(lifts_mb), max(lifts_mo)) * 1.15)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    grid = load_all_cells(DEFAULT_RUN)
    stats = _compute_stats(grid)
    _print_table(stats)
    _render(stats, OUT_PDF)
    print(f"[fig:memory-amplification] wrote {OUT_PDF}")
    print(f"[fig:memory-amplification] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
