#!/usr/bin/env python3
"""Generate Figure 3 (``fig:rec-rea-scatter``): Recall vs Reasoning trade-off.

The paper argues that structured-memory backends (Memobase / MemSearch) do not
simply dominate Oracle on every dimension: instead they rotate the operating
point toward higher Reasoning at the cost of less consistent Recall. The
simplest way to surface that rotation is a scatter plot where every evaluated
cell is a single point, colored by backend family and marked by extractor
tier. If the story is right, structured cells cluster upper-left while
Oracle / RAG sit on a diagonal band.

Output
------
``paper/figures/fig_rec_rea_scatter.pdf`` (and ``.png`` for convenience).

Source data
-----------
Rec and Rea per (backend, reader) from :mod:`memarena.figures.main_grid`
(the numbers of the main results table): mean over seeds s2/s3/s4, with the
seed SD as error bars.

Design notes
------------
* Legend is split by **backend family** (color) and **extractor model tier**
  (marker shape). This keeps the legend compact even with 6 backends × 5
  models = 30 possible cells.
* The dashed y=x diagonal is drawn for reference but labelled only in the
  figure caption — matplotlib legend entries for reference lines clutter the
  corner.
* Text annotations are kept minimal: a single tag per structured-memory
  cluster at a small serif. x/y labels use "pp" (percentage points) to
  emphasise that the axes are percentages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from memarena.figures.main_grid import main_grid_stats
from memarena.figures.paper_data import BACKEND_TEX, MODEL_ORDER, MODEL_TEX
from memarena.figures.paths import figure_path

OUT_PDF = figure_path("fig_rec_rea_scatter.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")

# Backends to plot (excludes ``mem0`` — it only exists at one weak endpoint and
# would clutter the main plot; a dedicated appendix figure can cover it).
PLOT_BACKENDS = ["vanilla", "rag", "oracle", "memobase", "memsearch"]

# Matplotlib does not have a single consistent colormap for "six families", so
# we pick explicit hues. Blues for baselines (vanilla/rag/oracle share a cool
# hue ramp), reds/oranges for structured. Colourblind-friendly check via
# https://davidmathlogic.com/colorblind.
BACKEND_COLORS = {
    "vanilla":  "#7f7f7f",   # gray
    "rag":      "#1f77b4",   # blue
    "oracle":   "#2ca02c",   # green
    "memobase": "#d62728",   # red
    "memsearch": "#ff7f0e",  # orange
    "mem0":     "#9467bd",   # purple
}

# Marker per model tier; ordered from smallest to largest extractor.
MODEL_MARKERS = {
    "0_6b":    "o",
    "llama3b": "s",
    "7b":      "^",
    "8b":      "D",
    "32b":     "P",
}

# Marker sizes scaled slightly by model tier so overlapping cells still read.
MODEL_SIZES = {
    "0_6b":    60,
    "llama3b": 70,
    "7b":      80,
    "8b":      90,
    "32b":     110,
}


def _collect_points(stats):
    """Yield ``(backend, model, rec_mean, rec_std, rea_mean, rea_std, n_seeds)``."""
    for backend in PLOT_BACKENDS:
        for model in MODEL_ORDER:
            rec, rea = stats[(backend, model)]["Rec"], stats[(backend, model)]["Rea"]
            # main_grid reports percentages; the axes are fractions.
            yield (backend, model, rec.mean / 100, rec.sd / 100, rea.mean / 100, rea.sd / 100, rec.n)


def _render(points, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    # The y=x reference is a weak visual prior that "more Recall → more
    # Reasoning"; structured cells explicitly break this assumption, so the
    # diagonal is the right reference to draw.
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1.0, ls="--", zorder=0)

    # Scatter. Each cell is one point with x/y error bars.
    seen_backends = set()
    seen_models = set()
    for backend, model, rec_mean, rec_std, rea_mean, rea_std, n in points:
        ax.errorbar(
            rec_mean, rea_mean,
            xerr=rec_std, yerr=rea_std,
            fmt="none", ecolor=BACKEND_COLORS[backend], alpha=0.55,
            elinewidth=0.8, capsize=2, zorder=1,
        )
        ax.scatter(
            rec_mean, rea_mean,
            marker=MODEL_MARKERS[model],
            s=MODEL_SIZES[model],
            color=BACKEND_COLORS[backend],
            edgecolor="black", linewidth=0.6,
            label=None,  # legend built manually below
            zorder=2,
        )
        seen_backends.add(backend)
        seen_models.add(model)

    # Manual two-legend layout so the colour/shape decoupling stays readable.
    from matplotlib.lines import Line2D

    backend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=BACKEND_COLORS[b], markersize=9,
               markeredgecolor="black", markeredgewidth=0.6,
               label=BACKEND_TEX[b])
        for b in PLOT_BACKENDS if b in seen_backends
    ]
    model_handles = [
        Line2D([0], [0], marker=MODEL_MARKERS[m], color="w",
               markerfacecolor="#555555", markersize=9,
               markeredgecolor="black", markeredgewidth=0.6,
               label=MODEL_TEX[m])
        for m in MODEL_ORDER if m in seen_models
    ]

    leg1 = ax.legend(handles=backend_handles, title="Backend",
                     loc="upper left", fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Extractor",
              loc="lower right", fontsize=8, title_fontsize=9, frameon=False)

    ax.set_xlabel("Recall (D1+D2 mean, pp)")
    ax.set_ylabel("Reasoning (D3+D4 mean, pp)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title("Recall vs Reasoning")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points = list(_collect_points(main_grid_stats()))
    print(f"[fig:rec-rea-scatter] plotting {len(points)} cells")
    _render(points, OUT_PDF)
    print(f"[fig:rec-rea-scatter] wrote {OUT_PDF}")
    print(f"[fig:rec-rea-scatter] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
