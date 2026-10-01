#!/usr/bin/env python3
"""Generate Figure 3 (``fig:rec-rea-scatter``): Recall vs Reasoning trade-off.

The paper argues that structured-memory backends (Memobase / MemSearch) do not
simply dominate Oracle on every dimension: instead they rotate the operating
point toward higher Reasoning at the cost of less consistent Recall. The
simplest way to surface that rotation is a scatter plot where every evaluated
cell is a single point, colored by backend family and marked by reader
model. If the story is right, structured cells cluster upper-left while
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
* Legend is split by **backend family** (color) and **reader model**
  (marker shape). This keeps the legend compact even with 6 backends × 5
  models = 30 possible cells.
* The dashed y=x diagonal is drawn for reference but labelled only in the
  figure caption — matplotlib legend entries for reference lines clutter the
  corner.
* Both axes are accuracies in percent (0--100), the unit of the main
  results table.
* The figure is drawn at its printed size (0.62\textwidth = 3.41 in) on a
  fixed canvas, so every font prints at its nominal size (>= 7 pt).
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

# Marker per reader; ordered from smallest to largest model.
MODEL_MARKERS = {
    "0_6b":    "o",
    "llama3b": "s",
    "7b":      "^",
    "8b":      "D",
    "32b":     "P",
}

# Marker sizes (pt^2 at print size) scaled slightly by model so overlapping
# cells still read.
MODEL_SIZES = {
    "0_6b":    22,
    "llama3b": 26,
    "7b":      30,
    "8b":      34,
    "32b":     42,
}
FIG_W, FIG_H = 3.41, 3.42   # printed size in inches (0.62\textwidth)
FS = 7.0


def _collect_points(stats):
    """Yield ``(backend, model, rec_mean, rec_std, rea_mean, rea_std, n_seeds)``."""
    for backend in PLOT_BACKENDS:
        for model in MODEL_ORDER:
            rec, rea = stats[(backend, model)]["Rec"], stats[(backend, model)]["Rea"]
            # main_grid reports percentages, the unit of both axes.
            yield (backend, model, rec.mean, rec.sd, rea.mean, rea.sd, rec.n)


def _render(points, out_pdf: Path) -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    side = 2.94  # square axes, inches
    ax = fig.add_axes([0.37 / FIG_W, 0.28 / FIG_H, side / FIG_W, side / FIG_H])

    # The y=x reference is a weak visual prior that "more Recall → more
    # Reasoning"; structured cells explicitly break this assumption, so the
    # diagonal is the right reference to draw.
    ax.plot([0, 100], [0, 100], color="#cccccc", lw=0.8, ls="--", zorder=0)

    # Scatter. Each cell is one point with x/y error bars.
    seen_backends = set()
    seen_models = set()
    for backend, model, rec_mean, rec_std, rea_mean, rea_std, n in points:
        ax.errorbar(
            rec_mean, rea_mean,
            xerr=rec_std, yerr=rea_std,
            fmt="none", ecolor=BACKEND_COLORS[backend], alpha=0.55,
            elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=1,
        )
        ax.scatter(
            rec_mean, rea_mean,
            marker=MODEL_MARKERS[model],
            s=MODEL_SIZES[model],
            color=BACKEND_COLORS[backend],
            edgecolor="black", linewidth=0.4,
            label=None,  # legend built manually below
            zorder=2,
        )
        seen_backends.add(backend)
        seen_models.add(model)

    # Manual two-legend layout so the colour/shape decoupling stays readable.
    from matplotlib.lines import Line2D

    backend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=BACKEND_COLORS[b], markersize=5.5,
               markeredgecolor="black", markeredgewidth=0.4,
               label=BACKEND_TEX[b])
        for b in PLOT_BACKENDS if b in seen_backends
    ]
    model_handles = [
        Line2D([0], [0], marker=MODEL_MARKERS[m], color="w",
               markerfacecolor="#555555", markersize=5.5,
               markeredgecolor="black", markeredgewidth=0.4,
               label=MODEL_TEX[m])
        for m in MODEL_ORDER if m in seen_models
    ]

    legend_kw = dict(fontsize=FS, title_fontsize=FS, frameon=False, handletextpad=0.3,
                     labelspacing=0.35, borderaxespad=0.4)
    leg1 = ax.legend(handles=backend_handles, title="Backend", loc="upper left", **legend_kw)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Reader", loc="lower right", **legend_kw)

    ax.set_xlabel("Recall (D1+D2 mean accuracy, %)", fontsize=FS, labelpad=2)
    ax.set_ylabel("Reasoning (D3+D4 mean accuracy, %)", fontsize=FS, labelpad=2)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FS, length=2, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_title("Recall vs Reasoning", fontsize=8, pad=3)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    points = list(_collect_points(main_grid_stats()))
    print(f"[fig:rec-rea-scatter] plotting {len(points)} cells")
    _render(points, OUT_PDF)
    print(f"[fig:rec-rea-scatter] wrote {OUT_PDF}")
    print(f"[fig:rec-rea-scatter] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
