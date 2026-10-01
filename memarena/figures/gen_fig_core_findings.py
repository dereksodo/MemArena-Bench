"""Combined figure 2 for paper Sec 7: D6 two-attractor scatter (panel a) +
TTFT decomposition (panel b), rendered as a single 1x2 matplotlib figure.

Panel A reuses _draw_panel_a / _per_cell_tp_stats from gen_fig_d6_finding1.py.
Panel B inlines the TTFT-decomposition stacked-bar logic from
gen_fig_ablation_findings.py::figure_f2 (numbers from Table 3 main_SML and the
T_search row of ttft_main).

Panel B reads T_search and TTFT per cell from the cache-off latency model
(``memarena.figures.latency_model``, the source of the main table's TTFT row).

Output: <artifact root>/figures/fig_core_findings.{pdf,png}
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# Reuse data + drawing helpers from the standalone d6 script.
from memarena.figures.gen_fig_d6_finding1 import (
    _draw_panel_a, _per_cell_tp_stats,
    BACKENDS, BACKEND_COLOR, BACKEND_LABEL,
    READER_MARKER,
)
from memarena.figures.gen_main_table import ms
from memarena.figures.latency_model import latency_model
from memarena.figures.paper_data import MODEL_ORDER, MODEL_TEX
from memarena.figures.paths import figures_dir

OUT_DIR = figures_dir()
PANEL_B_CELLS = [  # (reader tag, reader label, latency-fit backend key, backend label)
    ("0_6b", "0.6B", "vanilla", "Vanilla"),
    ("0_6b", "0.6B", "inmem", "RAG"),
    ("0_6b", "0.6B", "memobase", "Memobase"),
    ("0_6b", "0.6B", "memsearch", "MemSearch"),
    ("32b", "32B", "vanilla", "Vanilla"),
    ("32b", "32B", "inmem", "RAG"),
    ("32b", "32B", "memobase", "Memobase"),
    ("32b", "32B", "memsearch", "MemSearch"),
]


def _ttft_cells():
    """(reader_label, backend_label, T_search_ms, T_prefill_ms) from the latency model."""
    lat = latency_model()["cells"]
    cells = []
    for r, rl, bk, bl in PANEL_B_CELLS:
        c = lat[f"{bk}_{r}"]
        # Round the total exactly as the main table's TTFT row does, so bar
        # labels match it; the prefill segment is the remainder.
        search = int(ms(c["t_search_p50_ms"]))
        total = int(ms(c["ttft_p50_ms"]))
        cells.append((rl, bl, search, total - search))
    return cells


# The figure is drawn at its printed size (\textwidth = 5.5 in, height kept at
# the 2.41 in the paper reserves), so font sizes below are printed points.
FIG_W, FIG_H = 5.5, 2.41
FS = 7.0          # every label, tick and legend entry
FS_PANEL = 8.0    # "(a) ..." / "(b) ..." panel titles
SEARCH_COLOR, PREFILL_COLOR = "#1f77b4", "#ff7f0e"


def draw_panel_b(ax) -> None:
    """Stacked-bar TTFT decomposition: 8 cells, search vs prefill.
    Adds a magnifier-style inset above the Qwen3-0.6B bars so the short bars
    (dwarfed by Qwen3-32B-AWQ's scale) remain readable.
    """
    x = np.array([0, 1, 2, 3, 4.5, 5.5, 6.5, 7.5])
    cells = _ttft_cells()
    search = np.array([c[2] for c in cells])
    prefill = np.array([c[3] for c in cells])
    total = search + prefill
    labels = [c[1] for c in cells]

    ax.bar(x, search, width=0.85, color=SEARCH_COLOR, edgecolor="white", lw=0.4,
           label="Memory search")
    ax.bar(x, prefill, width=0.85, bottom=search, color=PREFILL_COLOR,
           edgecolor="white", lw=0.4, label="LLM prefill")
    # Headroom above the tallest bar for the group labels and the magnifier inset
    # (the legend sits with panel A's under the figure).
    ymax = max(total) * 1.27
    for xi, t in zip(x, total):
        ax.text(xi, t + ymax * 0.01, f"{t}", ha="center", va="bottom",
                fontsize=FS, color="#333")
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.6)
    # Backend names slanted, so they fit at 7 pt under bars ~18 pt apart.
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS, rotation=25, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=FS, length=2, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    # Group labels (Qwen3-0.6B / Qwen3-32B-AWQ) along the top of the axes.
    ax.text(0.29, 0.97, "Qwen3-0.6B", transform=ax.transAxes,
            ha="center", va="top", fontsize=FS, fontweight="bold", color="#222")
    ax.text(0.79, 0.97, "Qwen3-32B-AWQ", transform=ax.transAxes,
            ha="center", va="top", fontsize=FS, fontweight="bold", color="#222")
    ax.axvline((x[3] + x[4]) / 2, color="#bbbbbb", lw=0.6, ls="--", alpha=0.8)
    ax.set_ylabel("End-to-end TTFT (ms)", fontsize=FS, labelpad=2)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.18)

    # === Magnifier inset over the 0.6B bars =================================
    # 0.6B bars sit at x=0..3 with totals of a few hundred ms; on the 0..ymax
    # scale they are barely visible. Inset shows the same bars on their own
    # scale, anchored above them so it reads like a magnifying glass.
    src_x0, src_x1 = -0.55, 3.55
    src_y0, src_y1 = 0.0, max(total[:4]) * 1.22
    axins = ax.inset_axes(
        [0.135, 0.20, 0.33, 0.56],   # [x, y, w, h] in axes fraction
        xlim=(src_x0, src_x1), ylim=(src_y0, src_y1),
    )
    axins.bar(x[:4], search[:4], width=0.85, color=SEARCH_COLOR,
              edgecolor="white", lw=0.4)
    axins.bar(x[:4], prefill[:4], width=0.85, bottom=search[:4],
              color=PREFILL_COLOR, edgecolor="white", lw=0.4)
    for xi, t in zip(x[:4], total[:4]):
        axins.text(xi, t + src_y1 * 0.02, f"{t}", ha="center", va="bottom",
                   fontsize=FS, color="#333")
    axins.set_xticks([])
    axins.set_yticks([0, 100, 200])
    axins.tick_params(axis="y", labelsize=FS, length=2, width=0.5, pad=1.5)
    axins.set_title("zoomed (0.6B)", fontsize=FS, fontweight="bold",
                    color="#444", pad=2)
    for spine in axins.spines.values():
        spine.set_edgecolor("#666")
        spine.set_linewidth(0.6)
    axins.grid(axis="y", alpha=0.18)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig_core_findings.pdf"
    out_png = OUT_DIR / "fig_core_findings.png"

    tp = _per_cell_tp_stats()

    # Fixed layout in inches (no tight bbox), so the PDF is exactly
    # FIG_W x FIG_H and \includegraphics[width=\textwidth] prints it 1:1.
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_bottom, ax_top = 0.72, FIG_H - 0.20
    ax_h = (ax_top - ax_bottom) / FIG_H
    ax_a = fig.add_axes([0.49 / FIG_W, ax_bottom / FIG_H, 2.09 / FIG_W, ax_h])
    ax_b = fig.add_axes([3.01 / FIG_W, ax_bottom / FIG_H, 2.35 / FIG_W, ax_h])

    # Panel A — scatter
    _draw_panel_a(ax_a, tp, compact=True)

    # One legend under both panels: backends on row 1 (colour), readers on row 2
    # (marker), and panel B's two bar segments as the last column.
    backend_handles = [plt.Line2D([0], [0], marker="o", lw=0, markersize=5,
                                  markerfacecolor=BACKEND_COLOR[b],
                                  markeredgecolor="white", markeredgewidth=0.4,
                                  label=BACKEND_LABEL[b]) for b in BACKENDS]
    reader_handles = [plt.Line2D([0], [0], marker=READER_MARKER[r], lw=0,
                                 markersize=4.5, color="black", markeredgewidth=0.6,
                                 markerfacecolor="white",
                                 label=MODEL_TEX[r]) for r in MODEL_ORDER]
    bar_handles = [Patch(facecolor=SEARCH_COLOR, edgecolor="none", label="Memory search"),
                   Patch(facecolor=PREFILL_COLOR, edgecolor="none", label="LLM prefill")]
    # ncol=6 fills column-wise, so interleave to get backends on row 1, readers on row 2.
    handles = [h for pair in zip(backend_handles, reader_handles) for h in pair] + bar_handles
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                     ncol=6, fontsize=FS, frameon=False, handletextpad=0.25,
                     columnspacing=0.9, labelspacing=0.35, borderaxespad=0.15,
                     handlelength=1.0, handleheight=0.7)

    # Panel B — TTFT bars
    draw_panel_b(ax_b)

    # Panel titles above each panel.
    for ax, title in ((ax_a, "(a) Permission-aware access"),
                      (ax_b, "(b) End-to-end TTFT decomposition")):
        ax.text(0.5, 1.0 + 0.06 / (ax_top - ax_bottom), title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=FS_PANEL, fontweight="bold")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
