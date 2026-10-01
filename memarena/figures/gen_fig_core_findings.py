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


def draw_panel_b(ax) -> None:
    """Stacked-bar TTFT decomposition: 8 cells, search vs prefill.
    Adds a magnifier-style inset above the Qwen3-0.6B bars so the short bars
    (totals 32-213 ms, dwarfed by 32B's 290-3630 ms scale) remain readable.
    """
    x = np.array([0, 1, 2, 3, 4.7, 5.7, 6.7, 7.7])
    cells = _ttft_cells()
    search = np.array([c[2] for c in cells])
    prefill = np.array([c[3] for c in cells])
    total = search + prefill
    labels = [c[1] for c in cells]

    ax.bar(x, search, width=0.85, color="#1f77b4", edgecolor="white", lw=0.7,
           label="Memory search")
    ax.bar(x, prefill, width=0.85, bottom=search, color="#ff7f0e",
           edgecolor="white", lw=0.7, label="LLM prefill")
    for xi, t in zip(x, total):
        ax.text(xi, t + max(total) * 0.012, f"{t}", ha="center", va="bottom",
                fontsize=8.2, color="#333")
    ymax = max(total) * 1.30  # extra headroom so the magnifier inset has room
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.8)
    # Group labels (Qwen3-0.6B / Qwen3-32B-AWQ) on one line at the top of each half.
    ax.text(0.25, 0.95, "Qwen3-0.6B", transform=ax.transAxes,
            ha="center", va="top", fontsize=12, fontweight="bold", color="#222")
    ax.text(0.78, 0.95, "Qwen3-32B-AWQ", transform=ax.transAxes,
            ha="center", va="top", fontsize=12, fontweight="bold", color="#222")
    ax.axvline(4.0 - 0.15, color="#bbbbbb", lw=0.7, ls="--", alpha=0.8)
    ax.set_ylabel("End-to-end TTFT (ms)", fontsize=10)
    ax.set_ylim(0, ymax)
    # Legend below the panel (same row as panel A's legend), which frees the
    # upper-left corner for a larger magnifier inset.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              fontsize=11, frameon=False, handlelength=1.6, columnspacing=1.5)
    ax.grid(axis="y", alpha=0.18)

    # === Magnifier inset over the 0.6B bars =================================
    # 0.6B bars sit at x=0..3 with totals of a few hundred ms; on the 0..ymax
    # scale they are barely visible. Inset shows the same bars on their own
    # scale, anchored above them with connector lines so it reads
    # like a magnifying glass.
    src_x0, src_x1 = -0.55, 3.55
    src_y0, src_y1 = 0.0, max(total[:4]) * 1.18  # ~250 ms
    axins = ax.inset_axes(
        [0.08, 0.17, 0.38, 0.64],   # [x, y, w, h] in axes fraction
        xlim=(src_x0, src_x1), ylim=(src_y0, src_y1),
    )
    axins.bar(x[:4], search[:4], width=0.85, color="#1f77b4",
              edgecolor="white", lw=0.7)
    axins.bar(x[:4], prefill[:4], width=0.85, bottom=search[:4],
              color="#ff7f0e", edgecolor="white", lw=0.7)
    for xi, t in zip(x[:4], total[:4]):
        axins.text(xi, t + src_y1 * 0.02, f"{t}", ha="center", va="bottom",
                   fontsize=9.5, color="#333")
    axins.set_xticks([])
    axins.tick_params(axis="y", labelsize=9.0)
    axins.set_title("zoomed (0.6B)", fontsize=10.0, fontweight="bold",
                    color="#444", pad=2)
    for spine in axins.spines.values():
        spine.set_edgecolor("#666")
        spine.set_linewidth(1.0)
    axins.grid(axis="y", alpha=0.18)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig_core_findings.pdf"
    out_png = OUT_DIR / "fig_core_findings.png"

    tp = _per_cell_tp_stats()

    # 1x2 layout. Panel A is square-ish; panel B is wide. Equal widths give
    # both subplots the same vertical extent.
    fig = plt.figure(figsize=(13.0, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.00, 1.05], wspace=0.16)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # Panel A — scatter
    _draw_panel_a(ax_a, tp)

    # Legend below panel A (backends on one row, readers on the next), above the
    # "(a)" label, so neither panel loses width to it.
    backend_handles = [plt.Line2D([0], [0], marker="o", lw=0, markersize=8,
                                  markerfacecolor=BACKEND_COLOR[b],
                                  markeredgecolor="white", markeredgewidth=0.5,
                                  label=BACKEND_LABEL[b]) for b in BACKENDS]
    reader_handles = [plt.Line2D([0], [0], marker=READER_MARKER[r], lw=0,
                                 markersize=8, color="black",
                                 markerfacecolor="white",
                                 label=MODEL_TEX[r]) for r in MODEL_ORDER]
    # ncol=5 fills column-wise, so interleave to get backends on row 1, readers on row 2.
    handles = [h for pair in zip(backend_handles, reader_handles) for h in pair]
    leg = ax_a.legend(handles=handles,
                      loc="upper center", bbox_to_anchor=(0.5, -0.20),
                      ncol=5, fontsize=11, frameon=False, handletextpad=0.2,
                      columnspacing=0.5, labelspacing=0.3, markerscale=1.1)

    # Panel B — TTFT bars
    draw_panel_b(ax_b)

    # Subplot labels below each panel
    # Both labels at the same axes-y so they align across the two panels
    # (axes have equal height in the gridspec, so axes-y maps to the same display-y).
    label_y = -0.43
    label_a = ax_a.text(0.5, label_y, "(a) Permission-aware access",
                        transform=ax_a.transAxes, ha="center", va="top",
                        fontsize=17, fontweight="bold")
    label_b = ax_b.text(0.5, label_y, "(b) End-to-end TTFT decomposition",
                        transform=ax_b.transAxes, ha="center", va="top",
                        fontsize=17, fontweight="bold")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15,
                bbox_extra_artists=[leg, label_a, label_b])
    fig.savefig(out_png, bbox_inches="tight", dpi=160, pad_inches=0.15,
                bbox_extra_artists=[leg, label_a, label_b])
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
