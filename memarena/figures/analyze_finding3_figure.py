#!/usr/bin/env python3
"""Generate the merged 2-panel figure for Finding 3 (``fig:finding3``).

Combines two figures previously kept separate:

  Panel (a): On-device Pareto scatter --- Avg accuracy vs.\\ log
             answer-time energy on DGX Spark, $4$ readers $\\times$ $3$
             backends (Mistral-7B not on Spark), peak-temp colormap,
             within-backend curves, dominance arrow. Was
             ``fig_pareto_dominance``.
  Panel (b): Memobase / MemOS / RAG lift over Vanilla per reader tier.
             Was ``fig_finding2`` panel (b).

Output: ``paper/figures/fig_finding3.{pdf,png}``.

Source data:
- Avg-accuracy values (panels a/b/c): ``analyze_finding2.PUBLISHED_AVG``,
  the single-source-of-truth table mirroring ``main_SML.tex``.
- Energy + peak temp (panel a): live read from
  ``MASim/runs/l_20260408_111046/spark_results_s1/hw/hw_agg_*.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analyze_finding2 import _load_avg_points  # noqa: E402
from paper_data import BACKEND_TEX, MODEL_TEX  # noqa: E402

try:
    from .paths import figure_path
except ImportError:  # pragma: no cover
    from paths import figure_path

# PUBLISHED_AVG-shaped dict computed live from experiments_index.csv via
# analyze_finding2._load_avg_points (default). Set
# MEMARENA_FINDING2_SOURCE=published to fall back to the archived numbers.
PUBLISHED_AVG, _AVG_LABEL, _AVG_HAS_DENSE = _load_avg_points()

REPO_ROOT = Path(__file__).resolve().parents[2]
# 2026-05 sweep V5/V6: per-reader hw probe dirs under out/latency_spark_<r>_s2/hw/
# with naming hw_agg_<backend>_<reader>_s2.json. Legacy: ONE dir with stubs.
SPARK_LATENCY_BASE = REPO_ROOT / "out"
SPARK_HW_DIR = (
    REPO_ROOT
    / "MASim/runs/l_20260408_111046/spark_results_s1/hw"
)  # legacy fallback
OUT_PDF = figure_path("fig_finding3.pdf")

# ---------------------------------------------------------------------------
# Shared panel-style constants
# ---------------------------------------------------------------------------

COLOR = {
    "vanilla":   "#7f7f7f",
    "rag":       "#1f77b4",
    "inmem":     "#1f77b4",
    "dense_rag": "#8ec7ff",
    "memobase":  "#d62728",
    "memsearch": "#9467bd",
    "memos":     "#ff7f0e",
    "oracle":    "#2ca02c",
}

LABEL_A_FULL = {
    "vanilla":   BACKEND_TEX["vanilla"],
    "rag":       BACKEND_TEX["rag"],
    "dense_rag": "Dense RAG",
    "memobase":  BACKEND_TEX["memobase"],
    "memos":     BACKEND_TEX["memos"],
    "oracle":    BACKEND_TEX["oracle"],
}

# ---------------------------------------------------------------------------
# Panel (a): on-device Pareto scatter
# ---------------------------------------------------------------------------

PARETO_MODELS = ["0_6b", "llama3b", "7b", "8b", "32b"]
# Backend names match `experiments_index.csv` / accuracy convention. The
# 2026-05 sweep uses `inmem` (BM25) instead of legacy `baseline_simplerag`,
# and `memsearch` replaces the previous `memobase` cell in main-table-position.
# All five headline backends are plotted on the Pareto panel.
PARETO_BACKENDS = ["vanilla", "inmem", "oracle", "memsearch", "memobase"]
BACKEND_MARKER = {
    "vanilla":   "o",
    "inmem":     "s",
    "oracle":    "*",
    "memsearch": "D",
    "memobase":  "P",
}
BACKEND_LINESTYLE = {
    "vanilla":   (0, (4, 2)),
    "inmem":     (0, (1, 2)),
    "oracle":    (0, (3, 1, 1, 1)),
    "memsearch": "-",
    "memobase":  (0, (5, 1)),
}


def _hw(reader: str, backend: str) -> tuple[float, float] | None:
    """Return ``(total_J_per_query, peak_temp_c)`` for one (reader, backend).

    ``total_J_per_query = build_energy_per_query + answer_energy_per_query``,
    where the build component reads ``hw_agg_<backend>_<reader>_s2_ingest.json``
    if it exists and is 0 otherwise (vanilla / oracle / inmem all have no
    build phase by definition; memsearch / memobase typically would, but
    the current Spark sweep V5/V6 only probed the answer phase, so build
    energy is recorded as 0 until a separate ingest probe lands).

    New layout (2026-05 sweep):
      out/latency_spark_<reader>_s2/hw/hw_agg_<backend>_<reader>_s2.json
    """
    base = SPARK_LATENCY_BASE / f"latency_spark_{reader}_s2" / "hw"
    answer_path = base / f"hw_agg_{backend}_{reader}_s2.json"
    if not answer_path.exists():
        return None
    with answer_path.open() as fh:
        ans = json.load(fh)["per_cell"]
    answer_j = (
        ans.get("mean_gpu_energy_j_per_query")
        or ans.get("mean_energy_j_per_query")
        or 0.0
    )
    peak_t = ans.get("peak_temp_c") or 0.0

    # Build phase (ingest) — measured separately if present.
    build_path = base / f"hw_agg_{backend}_{reader}_s2_ingest.json"
    build_j = 0.0
    if build_path.exists():
        with build_path.open() as fh:
            ing = json.load(fh)["per_cell"]
        # Build energy is one-shot; we amortize per-query the same way the
        # old paper did: split total ingest energy across the answer query
        # count (n_queries on the answer side), so the Pareto x-axis stays
        # in J/query units.
        n_q = max(int(ans.get("n_queries", 0)), 1)
        build_j = float(ing.get("total_gpu_energy_j", 0.0)) / n_q
    return answer_j + build_j, peak_t


def _plot_pareto(ax, fig) -> None:
    cells = []
    missing_hw: list[str] = []
    # PUBLISHED_AVG (from analyze_finding2) keys backends as paper-side
    # labels: rag (BM25 RAG) on the paper side is "inmem" on the CSV side.
    paper_to_csv_backend = {
        "vanilla":   "vanilla",
        "inmem":     "rag",
        "oracle":    "oracle",
        "memsearch": "memsearch",
        "memobase":  "memobase",
    }
    for model in PARETO_MODELS:
        for backend in PARETO_BACKENDS:
            paper_backend = paper_to_csv_backend.get(backend, backend)
            avg = (PUBLISHED_AVG.get(model) or {}).get(paper_backend)
            if avg is None:
                missing_hw.append(f"{backend}/{model}: no Avg")
                continue
            avg_pp = avg[0]
            hw = _hw(model, backend)
            if hw is None or hw[0] is None:
                missing_hw.append(f"{backend}/{model}: hw_agg missing")
                continue
            j_q, t_c = hw
            cells.append((model, backend, avg_pp, j_q, t_c))

    if not cells:
        # No Spark hardware aggregates available; render an explanatory
        # placeholder instead of crashing. Energy/temp data for this panel
        # lives in MASim/runs/.../spark_results_s1/hw/ which is not in the
        # repo or experiments_index.csv.
        ax.text(0.5, 0.5,
                "Panel (a) skipped: hw_agg_*.json not found.\n"
                "Energy/temp probe data must be synced from Spark.",
                ha="center", va="center", fontsize=9, color="#666",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("(a) Pareto: Avg vs.\\ energy on DGX Spark", fontsize=10)
        return

    temps = [c[4] for c in cells]
    norm = Normalize(vmin=min(temps), vmax=max(temps))
    cmap = plt.get_cmap("coolwarm")

    for bk in PARETO_BACKENDS:
        bk_cells = sorted([c for c in cells if c[1] == bk],
                          key=lambda c: PARETO_MODELS.index(c[0]))
        ax.plot([c[3] for c in bk_cells], [c[2] for c in bk_cells],
                color="#888", linewidth=1.0, linestyle=BACKEND_LINESTYLE[bk],
                alpha=0.6, zorder=1)

    for _, bk, avg, jq, tc in cells:
        ax.scatter(jq, avg, c=[cmap(norm(tc))], marker=BACKEND_MARKER[bk],
                   s=100, edgecolor="black", linewidth=0.6, zorder=3)

    # Dominance arrow: small-reader-with-structured-memory dominates large-vanilla.
    p_dom = next((c for c in cells if c[0] == "0_6b" and c[1] == "memsearch"), None)
    p_van = next((c for c in cells if c[0] == "32b"  and c[1] == "vanilla"), None)
    if p_dom is not None and p_van is not None:
        ax.annotate(
            "", xy=(p_dom[3], p_dom[2]), xytext=(p_van[3], p_van[2]),
            arrowprops=dict(arrowstyle="->", color="#222", lw=1.3, alpha=0.85),
            zorder=2,
        )
        ax.text(p_dom[3] * 1.45, p_dom[2] + 5.5,
                "MemSearch $\\times$ Q3-0.6B",
                fontsize=7.5, color="#222", ha="left", va="bottom", zorder=4)
        ax.text(p_van[3] * 0.85, p_van[2] - 6,
                "Vanilla $\\times$ Q3-32B-AWQ",
                fontsize=7.5, color="#222", ha="right", va="top", zorder=4)

    ax.set_xscale("log")
    # Auto-fit x/y range to actual data with a small log-space / linear margin
    # so all rendered cells (and the dominance arrow endpoints) stay inside
    # the visible region. Avoids clipping markers that fall outside the
    # historical 1..10^2.5 J / 20..60 pp window.
    xs = [c[3] for c in cells if c[3] is not None and c[3] > 0]
    ys = [c[2] for c in cells if c[2] is not None]
    import math as _math
    if xs:
        lo = _math.log10(min(xs)); hi = _math.log10(max(xs))
        pad = max(0.15, (hi - lo) * 0.08)
        ax.set_xlim(10 ** (lo - pad), 10 ** (hi + pad))
    if ys:
        lo = min(ys); hi = max(ys)
        pad = max(2.0, (hi - lo) * 0.08)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Answer energy  (J / query, log)")
    ax.set_ylabel("Avg accuracy  (pp)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", which="both", alpha=0.12)
    ax.set_title("(a) Pareto: Avg vs.\\ energy on DGX Spark", fontsize=10)

    # Map CSV backend names back to paper-side BACKEND_TEX entries:
    #   inmem -> rag (BM25 RAG); the rest are identity.
    paper_label = {
        "vanilla":   "vanilla",
        "inmem":     "rag",
        "oracle":    "oracle",
        "memsearch": "memsearch",
        "memobase":  "memobase",
    }
    legend_elems = [
        Line2D([0], [0], marker=BACKEND_MARKER[bk], color="w",
               markerfacecolor="#bbb", markeredgecolor="black",
               markersize=9, label=BACKEND_TEX.get(paper_label.get(bk, bk), bk))
        for bk in PARETO_BACKENDS
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, frameon=False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        pad=0.02, fraction=0.045, shrink=0.85)
    cbar.set_label("Peak temp ($^\\circ$C)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Panel (b): Qwen3-8B anchor bar
# ---------------------------------------------------------------------------

ANCHOR = "8b"
BACKENDS_B = ["vanilla", "rag", "dense_rag", "memobase", "memos", "oracle"]
DENSE_RAG_8B = ("BGE-M3", 50.9)


def _plot_anchor(ax) -> None:
    bars = []
    for bk in BACKENDS_B:
        if bk == "dense_rag":
            mean, std = DENSE_RAG_8B[1], 0.0
        else:
            mean, std = PUBLISHED_AVG[ANCHOR][bk]
        bars.append((bk, mean, std))

    xs = list(range(len(BACKENDS_B)))
    for i, (bk, v, s) in enumerate(bars):
        hatch = "//" if bk == "dense_rag" else None
        ax.bar(i, v, 0.68, color=COLOR[bk], edgecolor="black", linewidth=0.5,
               yerr=s if s > 0 else None, capsize=2.5, ecolor="#333", hatch=hatch)
        ax.text(i, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8, color="#222")

    oracle_v = bars[-1][1]
    ax.axhline(oracle_v, color=COLOR["oracle"], lw=0.8, ls="--", alpha=0.55)

    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL_A_FULL[bk] for bk in BACKENDS_B],
                       rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Avg  (pp)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) Q3-8B anchor: backend $\\times$ Avg", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)


# ---------------------------------------------------------------------------
# Panel (b): lift over Vanilla per reader tier
# ---------------------------------------------------------------------------

LIFT_TIERS = ["0_6b", "llama3b", "7b", "8b", "32b"]
# Five non-vanilla bars per tier: oracle is the ceiling, the four memory
# backends are the candidates. Vanilla itself is the 0 baseline so it does
# not appear as its own bar.
LIFT_BACKENDS = ["rag", "oracle", "memsearch", "memobase", "memos"]


def _plot_lifts(ax) -> None:
    width = 0.16
    offsets = {bk: (i - (len(LIFT_BACKENDS) - 1) / 2) * width for i, bk in enumerate(LIFT_BACKENDS)}
    x_tiers = list(range(len(LIFT_TIERS)))

    for bk in LIFT_BACKENDS:
        lifts = []
        present = []
        for t in LIFT_TIERS:
            van = (PUBLISHED_AVG.get(t) or {}).get("vanilla")
            bv  = (PUBLISHED_AVG.get(t) or {}).get(bk)
            if van is None or bv is None:
                # In-flight cell (e.g. MemOS x {8b,32b} pending). Render
                # as a gap rather than crash; the missing bar is then
                # visually distinct from a zero-lift bar.
                lifts.append(0.0)
                present.append(False)
            else:
                lifts.append(bv[0] - van[0])
                present.append(True)
        xs = [x + offsets[bk] for x in x_tiers]
        plotted_xs = [x for x, ok in zip(xs, present) if ok]
        plotted_lifts = [v for v, ok in zip(lifts, present) if ok]
        ax.bar(plotted_xs, plotted_lifts, width,
               color=COLOR[bk], edgecolor="black", linewidth=0.5,
               label=BACKEND_TEX[bk])
        for x, v, ok in zip(xs, lifts, present):
            if not ok:
                continue
            ax.text(x, v + 1.0 if v >= 0 else v - 1.4,
                    f"{v:+.0f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=6, color="#222")

    ax.axhline(0, color="#555", lw=0.6)
    ax.set_xticks(x_tiers)
    ax.set_xticklabels([MODEL_TEX[t] for t in LIFT_TIERS],
                       rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Lift over Vanilla  (pp)")
    ax.set_ylim(-35, 60)
    ax.set_title("(b) Memory lift, by reader", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3,
              bbox_to_anchor=(0, 1.0))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _render(out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    _plot_pareto(axes[0], fig)
    _plot_lifts(axes[1])
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # New layout: out/latency_spark_<reader>_s2/hw/. If no reader has hw_agg,
    # skip. (The stale pre-release spark_results_s1/hw/ is not consulted.)
    new_hw_dirs = sorted(SPARK_LATENCY_BASE.glob("latency_spark_*_s2/hw"))
    has_data = any((d / f"hw_agg_vanilla_*_s2.json").parent.exists() and any(d.glob("hw_agg_*.json")) for d in new_hw_dirs)
    if not new_hw_dirs or not has_data:
        for p in (OUT_PDF, OUT_PDF.with_suffix(".png")):
            if p.exists():
                p.unlink()
        print(
            "[fig:finding3] skipped: no hw_agg data under "
            f"{SPARK_LATENCY_BASE}/latency_spark_*_s2/hw/. "
            "Run scripts/run_latency_spark.sh on Spark with the hw_probe sidecar."
        )
        return
    _render(OUT_PDF)
    print(f"[fig:finding3] wrote {OUT_PDF}")
    print(f"[fig:finding3] wrote {OUT_PDF.with_suffix('.png')}")


if __name__ == "__main__":
    main()
