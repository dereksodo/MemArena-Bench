#!/usr/bin/env python3
"""Generate Figure 4 (``fig:privacy-auc``): D6 Privacy-Utility trade-off.

Each point is one (model, backend) cell averaged over stochastic seeds. The
y-axis is *withhold accuracy* (fraction of queries with ``policy_expected ==
DENY_NO_ACCESS`` correctly refused); the x-axis is *false refusal* (fraction
of queries with ``policy_expected == ALLOW`` incorrectly refused, i.e.
``1 - disclose_rate``). The ideal operating point is the upper-left corner
(perfect withholding at zero utility cost); the dashed diagonal marks the
chance-rate trade-off (e.g. a constant ``refuse`` policy sits at (1, 1)).

Scoring is deterministic (not judge-based): it reads ``policy_compliant``
from each D6 instance (the simulator's pre-registered refusal-keyword check),
averaged across s2/s3/s4 seeds.

Output
------
``paper/figures/fig_privacy_utility_auc.pdf`` (and ``.png``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

from paper_data import policy_compliant_from_armB  # noqa: E402
from paper_data import (  # noqa: E402
    BACKEND_TEX,
    MODEL_ORDER, MODEL_TEX,
    SEEDS, DEFAULT_RUN,
    load_all_cells,
)
from paths import figure_path  # noqa: E402

MODEL_MARKERS = {"0_6b": "o", "llama3b": "s", "7b": "^", "8b": "D", "32b": "P"}
MODEL_SIZES   = {"0_6b": 60,  "llama3b": 70, "7b": 80, "8b": 90, "32b": 110}
BACKEND_COLORS = {
    "vanilla":  "#7f7f7f",
    "rag":      "#1f77b4",
    "oracle":   "#2ca02c",
    "memobase": "#d62728",
    "memos":    "#ff7f0e",
    "mem0":     "#9467bd",
}
PLOT_BACKENDS = ["vanilla", "rag", "oracle", "memobase", "memos"]

OUT_PDF = figure_path("fig_privacy_utility_auc.pdf")


def _cell_privacy_metrics(cell):
    """Return (withhold_acc, false_refusal) for a single cell.

    D6 instances only. ``policy_compliant`` is the deterministic keyword-based
    label. For DENY queries, ``policy_compliant`` True means "correctly
    refused". For ALLOW queries, True means "correctly disclosed", so the
    false-refusal rate is ``1 - mean(policy_compliant | ALLOW)``.
    """
    # cell holds aggregated sub-dim counts; the raw per-instance breakdown
    # lives in the source JSON. Re-read it to get policy-compliance per row.
    import json
    data = json.loads(cell.source_path.read_text())
    deny_c = deny_n = allow_c = allow_n = 0
    for r in data.get("details", []):
        qid = r.get("question_id") or ""
        if not qid.startswith("d4_"):
            continue
        exp = r.get("policy_expected")
        compliant = policy_compliant_from_armB(r)
        if exp == "DENY_NO_ACCESS":
            deny_n += 1
            deny_c += int(compliant)
        elif exp == "ALLOW":
            allow_n += 1
            allow_c += int(compliant)
    withhold = deny_c / deny_n if deny_n else None
    false_ref = (1.0 - (allow_c / allow_n)) if allow_n else None
    return withhold, false_ref


def _aggregate_seeds(grid, backend, model):
    """Return ``(withhold_mean, withhold_std, fr_mean, fr_std, n)`` over SEEDS."""
    ws, frs = [], []
    for seed in SEEDS:
        cell = grid.get((seed, backend, model))
        if cell is None:
            continue
        w, fr = _cell_privacy_metrics(cell)
        if w is None or fr is None:
            continue
        ws.append(w)
        frs.append(fr)
    if not ws:
        return None
    import statistics
    w_m = statistics.mean(ws)
    fr_m = statistics.mean(frs)
    w_s = statistics.stdev(ws) if len(ws) > 1 else 0.0
    fr_s = statistics.stdev(frs) if len(frs) > 1 else 0.0
    return (w_m, w_s, fr_m, fr_s, len(ws))


def main() -> None:
    grid = load_all_cells(DEFAULT_RUN)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))

    # Chance line: ``y = x`` is the "flip a coin" trade-off; anything below
    # this diagonal is strictly worse than random. Dashed, pale gray so the
    # real points stay visually dominant.
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1.0, ls="--", zorder=0)

    seen_b, seen_m = set(), set()
    n_plotted = 0
    for backend in PLOT_BACKENDS:
        for model in MODEL_ORDER:
            out = _aggregate_seeds(grid, backend, model)
            if out is None:
                continue
            w_m, w_s, fr_m, fr_s, n = out
            ax.errorbar(
                fr_m, w_m, xerr=fr_s, yerr=w_s,
                fmt="none", ecolor=BACKEND_COLORS[backend], alpha=0.55,
                elinewidth=0.8, capsize=2, zorder=1,
            )
            ax.scatter(
                fr_m, w_m,
                marker=MODEL_MARKERS[model], s=MODEL_SIZES[model],
                color=BACKEND_COLORS[backend],
                edgecolor="black", linewidth=0.6, zorder=2,
            )
            seen_b.add(backend)
            seen_m.add(model)
            n_plotted += 1

    from matplotlib.lines import Line2D
    backend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=BACKEND_COLORS[b], markersize=9,
               markeredgecolor="black", markeredgewidth=0.6,
               label=BACKEND_TEX[b])
        for b in PLOT_BACKENDS if b in seen_b
    ]
    model_handles = [
        Line2D([0], [0], marker=MODEL_MARKERS[m], color="w",
               markerfacecolor="#555555", markersize=9,
               markeredgecolor="black", markeredgewidth=0.6,
               label=MODEL_TEX[m])
        for m in MODEL_ORDER if m in seen_m
    ]
    leg1 = ax.legend(handles=backend_handles, title="Backend", loc="lower right",
                     fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Extractor", loc="upper right",
              fontsize=8, title_fontsize=9, frameon=False)

    ax.set_xlabel("False refusal rate on ALLOW queries (lower is better)")
    ax.set_ylabel("Withholding accuracy on DENY queries (higher is better)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title("D6 Privacy-Utility (mean ± std, s2/s3/s4)")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[fig:privacy-auc] plotted {n_plotted} cells → {OUT_PDF}")


if __name__ == "__main__":
    main()
