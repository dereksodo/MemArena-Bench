"""Generate panels (a), (b), and (c) of the rebuilt Figure 3 (fig:d6).

Panel (a): Content axis — Recall + Reasoning average accuracy, 4 backends x mean of 5 models.
Panel (b): Social axis — DENY-subset 3-category response distribution, pooled
           across readers, stacked-to-100%.
Panel (c): Privacy-utility scatter — one point per (backend, reader) cell;
           x = should-disclose disclose-rate on ALLOW, y = should-refuse
           abstain-rate on DENY (loose: REFUSAL + NONE).

Output:
  memarena/figures/figures/fig_d6_v2_bc.pdf  (layout is 1x3)
  memarena/figures/figures/fig_d6_v2_bc.png

Invocation:
  python scripts/reproduce_figures.py --name gen_fig_d6_v2
  python -m memarena.figures.gen_fig_d6_v2
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    from .paths import figure_path
except ImportError:  # pragma: no cover - direct script execution
    from paths import figure_path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent
RUN_ROOT = Path(os.getenv("MEMARENA_RUN_DIR", REPO_ROOT / "MASim/runs/l_20260408_111046"))
OUT_PDF = figure_path("fig_d6_v2_bc.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")

# Panel (d) data is computed via the paper-side analyzer.
sys.path.insert(0, str(PAPER_DIR))
from analyze_privacy_utility import (  # noqa: E402
    BACKENDS as PU_BACKENDS,
    BACKEND_LABEL,
    BACKEND_COLOR,
    collect_distribution,
    collect_distribution_per_trial,
)

BACKENDS = ["Vanilla", "RAG", "Oracle", "Memobase", "MemSearch"]
MODELS = ["Qwen3-0.6B", "Llama-3.2-3B", "Mistral-7B", "Qwen3-8B", "Qwen3-32B"]

# ---------------------------------------------------------------------------
# Panel (a) data — D1-D4 average accuracy, per (backend, model).
# Computed live from experiments_index.csv as (Rec + Rea) / 2 where
# Rec = category_accuracy(cell, CAT_RECALL [D1,D2]) and
# Rea = category_accuracy(cell, CAT_REASONING [D3,D4]). Each (backend, model)
# is averaged across the stochastic seeds that have a JSON in the index.
# ---------------------------------------------------------------------------
def _compute_content() -> dict[str, dict[str, float]]:
    from paper_data import (
        BACKEND_TEX, MODEL_TEX, SEEDS,
        CAT_RECALL, CAT_REASONING,
        category_accuracy, load_all_cells,
    )

    grid = load_all_cells()

    # Reverse maps from display label -> internal key.
    label_to_backend = {v: k for k, v in BACKEND_TEX.items()}
    label_to_model = {v: k for k, v in MODEL_TEX.items()}

    out: dict[str, dict[str, float]] = {b: {} for b in BACKENDS}
    for b_label in BACKENDS:
        backend = label_to_backend[b_label]
        for m_label in MODELS:
            model = label_to_model[m_label]
            scores: list[float] = []
            for seed in SEEDS:
                cell = grid.get((seed, backend, model))
                if cell is None:
                    continue
                rec = category_accuracy(cell, CAT_RECALL)
                rea = category_accuracy(cell, CAT_REASONING)
                if rec is None or rea is None:
                    continue
                scores.append((rec + rea) / 2.0)
            if scores:
                out[b_label][m_label] = 100.0 * (sum(scores) / len(scores))
    return out


CONTENT = _compute_content()

# ---------------------------------------------------------------------------
# Panel (b) data — DENY-subset 3-cat composition, pooled across readers and
# across the three stochastic trials s2/s3/s4 (n=3).
# Source: out/d6_armB_2026-05-03/per_backend_armB_{s2,s3,s4}.csv (rejudged on
# the new 5x5x3 raw responses via scripts/rejudge_d6_armB.py). Falls back to
# the legacy MASim/runs/.../d6_rebuild_2026-04-27/ path if the new dir does
# not exist yet.
# ---------------------------------------------------------------------------
import csv
import statistics

_NEW_D6_DIR = REPO_ROOT / "out" / "d6_armB_2026-05-03"
_LEGACY_D6_DIR = RUN_ROOT / "eval_results" / "d6_rebuild_2026-04-27"


def _load_social_pooled() -> dict[str, dict[str, dict[str, float]]]:
    """Read per_backend_armB_s{2,3,4}.csv, return mean and std per backend."""
    base = _NEW_D6_DIR if (_NEW_D6_DIR / "per_backend_armB_s2.csv").exists() else _LEGACY_D6_DIR
    csv_paths = [base / f"per_backend_armB_{t}.csv" for t in ("s2", "s3", "s4")]
    backend_csv_to_label = {
        "vanilla":   "Vanilla",
        "inmem":     "RAG",
        "oracle":    "Oracle",
        "memobase":  "Memobase",
        "memsearch": "MemSearch",
    }
    raw: dict[str, dict[str, list[float]]] = {}
    for path in csv_paths:
        if not path.exists():
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                label = backend_csv_to_label.get(row["backend"])
                if label is None:
                    continue
                raw.setdefault(label, {"REFUSAL": [], "NONE": [], "COMPLY": [], "ABSTAIN_LOOSE": []})
                ref = float(row["with_pct"])
                none = float(row["deny_none_pct"])
                com = float(row["leak_pct"])
                raw[label]["REFUSAL"].append(ref)
                raw[label]["NONE"].append(none)
                raw[label]["COMPLY"].append(com)
                raw[label]["ABSTAIN_LOOSE"].append(ref + none)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for label, cats in raw.items():
        out[label] = {}
        for cat, vals in cats.items():
            out[label][cat] = {
                "mean": statistics.fmean(vals) if vals else 0.0,
                "std":  statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
    return out


SOCIAL: dict[str, dict[str, dict[str, float]]] = _load_social_pooled()

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
COLOR_CONTENT = "#3e6fb5"
COLOR_REFUSAL = "#2b8a3e"   # green — privacy-grounded refusal (good)
COLOR_NONE    = "#ced4da"   # gray  — info-absent (neutral)
COLOR_COMPLY  = "#c92a2a"   # red   — disclose (bad)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
})


def plot_panel_b(ax):
    means, mins, maxs = [], [], []
    for b in BACKENDS:
        vals = list(CONTENT.get(b, {}).values())
        if not vals:
            means.append(0.0); mins.append(0.0); maxs.append(0.0)
            continue
        means.append(float(np.mean(vals)))
        mins.append(min(vals))
        maxs.append(max(vals))
    x = np.arange(len(BACKENDS))

    ax.bar(x, means, width=0.62, color=COLOR_CONTENT, edgecolor="black", linewidth=0.6)

    # Min-max whiskers (across 5 readers) — stays neutral on per-reader noise
    lower_err = np.array(means) - np.array(mins)
    upper_err = np.array(maxs) - np.array(means)
    ax.errorbar(
        x, means, yerr=[lower_err, upper_err], fmt="none",
        ecolor="black", elinewidth=1.0, capsize=5, capthick=1.0,
    )

    # Value labels on top of bar mean
    for i, m in enumerate(means):
        ax.text(i, m + 2.0, f"{m:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(BACKENDS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall + Reasoning average accuracy (%)")
    ax.set_title("(a) Content axis — memory helps recall")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def plot_panel_c(ax):
    x = np.arange(len(BACKENDS))
    refusal = []
    none_   = []
    comply  = []
    missing = []
    for b in BACKENDS:
        s = SOCIAL.get(b)
        if s is None or "REFUSAL" not in s:
            refusal.append(0.0); none_.append(0.0); comply.append(0.0)
            missing.append(True)
        else:
            refusal.append(s["REFUSAL"]["mean"])
            none_.append(s["NONE"]["mean"])
            comply.append(s["COMPLY"]["mean"])
            missing.append(False)

    refusal = np.array(refusal)
    none_   = np.array(none_)
    comply  = np.array(comply)

    ax.bar(x, refusal, width=0.62, color=COLOR_REFUSAL, edgecolor="black", linewidth=0.6, label="REFUSAL")
    ax.bar(x, none_,   width=0.62, bottom=refusal, color=COLOR_NONE, edgecolor="black", linewidth=0.6, label="NONE")
    ax.bar(x, comply,  width=0.62, bottom=refusal + none_, color=COLOR_COMPLY, edgecolor="black", linewidth=0.6, label="COMPLY")

    # In-bar annotations: show each segment percentage if >= 5%
    for i, b in enumerate(BACKENDS):
        if missing[i]:
            continue
        running = 0.0
        for seg_val, seg_color in [(refusal[i], "white"), (none_[i], "black"), (comply[i], "white")]:
            if seg_val >= 5.0:
                ax.text(i, running + seg_val / 2, f"{seg_val:.1f}%",
                        ha="center", va="center", fontsize=8, color=seg_color)
            running += seg_val

    # TBD placeholder for unfilled bars
    for i, b in enumerate(BACKENDS):
        if missing[i]:
            ax.text(i, 50, "TBD", ha="center", va="center", fontsize=11,
                    color="#6c757d", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="#6c757d", linewidth=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(BACKENDS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Response distribution on DENY queries (%)")
    ax.set_title("(b) Social axis — memory does not help withholding")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, ncol=3, fontsize=8.0,
              bbox_to_anchor=(0.5, -0.12), columnspacing=0.8,
              handlelength=1.2, handletextpad=0.35)


def plot_panel_d(ax):
    """Privacy-utility scatter, one point per (backend, reader) cell.

    Style mirrors fig:finding3(a): marker shape encodes backend; key
    (backend, reader) cells are annotated inline.
    """
    # Per-trial counts so the scatter point can sit at the trial-mean and
    # carry x/y error bars equal to the trial std across s2/s3/s4 (n=3).
    deny_trials  = collect_distribution_per_trial(
        lambda d: d["ground_truth"]["expected_answer_mode"] == "deny"
    )
    allow_trials = collect_distribution_per_trial(
        lambda d: d["ground_truth"]["expected_answer_mode"] == "disclose"
    )

    # Diagonal reference line + ideal-corner marker.
    ax.plot([0, 100], [100, 0], color="#cccccc", lw=0.7, ls=":", zorder=1)
    ax.scatter([97], [97], marker="*", s=130, facecolor="#fff7bc",
               edgecolor="#cc8800", lw=0.8, zorder=2)
    ax.annotate("ideal", xy=(97, 97), xytext=(-8, -2),
                textcoords="offset points",
                ha="right", va="center", fontsize=9, color="#cc8800")

    # Marker shape encodes reader; colour encodes backend.
    READER_MARKER = {
        "0_6b":    "o",
        "llama3b": "s",
        "7b":      "D",
        "8b":      "^",
        "32b":     "P",
    }
    READER_DISPLAY = {
        "0_6b":    "Q3-0.6B",
        "llama3b": "Llama-3.2-3B",
        "7b":      "Mistral-7B",
        "8b":      "Q3-8B",
        "32b":     "Q3-32B",
    }
    READER_LEGEND = {
        "0_6b":    "Q3-0.6B",
        "llama3b": "Llama-3B",
        "7b":      "Mistral-7B",
        "8b":      "Q3-8B",
        "32b":     "Q3-32B",
    }
    READER_ORDER = ["0_6b", "llama3b", "7b", "8b", "32b"]
    # Anchor cells that frame the privacy-utility story: Q3-32B (the
    # strongest reader) for each backend, so the reader can see that
    # even at scale no architecture lands near the ideal corner.
    ANCHOR_OFFSET = {
        ("vanilla",   "32b"): (8,  4),
        ("inmem",     "32b"): (8, -4),
        ("oracle",    "32b"): (10,  6),
        ("memobase",  "32b"): (-6, -10),
        ("memsearch", "32b"): (8,  -2),
    }
    ANCHOR_ALIGN = {
        ("vanilla",   "32b"): ("left",  "bottom"),
        ("inmem",     "32b"): ("left",  "top"),
        ("oracle",    "32b"): ("left",  "bottom"),
        ("memobase",  "32b"): ("right", "top"),
        ("memsearch", "32b"): ("left",  "center"),
    }

    # Plot one point per evaluated cell at the trial-mean position; error
    # bars on x and y are pstdev across the n=3 stochastic trials s2/s3/s4.
    points = []
    for (b, r), trials_d in deny_trials.items():
        if (b, r) not in allow_trials:
            continue
        trials_a = allow_trials[(b, r)]
        n = min(len(trials_d), len(trials_a))
        if n == 0:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for k in range(n):
            n_d = sum(trials_d[k].values())
            n_a = sum(trials_a[k].values())
            if n_d == 0 or n_a == 0:
                continue
            xs.append(trials_a[k]["COMPLY"] / n_a * 100)
            ys.append((trials_d[k]["REFUSAL"] + trials_d[k]["NONE"]) / n_d * 100)
        if not xs:
            continue
        ux = statistics.fmean(xs)
        ay = statistics.fmean(ys)
        x_std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        y_std = statistics.pstdev(ys) if len(ys) > 1 else 0.0
        ax.errorbar([ux], [ay], xerr=x_std, yerr=y_std, fmt="none",
                    ecolor=BACKEND_COLOR[b], elinewidth=0.8,
                    capsize=2.5, capthick=0.8, alpha=0.55, zorder=2)
        ax.scatter([ux], [ay], s=70, marker=READER_MARKER[r],
                   color=BACKEND_COLOR[b], edgecolor="black",
                   lw=0.5, zorder=3)
        points.append((b, r, ux, ay))

    # Inline annotations for the four Q3-32B anchor cells.
    for b, r, ux, ay in points:
        key = (b, r)
        if key not in ANCHOR_OFFSET:
            continue
        dx, dy = ANCHOR_OFFSET[key]
        ha, va = ANCHOR_ALIGN[key]
        ax.annotate(f"{BACKEND_LABEL[b]} $\\times$ {READER_DISPLAY[r]}",
                    xy=(ux, ay), xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha, va=va, fontsize=8, color="#222222")

    # Two legends: backend (colour) at lower-left, reader (shape) at upper-right.
    backend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=BACKEND_COLOR[b], markeredgecolor="black",
                   markeredgewidth=0.4, markersize=7, label=BACKEND_LABEL[b])
        for b in PU_BACKENDS
    ]
    leg_b = ax.legend(handles=backend_handles, loc="lower left",
                      bbox_to_anchor=(0.0, 0.32),
                      fontsize=7.5, frameon=False, handletextpad=0.4,
                      borderpad=0.2, labelspacing=0.3)
    ax.add_artist(leg_b)

    reader_handles = [
        plt.Line2D([0], [0], marker=READER_MARKER[r], color="w",
                   markerfacecolor="#888888", markeredgecolor="black",
                   markeredgewidth=0.4, markersize=6.5,
                   label=READER_LEGEND[r])
        for r in READER_ORDER
    ]
    ax.legend(handles=reader_handles, loc="lower left",
              bbox_to_anchor=(0.0, 0.0),
              fontsize=7, frameon=False, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.3)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("Utility on ALLOW: disclose-rate (%)")
    ax.set_ylabel("Privacy on DENY: abstain-rate, loose (%)")
    ax.set_title("(c) Privacy–utility scatter, per cell")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls=":", lw=0.5, color="#dddddd")
    ax.set_axisbelow(True)


def main():
    fig, (ax_b, ax_c, ax_d) = plt.subplots(1, 3, figsize=(15.0, 4.4))
    plot_panel_b(ax_b)
    plot_panel_c(ax_c)
    plot_panel_d(ax_d)
    plt.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
