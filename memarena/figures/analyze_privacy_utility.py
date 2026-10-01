#!/usr/bin/env python3
"""Generate the D6 privacy-utility scatter (``fig:privacy_utility``).

Single-panel scatter, four points (one per backend):

  x = utility on ALLOW    = COMPLY% on the ALLOW pool (correct disclose)
  y = abstention on DENY  = (REFUSAL + NONE)% on the DENY pool (loose)

The "loose" y-axis intentionally bundles policy-grounded REFUSAL with
silence-shaped NONE (recall failure / off-topic), because that is the
metric a reviewer who reads only an abstention column would compute. The
figure's claim is that no backend lands in the upper-right corner: every
architecture trades one axis for the other, so the privacy-utility gap
on \\DSixID{} is structural rather than a missing-prompt-engineering gap.

Source data
-----------
``MASim/runs/l_20260408_111046/eval_results/d6_rebuild_2026-04-27/
per_item_armB_s2/{backend}_{reader}.jsonl`` (19 cells).

Eval-instance ground truth from
``MASim/runs/l_20260408_111046/eval_instances/d4_permission.jsonl``
(``ground_truth.expected_answer_mode`` is ``deny`` / ``allow``).

Output
------
``paper/figures/fig_privacy_utility.pdf`` (and ``.png``).
The label ``fig:privacy_utility`` is preserved so existing references in
\\S7 continue to resolve; the figure content is what changed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from paths import figure_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(os.getenv("MEMARENA_RUN_DIR", REPO_ROOT / "MASim/runs/l_20260408_111046"))
TRIAL_NAMES = ["s2", "s3", "s4"]

# Prefer the new ArmB rejudge results (rebuilt on the 5x5x3 raw responses by
# scripts/rejudge_d6_armB.py); fall back to the legacy 2026-04-27 dir if the
# new outputs have not been produced yet.
_NEW_D6_BASE = REPO_ROOT / "out" / "d6_armB_2026-05-03"
_LEGACY_D6_BASE = RUN_ROOT / "eval_results" / "d6_rebuild_2026-04-27"
_D6_BASE = _NEW_D6_BASE if (_NEW_D6_BASE / "per_item_armB_s2").exists() else _LEGACY_D6_BASE
PER_ITEM_DIRS = [_D6_BASE / f"per_item_armB_{t}" for t in TRIAL_NAMES]
# d4_permission.jsonl ships with the MemArena-L Hugging Face dataset
# (dereksodo/memarena-l) and is materialised under data/benchmark/
# by scripts/download_dataset.py. Fall back to the legacy MASim/runs/ path.
_DATASET_EVAL_INSTANCES = REPO_ROOT / "data" / "benchmark" / "eval_instances" / "d4_permission.jsonl"
EVAL_INSTANCES = (
    _DATASET_EVAL_INSTANCES if _DATASET_EVAL_INSTANCES.exists()
    else RUN_ROOT / "eval_instances/d4_permission.jsonl"
)

OUT_PDF = figure_path("fig_privacy_utility.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")

BACKENDS = ["vanilla", "inmem", "oracle", "memobase", "memsearch"]
BACKEND_LABEL = {
    "vanilla":   "Vanilla",
    "inmem":     "RAG",
    "oracle":    "Oracle",
    "memobase":  "Memobase",
    "memsearch": "MemSearch",
}
BACKEND_COLOR = {
    "vanilla":   "#7f7f7f",
    "inmem":     "#1f77b4",
    "oracle":    "#2ca02c",
    "memobase":  "#d62728",
    "memsearch": "#ff7f0e",
}
READERS = ["0_6b", "llama3b", "7b", "8b", "32b"]
CAT_ORDER = ["REFUSAL", "COMPLY", "NONE"]


def load_eval_instances():
    inst = {}
    with EVAL_INSTANCES.open() as f:
        for line in f:
            d = json.loads(line)
            inst[d["instance_id"]] = d
    return inst


def collect_distribution(filter_fn):
    """Return {(backend, reader): {category: count}} per evaluated cell,
    pooled across the three stochastic trials s2/s3/s4 (n=3)."""
    inst = load_eval_instances()
    by_cell: dict[tuple[str, str], dict[str, int]] = {}
    for trial_dir in PER_ITEM_DIRS:
        if not trial_dir.exists():
            continue
        for backend in BACKENDS:
            for reader in READERS:
                p = trial_dir / f"{backend}_{reader}.jsonl"
                if not p.exists():
                    continue  # memobase_mistral missing across all trials
                key = (backend, reader)
                if key not in by_cell:
                    by_cell[key] = {c: 0 for c in CAT_ORDER}
                with p.open() as f:
                    for line in f:
                        v = json.loads(line)
                        iid = v["instance_id"]
                        if iid not in inst:
                            continue
                        if not filter_fn(inst[iid]):
                            continue
                        cat = v["category"]
                        if cat in by_cell[key]:
                            by_cell[key][cat] += 1
    return by_cell


def collect_distribution_per_trial(filter_fn):
    """Return {(backend, reader): [{category: count}, ...]} with one dict per
    trial (s2/s3/s4) where the per-item file exists. Cells with fewer trials
    yield a shorter list. Used by panel (c) to compute x/y trial-std bars."""
    inst = load_eval_instances()
    by_cell: dict[tuple[str, str], list[dict[str, int]]] = {}
    for trial_dir in PER_ITEM_DIRS:
        if not trial_dir.exists():
            continue
        for backend in BACKENDS:
            for reader in READERS:
                p = trial_dir / f"{backend}_{reader}.jsonl"
                if not p.exists():
                    continue
                trial_counts = {c: 0 for c in CAT_ORDER}
                with p.open() as f:
                    for line in f:
                        v = json.loads(line)
                        iid = v["instance_id"]
                        if iid not in inst:
                            continue
                        if not filter_fn(inst[iid]):
                            continue
                        cat = v["category"]
                        if cat in trial_counts:
                            trial_counts[cat] += 1
                if sum(trial_counts.values()) > 0:
                    by_cell.setdefault((backend, reader), []).append(trial_counts)
    return by_cell


def main():
    dist_deny = collect_distribution(
        lambda d: d["ground_truth"]["expected_answer_mode"] == "deny"
    )
    dist_allow = collect_distribution(
        lambda d: d["ground_truth"]["expected_answer_mode"] == "disclose"
    )

    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    # Diagonal reference line and ideal-corner marker.
    ax.plot([0, 100], [100, 0], color="#cccccc", lw=0.7, ls=":", zorder=1)
    ax.scatter([97], [97], marker="*", s=140, facecolor="#fff7bc",
               edgecolor="#cc8800", lw=0.8, zorder=2)
    ax.annotate("ideal", xy=(97, 97), xytext=(-8, -2),
                textcoords="offset points",
                ha="right", va="center", fontsize=9, color="#cc8800")

    # Plot one point per (backend, reader) evaluated cell.
    points = []
    for (b, r), d_d in dist_deny.items():
        if (b, r) not in dist_allow:
            continue
        d_a = dist_allow[(b, r)]
        n_d = sum(d_d.values())
        n_a = sum(d_a.values())
        if n_d == 0 or n_a == 0:
            continue
        utility = d_a["COMPLY"] / n_a * 100  # x: should-disclose disclose-rate
        absten  = (d_d["REFUSAL"] + d_d["NONE"]) / n_d * 100  # y: should-refuse abstain-rate (loose)
        points.append((b, r, utility, absten, n_a, n_d))

    # Per-backend cluster centroid for the inline label.
    cluster: dict[str, list[tuple[float, float]]] = {b: [] for b in BACKENDS}
    for b, _, ux, ay, _, _ in points:
        cluster[b].append((ux, ay))

    # Plot scatter per cell, semi-transparent + black edge for legibility.
    for b, _, ux, ay, _, _ in points:
        ax.scatter([ux], [ay], s=70, color=BACKEND_COLOR[b],
                   edgecolor="black", lw=0.4, alpha=0.85, zorder=3)

    # Backend label placement, offset from cluster centroid to avoid overlap.
    label_offset = {
        "vanilla":  (10, 6),
        "inmem":    (10, 0),
        "oracle":   (10, 6),
        "memobase": (-10, -6),
    }
    label_align = {
        "vanilla":  ("left", "bottom"),
        "inmem":    ("left", "center"),
        "oracle":   ("left", "bottom"),
        "memobase": ("right", "top"),
    }
    for b in BACKENDS:
        if not cluster[b]:
            continue
        cx = sum(p[0] for p in cluster[b]) / len(cluster[b])
        cy = sum(p[1] for p in cluster[b]) / len(cluster[b])
        dx, dy = label_offset[b]
        ha, va = label_align[b]
        ax.annotate(BACKEND_LABEL[b],
                    xy=(cx, cy), xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha, va=va, fontsize=10, fontweight="bold",
                    color=BACKEND_COLOR[b])

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("Utility on ALLOW: should-disclose disclose-rate (%)")
    ax.set_ylabel("Privacy on DENY: should-refuse abstain-rate, loose (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls=":", color="#dddddd", lw=0.5)

    plt.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Wrote {OUT_PDF.relative_to(Path.cwd())}")
    print()
    print("=== Cell coordinates (utility vs loose abstention) ===")
    for b, r, ux, ay, n_a, n_d in points:
        print(f"  {BACKEND_LABEL[b]:>9} x {r:<8}: x={ux:5.1f}%  y={ay:5.1f}%  "
              f"(ALLOW n={n_a}, DENY n={n_d})")


if __name__ == "__main__":
    main()
