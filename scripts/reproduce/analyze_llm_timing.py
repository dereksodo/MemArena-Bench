#!/usr/bin/env python3
"""Analyze LLM call timing from a pipeline run, grouped by phase tag.

Usage:
    python scripts/analyze_llm_timing.py MASim/runs/small_20260306_222800
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_timing(run_dir: Path) -> list:
    timing_path = run_dir / "llm_timing.jsonl"
    if not timing_path.exists():
        print(f"Error: {timing_path} not found")
        sys.exit(1)
    records = []
    with timing_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_llm_timing.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    records = load_timing(run_dir)
    print(f"Loaded {len(records)} LLM call records from {run_dir}")

    # Check for untagged calls
    untagged = [r for r in records if "phase" not in r]
    if untagged:
        print(f"\nWARNING: {len(untagged)} / {len(records)} calls have no phase tag!")

    # Group by phase tag directly (no heuristics)
    by_phase: dict = defaultdict(list)
    for rec in records:
        phase = rec.get("phase", "<untagged>")
        by_phase[phase].append(rec)

    # ---- Print summary table ----
    print(f"\n{'Phase':<28} {'Count':>6} {'Total(s)':>10} {'Mean(s)':>9} "
          f"{'Median(s)':>9} {'P95(s)':>9} {'Max(s)':>9}")
    print("-" * 90)

    phase_stats = {}
    for phase in sorted(by_phase.keys(), key=lambda p: -len(by_phase[p])):
        walls = np.array([r["wall_s"] for r in by_phase[phase]])
        stats = {
            "count": len(walls),
            "total": walls.sum(),
            "mean": walls.mean(),
            "median": float(np.median(walls)),
            "p95": float(np.percentile(walls, 95)),
            "max": float(walls.max()),
            "walls": walls,
        }
        phase_stats[phase] = stats
        print(f"{phase:<28} {stats['count']:>6} {stats['total']:>10.2f} "
              f"{stats['mean']:>9.3f} {stats['median']:>9.3f} "
              f"{stats['p95']:>9.3f} {stats['max']:>9.3f}")

    # ---- Token stats ----
    print(f"\n{'Phase':<28} {'Avg Prompt':>12} {'Avg Completion':>15} "
          f"{'Avg Total Tok':>14} {'Tok/s':>8}")
    print("-" * 83)
    for phase in sorted(by_phase.keys(), key=lambda p: -len(by_phase[p])):
        recs = by_phase[phase]
        prompt_avg = np.mean([r.get("prompt_tokens", 0) for r in recs])
        comp_avg = np.mean([r.get("completion_tokens", 0) for r in recs])
        total_comp = sum(r.get("completion_tokens", 0) for r in recs)
        total_wall = sum(r["wall_s"] for r in recs)
        tps = total_comp / total_wall if total_wall > 0 else 0
        print(f"{phase:<28} {prompt_avg:>12.0f} {comp_avg:>15.0f} "
              f"{prompt_avg + comp_avg:>14.0f} {tps:>8.1f}")

    # ---- PP step wall-clock span analysis ----
    pp_calls = [r for r in records if r.get("phase") == "pp_turn" and "step" in r]
    step_spans = {}
    if pp_calls:
        by_step = defaultdict(list)
        for r in pp_calls:
            by_step[r["step"]].append(r)

        print(f"\n{'PP Step':<10} {'N Calls':>8} {'Span(s)':>10} "
              f"{'Mean Call':>10} {'Max Call':>10} {'Straggler%':>11}")
        print("-" * 65)
        for step in sorted(by_step.keys()):
            calls = by_step[step]
            ts_arr = np.array([c["ts"] for c in calls])
            wall_arr = np.array([c["wall_s"] for c in calls])
            span = float((ts_arr + wall_arr).max() - ts_arr.min())
            mean_call = float(wall_arr.mean())
            max_call = float(wall_arr.max())
            straggler_pct = ((span - mean_call) / mean_call * 100) if mean_call > 0 else 0
            step_spans[step] = span
            print(f"step {step:<5} {len(calls):>8} {span:>10.3f} "
                  f"{mean_call:>10.3f} {max_call:>10.3f} {straggler_pct:>10.1f}%")

    # ====================================================================
    # PLOTS — multi-panel figure
    # ====================================================================
    sorted_phases = sorted(phase_stats.keys(), key=lambda p: -phase_stats[p]["count"])
    n_phases = len(sorted_phases)

    # Panel layout:
    #   Row 0..N: per-phase histograms (3 columns)
    #   Row N+1: PP step timing chart (full width) — if applicable
    #   Row N+2: summary bar chart (full width)
    hist_ncols = min(3, n_phases)
    hist_nrows = (n_phases + hist_ncols - 1) // hist_ncols

    extra_rows = 1  # summary bar chart always
    if step_spans:
        extra_rows += 1  # PP step timing

    total_rows = hist_nrows + extra_rows
    fig_height = 4.5 * hist_nrows + 5 * extra_rows

    fig = plt.figure(figsize=(6 * hist_ncols, fig_height))

    # Use GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(total_rows, hist_ncols, figure=fig, hspace=0.4, wspace=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # ---- Per-phase histograms ----
    for idx, phase in enumerate(sorted_phases):
        row, col = divmod(idx, hist_ncols)
        ax = fig.add_subplot(gs[row, col])
        walls = phase_stats[phase]["walls"]
        stats = phase_stats[phase]

        n_bins = min(40, max(10, len(walls) // 3))
        ax.hist(walls, bins=n_bins, color=colors[idx % 10], alpha=0.75,
                edgecolor="white", linewidth=0.5)
        ax.axvline(stats["mean"], color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={stats['mean']:.3f}s")
        ax.axvline(stats["median"], color="blue", linestyle="-.", linewidth=1.2,
                   label=f"median={stats['median']:.3f}s")
        ax.axvline(stats["p95"], color="orange", linestyle=":", linewidth=1.2,
                   label=f"p95={stats['p95']:.3f}s")
        ax.set_title(f"{phase}  (n={stats['count']})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused histogram cells
    for idx in range(n_phases, hist_nrows * hist_ncols):
        row, col = divmod(idx, hist_ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_visible(False)

    current_row = hist_nrows

    # ---- PP step timing chart ----
    if step_spans:
        ax_step = fig.add_subplot(gs[current_row, :])
        steps_sorted = sorted(step_spans.keys())
        spans = [step_spans[s] for s in steps_sorted]
        ax_step.bar(range(len(steps_sorted)), spans, color="#4C72B0", alpha=0.85,
                    edgecolor="white", linewidth=0.5)
        ax_step.set_xticks(range(len(steps_sorted)))
        ax_step.set_xticklabels([str(s) for s in steps_sorted], fontsize=8)
        ax_step.set_xlabel("PP Turn Step")
        ax_step.set_ylabel("Wall-clock Span (s)")
        ax_step.set_title("PP Step Wall-Clock Span (first call start → last call end)",
                          fontsize=11, fontweight="bold")
        ax_step.grid(axis="y", alpha=0.3)
        current_row += 1

    # ---- Summary bar chart: total wall-clock per phase ----
    ax_bar = fig.add_subplot(gs[current_row, :])
    totals = [(p, phase_stats[p]["total"]) for p in sorted_phases]
    totals.sort(key=lambda x: -x[1])
    labels = [t[0] for t in totals]
    values = [t[1] for t in totals]
    bar_colors = [colors[i % 10] for i in range(len(labels))]

    bars = ax_bar.barh(range(len(labels)), values, color=bar_colors, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
    ax_bar.set_yticks(range(len(labels)))
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Total Wall-Clock Seconds")
    ax_bar.set_title("Total Wall-Clock by Phase", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.3)

    # Annotate bars with values
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}s", va="center", fontsize=8)

    fig.suptitle(
        f"LLM Timing Analysis — {run_dir.name}\n"
        f"{len(records)} calls, {n_phases} phases"
        + (f", {len(untagged)} untagged" if untagged else ""),
        fontsize=14, fontweight="bold", y=1.005,
    )

    out_path = run_dir / "llm_timing_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
