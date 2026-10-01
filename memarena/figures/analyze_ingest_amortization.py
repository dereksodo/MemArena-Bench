#!/usr/bin/env python3
"""Generate the ingest-amortization break-even figure and K table.

Background
----------
Structured-memory backends (Memobase, MemOS) pay a one-time ingest cost: the
extractor LLM compresses the raw sessions of each ego into a structured cache
of facts/events. Every downstream answer query then retrieves from that cache
(``inmem``) at low per-query cost. Oracle routing pays zero ingest but runs the
longer per-query path on the full session window, so its per-query energy is
higher. For an on-device practitioner the natural question is

    "how many answer queries must I serve from this memory cache before the
    structured-memory path (ingest + K * inmem) matches the zero-ingest Oracle
    baseline (K * oracle)?"

The break-even number of queries K is given by

    ingest_J + K * inmem_J_per_query == K * oracle_J_per_query
    =>   K = ingest_J / (oracle_J_per_query - inmem_J_per_query)

When oracle is cheaper per query than inmem -- i.e. the denominator is non-positive
-- there is no break-even and the structured-memory backend is a *capability*
argument (coverage of D6 permission / D1 cloze) rather than an *efficiency*
argument. We flag this explicitly in the paper.

Outputs
-------
* ``paper/figures/fig_ingest_amortization.pdf`` (+ .png)
    Two-panel cumulative-energy plot: Oracle (linear through origin) vs
    Memobase (ingest-offset linear). Panel (a) = Qwen3-0.6B tier, (b) =
    Qwen3-32B AWQ tier. Separate panels because the two tiers differ by two
    orders of magnitude on the y-axis and would visually collapse on shared axes.
* stdout: a numeric K table for all structured-memory cells we have at
  MemArena-L scale (s1 trial seed, 8 ego x 15 day spark run).

Data sources (all from ``MASim/runs/l_20260408_111046/spark_results_s1``):

* Ingest energy (summed over the 15 day_*_ingest rows):
    ``hw/hw_agg_<model>_memobase_ingest.json``
    ``hw/hw_agg_qwen3_0.6b_memos_ingest.json``

* Answer per-query energy (per_cell.median_energy_j over 250 queries):
    ``hw/hw_agg_<model>_oracle.json``
    ``hw/hw_agg_<model>_inmem.json``

* Cache ingest wall-time (elapsed_seconds, n_ingest_batches):
    ``memory_cache/memcache_<backend>_<model>_spark8.summary.json``
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

try:
    from .paths import figure_path
except ImportError:  # pragma: no cover - direct script execution
    from paths import figure_path


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent


def _resolve_spark_root() -> Path:
    override = os.getenv("MEMARENA_SPARK_RUN_DIR") or os.getenv("MEMARENA_INGEST_ROOT")
    if override:
        return Path(override).expanduser()
    run_override = os.getenv("MEMARENA_RUN_DIR")
    if run_override:
        return Path(run_override).expanduser() / "spark_results_s1"
    return REPO_ROOT / "MASim" / "runs" / "l_20260408_111046" / "spark_results_s1"


SPARK_S1 = _resolve_spark_root()
OUT_PDF = figure_path("fig_ingest_amortization.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")


# ---------------------------------------------------------------------------
# Cell registry.

@dataclass
class Cell:
    backend: str           # "memobase" | "memos"
    model: str             # "qwen3_0.6b" | "qwen3_32b_awq"
    display_backend: str
    display_model: str
    ingest_hw: Path        # hw_agg_<model>_<backend>_ingest.json
    oracle_hw: Path        # hw_agg_<model>_oracle.json
    inmem_hw: Path         # hw_agg_<model>_inmem.json
    summary: Path          # memcache_<backend>_<model>_spark8.summary.json
    color: str             # figure color

    # Filled after read:
    ingest_energy_j: float = 0.0
    ingest_elapsed_s: float = 0.0
    oracle_median_j: float = 0.0
    inmem_median_j: float = 0.0
    summary_elapsed_s: float = 0.0
    n_ingest_batches: int = 0


def _mk_cell(backend: str, model: str, disp_m: str, color: str) -> Cell:
    return Cell(
        backend=backend,
        model=model,
        display_backend={"memobase": "Memobase", "memos": "MemOS"}[backend],
        display_model=disp_m,
        ingest_hw=SPARK_S1 / "hw" / f"hw_agg_{model}_{backend}_ingest.json",
        oracle_hw=SPARK_S1 / "hw" / f"hw_agg_{model}_oracle.json",
        inmem_hw=SPARK_S1 / "hw" / f"hw_agg_{model}_inmem.json",
        summary=SPARK_S1 / "memory_cache" / f"memcache_{backend}_{model}_spark8.summary.json",
        color=color,
    )


CELLS = [
    _mk_cell("memobase", "qwen3_0.6b",    "Qwen3-0.6B",      "#d62728"),
    _mk_cell("memos",    "qwen3_0.6b",    "Qwen3-0.6B",      "#ff7f0e"),
    _mk_cell("memobase", "qwen3_32b_awq", "Qwen3-32B (AWQ)", "#d62728"),
]


# ---------------------------------------------------------------------------
# IO helpers.

def _sum_ingest_energy(path: Path) -> tuple[float, float]:
    """Return (sum energy_j_net, sum elapsed_s) over the 15 ``day_N_ingest`` rows.

    The companion ``day_N_query`` rows are *probe queries during ingest* and
    must not be included in the ingest total.
    """
    if not path.exists():
        return float("nan"), float("nan")
    d = json.load(open(path))
    total_e = 0.0
    total_s = 0.0
    for row in d.get("per_query", []):
        inst = row.get("instance_id", "")
        if "ingest" not in inst:
            continue
        total_e += float(row.get("energy_j_net", 0.0) or 0.0)
        total_s += float(row.get("elapsed_s", 0.0) or 0.0)
    return total_e, total_s


def _median_energy(path: Path) -> float:
    """Return per_cell.median_energy_j for the 250-query answer run."""
    if not path.exists():
        return float("nan")
    d = json.load(open(path))
    return float(d.get("per_cell", {}).get("median_energy_j", float("nan")))


def _summary_numbers(path: Path) -> tuple[float, int]:
    if not path.exists():
        return float("nan"), 0
    d = json.load(open(path))
    return float(d.get("elapsed_seconds", float("nan"))), int(d.get("n_ingest_batches", 0))


def load_cell(c: Cell) -> None:
    c.ingest_energy_j, c.ingest_elapsed_s = _sum_ingest_energy(c.ingest_hw)
    c.oracle_median_j = _median_energy(c.oracle_hw)
    c.inmem_median_j = _median_energy(c.inmem_hw)
    c.summary_elapsed_s, c.n_ingest_batches = _summary_numbers(c.summary)


def breakeven_K(c: Cell) -> Optional[float]:
    """Energy-basis break-even K in number of downstream queries."""
    delta = c.oracle_median_j - c.inmem_median_j
    if delta <= 0:
        return None
    return c.ingest_energy_j / delta


# ---------------------------------------------------------------------------
# Figure.

def plot(cells: list[Cell]) -> None:
    """Two-panel figure: panel (a) = 0.6B tier, panel (b) = 32B tier.

    Each panel shows cumulative joules vs cumulative queries served. Oracle is a
    single straight line through origin (slope = oracle_median_j). Each
    structured-memory backend starts at its ingest energy and rises with slope
    inmem_median_j. The crossover with the Oracle line is K.
    """
    # Partition by display-model tier.
    tier_a = [c for c in cells if c.model == "qwen3_0.6b"]
    tier_b = [c for c in cells if c.model == "qwen3_32b_awq"]
    tiers = [
        ("(a) Qwen3-0.6B tier",  tier_a, 6000),
        ("(b) Qwen3-32B (AWQ) tier", tier_b, 3500),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    for ax, (title, tcells, x_max) in zip(axes, tiers):
        if not tcells:
            ax.set_visible(False)
            continue
        # Oracle line (same oracle for both backends at a tier).
        oracle_slope = tcells[0].oracle_median_j
        xs = [0, x_max]
        ys = [0, oracle_slope * x_max]
        ax.plot(xs, ys, color="#2ca02c", linewidth=2.0, linestyle="--",
                label=f"Oracle ({oracle_slope:.1f} J/query)")

        # Structured memory lines.
        for c in tcells:
            inmem_slope = c.inmem_median_j
            ingest_e = c.ingest_energy_j
            ys2 = [ingest_e, ingest_e + inmem_slope * x_max]
            ax.plot(xs, ys2, color=c.color, linewidth=2.0,
                    label=f"{c.display_backend} (ingest={ingest_e/1000:.1f} kJ, "
                          f"{inmem_slope:.1f} J/query)")

            K = breakeven_K(c)
            if K is not None and K <= x_max:
                y_k = ingest_e + inmem_slope * K
                ax.plot([K], [y_k], marker="o", color=c.color, markersize=6,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=5)
                ax.annotate(f"K={K:.0f}", xy=(K, y_k),
                            xytext=(6, -10), textcoords="offset points",
                            fontsize=8, color=c.color)
            elif K is not None:
                ax.annotate(
                    f"K={K:.0f} (off-axis)",
                    xy=(x_max, ingest_e + inmem_slope * x_max),
                    xytext=(-8, -18), textcoords="offset points",
                    fontsize=8, color=c.color, ha="right",
                )

        ax.set_xlim(0, x_max)
        ax.set_xlabel("Cumulative answer queries served from cache")
        ax.set_ylabel("Cumulative energy (J)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table.

def print_table(cells: list[Cell]) -> None:
    """Print the K table to stdout (energy-basis + wall-time-basis)."""
    header = (
        f"{'backend':10s}  {'model':18s}  "
        f"{'ingest_J':>12s}  {'oracle_J/q':>10s}  {'inmem_J/q':>10s}  "
        f"{'K_energy':>10s}  {'ingest_s':>10s}  "
        f"{'oracle_ms/q*':>12s}  {'inmem_ms/q*':>12s}  {'K_walltime':>12s}"
    )
    print(header)
    print("-" * len(header))

    # Walltime per-query from answer_results_run.json (median of answer_time_ms / 1000).
    for c in cells:
        oracle_s_q = _median_answer_time_s(c.oracle_hw.parent.parent / "oracle" / c.model / "answer_results_run.json")
        inmem_s_q  = _median_answer_time_s(c.oracle_hw.parent.parent / "inmem"  / c.model / "answer_results_run.json")
        K_e = breakeven_K(c)
        if oracle_s_q is not None and inmem_s_q is not None and oracle_s_q > inmem_s_q:
            K_w = c.ingest_elapsed_s / (oracle_s_q - inmem_s_q)
        else:
            K_w = None
        print(
            f"{c.display_backend:10s}  {c.display_model:18s}  "
            f"{c.ingest_energy_j:12.1f}  {c.oracle_median_j:10.2f}  {c.inmem_median_j:10.2f}  "
            f"{(f'{K_e:10.1f}' if K_e is not None else 'no b.e.'):>10s}  "
            f"{c.ingest_elapsed_s:10.1f}  "
            f"{(f'{oracle_s_q*1000:12.1f}' if oracle_s_q else 'n/a'):>12s}  "
            f"{(f'{inmem_s_q*1000:12.1f}' if inmem_s_q else 'n/a'):>12s}  "
            f"{(f'{K_w:12.1f}' if K_w else 'n/a'):>12s}"
        )
    print()
    print("K_energy = ingest_J / (oracle_median_J_per_query - inmem_median_J_per_query)")
    print("K_walltime = ingest_elapsed_s / (oracle_median_s - inmem_median_s)")


def _median_answer_time_s(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if not isinstance(d, list):
        return None
    times_ms = [r.get("answer_time_ms") for r in d
                if isinstance(r, dict) and r.get("answer_time_ms") is not None]
    if not times_ms:
        return None
    times_ms.sort()
    n = len(times_ms)
    mid = times_ms[n // 2] if n % 2 else 0.5 * (times_ms[n // 2 - 1] + times_ms[n // 2])
    return mid / 1000.0


# ---------------------------------------------------------------------------
# Main.

def main() -> None:
    if not SPARK_S1.exists():
        for path in (OUT_PDF, OUT_PNG):
            if path.exists():
                path.unlink()
        print(
            f"[fig:ingest-amortization] skipped: Spark timing root not found: {SPARK_S1}. "
            "Set MEMARENA_SPARK_RUN_DIR to reproduce this optional deployment figure."
        )
        return

    for c in CELLS:
        load_cell(c)
    valid = [
        c for c in CELLS
        if c.ingest_energy_j == c.ingest_energy_j
        and c.oracle_median_j == c.oracle_median_j
        and c.inmem_median_j == c.inmem_median_j
    ]
    if not valid:
        for path in (OUT_PDF, OUT_PNG):
            if path.exists():
                path.unlink()
        print(f"[fig:ingest-amortization] skipped: no complete ingest cells found under {SPARK_S1}")
        return
    print_table(valid)
    plot(valid)
    print(f"\nWrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
