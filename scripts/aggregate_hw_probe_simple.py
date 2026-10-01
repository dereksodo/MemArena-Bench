#!/usr/bin/env python3
"""Per-cell-only aggregation of hw_probe CSV.

The legacy hw_aggregate.py expects per-query markers (begin/end timestamps
for each request). Producing those markers requires modifying the answering
loop. For the Pareto / deployment figures we only need *per_cell* numbers,
so this simpler aggregator skips markers and emits one ``per_cell`` block
covering the entire CSV.

Usage:
    python aggregate_hw_probe_simple.py \
        --csv hw_probe_<reader>_<backend>_<trial>.csv \
        --answer-results answer_results_<...>.json \
        --outfile hw_agg_<reader>_<backend>_<trial>.json

The answer-results JSON tells us how many queries the cell answered, so we
can divide total energy by n_queries to get a Pareto-quality per-query
estimate. The per-query field is left empty -- callers that need precise
per-query splits must rerun the cell with the marker-emitting answerer.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean as _mean


def load_samples(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v
            out.append(parsed)
    return out


def _agg(samples: list[dict], col: str, default: float = 0.0) -> tuple[float, float]:
    """Return ``(mean, max)`` of column ``col`` across samples."""
    vals = [float(s.get(col, default) or default) for s in samples]
    if not vals:
        return default, default
    return (sum(vals) / len(vals), max(vals))


def per_cell_stats(samples: list[dict], n_queries: int) -> dict:
    base = {
        "n_samples": 0,
        "elapsed_s": 0.0,
        # GPU
        "mean_gpu_power_w": 0.0, "peak_gpu_power_w": 0.0,
        "mean_util_pct": 0.0, "peak_temp_c": 0.0,
        # CPU
        "mean_cpu_pct": 0.0, "peak_cpu_pct": 0.0,
        "mean_cpu_freq_mhz": 0.0, "peak_cpu_temp_c": 0.0,
        # Memory
        "peak_sys_mem_gib": 0.0,
        "peak_sglang_rss_gib": 0.0, "peak_memobase_rss_gib": 0.0,
        "peak_memos_rss_gib": 0.0,
        # SoC rail buckets
        "mean_rails_total_w": 0.0, "peak_rails_total_w": 0.0,
        "mean_rails_cpu_w": 0.0, "mean_rails_gpu_w": 0.0,
        "mean_rails_dram_w": 0.0,
        # Energy split (uses rails when available, else GPU NVML).
        "total_energy_j": 0.0,
        "total_gpu_energy_j": 0.0,
        "total_cpu_energy_j": 0.0,
        "total_dram_energy_j": 0.0,
        "mean_energy_j_per_query": 0.0,
        "mean_gpu_energy_j_per_query": 0.0,
        "mean_cpu_energy_j_per_query": 0.0,
        "mean_dram_energy_j_per_query": 0.0,
        # Back-compat aliases for old aggregator consumers.
        "mean_power_w": 0.0, "peak_power_w": 0.0,
        "n_queries": int(n_queries),
    }
    if not samples:
        return base

    ts = [float(s["ts_epoch_ms"]) for s in samples]
    elapsed_s = (ts[-1] - ts[0]) / 1000.0 if len(ts) > 1 else 0.0

    gpu_p_mean, gpu_p_peak = _agg(samples, "gpu_power_w")
    if gpu_p_mean == 0.0:
        # Probe wrote only the legacy `power_w` alias; use that.
        gpu_p_mean, gpu_p_peak = _agg(samples, "power_w")
    util_mean, _ = _agg(samples, "gpu_util_pct")
    gpu_t_peak = max((float(s.get("gpu_temp_c", s.get("temp_c", 0.0)) or 0.0) for s in samples), default=0.0)

    cpu_pct_mean, cpu_pct_peak = _agg(samples, "cpu_pct")
    cpu_freq_mean, _ = _agg(samples, "cpu_freq_mhz")
    cpu_t_peak = max((float(s.get("cpu_temp_c", 0.0) or 0.0) for s in samples), default=0.0)

    mem_peak = max((float(s.get("sys_mem_used_gib", 0.0) or 0.0) for s in samples), default=0.0)
    sglang_peak = max((float(s.get("sglang_rss_gib", 0.0) or 0.0) for s in samples), default=0.0)
    memobase_peak = max((float(s.get("memobase_rss_gib", 0.0) or 0.0) for s in samples), default=0.0)
    memos_peak = max((float(s.get("memos_rss_gib", 0.0) or 0.0) for s in samples), default=0.0)

    rails_total_mean, rails_total_peak = _agg(samples, "rails_total_power_w")
    rails_cpu_mean, _ = _agg(samples, "rails_cpu_power_w")
    rails_gpu_mean, _ = _agg(samples, "rails_gpu_power_w")
    rails_dram_mean, _ = _agg(samples, "rails_dram_power_w")

    # Energy: rails (if any) take precedence — they cover the whole SoC.
    # When rails are zero (no hwmon support), we fall back to GPU NVML.
    total_energy_j = (rails_total_mean if rails_total_mean else gpu_p_mean) * elapsed_s
    gpu_energy_j  = (rails_gpu_mean   if rails_gpu_mean   else gpu_p_mean) * elapsed_s
    cpu_energy_j  = rails_cpu_mean  * elapsed_s
    dram_energy_j = rails_dram_mean * elapsed_s

    per_q = total_energy_j / n_queries if n_queries > 0 else 0.0
    per_q_gpu = gpu_energy_j / n_queries if n_queries > 0 else 0.0
    per_q_cpu = cpu_energy_j / n_queries if n_queries > 0 else 0.0
    per_q_dram = dram_energy_j / n_queries if n_queries > 0 else 0.0

    base.update({
        "n_samples": len(samples),
        "elapsed_s": round(elapsed_s, 3),
        "mean_gpu_power_w": round(gpu_p_mean, 3),
        "peak_gpu_power_w": round(gpu_p_peak, 3),
        "mean_util_pct": round(util_mean, 2),
        "peak_temp_c": round(gpu_t_peak, 1),
        "mean_cpu_pct": round(cpu_pct_mean, 2),
        "peak_cpu_pct": round(cpu_pct_peak, 2),
        "mean_cpu_freq_mhz": round(cpu_freq_mean, 1),
        "peak_cpu_temp_c": round(cpu_t_peak, 1),
        "peak_sys_mem_gib": round(mem_peak, 3),
        "peak_sglang_rss_gib": round(sglang_peak, 3),
        "peak_memobase_rss_gib": round(memobase_peak, 3),
        "peak_memos_rss_gib": round(memos_peak, 3),
        "mean_rails_total_w": round(rails_total_mean, 3),
        "peak_rails_total_w": round(rails_total_peak, 3),
        "mean_rails_cpu_w": round(rails_cpu_mean, 3),
        "mean_rails_gpu_w": round(rails_gpu_mean, 3),
        "mean_rails_dram_w": round(rails_dram_mean, 3),
        "total_energy_j": round(total_energy_j, 2),
        "total_gpu_energy_j": round(gpu_energy_j, 2),
        "total_cpu_energy_j": round(cpu_energy_j, 2),
        "total_dram_energy_j": round(dram_energy_j, 2),
        "mean_energy_j_per_query": round(per_q, 4),
        "mean_gpu_energy_j_per_query": round(per_q_gpu, 4),
        "mean_cpu_energy_j_per_query": round(per_q_cpu, 4),
        "mean_dram_energy_j_per_query": round(per_q_dram, 4),
        # Back-compat aliases (legacy aggregator consumers expect these names).
        "mean_power_w": round(gpu_p_mean, 3),
        "peak_power_w": round(gpu_p_peak, 3),
    })
    return base


def count_queries(answer_results_path: Path) -> int:
    if not answer_results_path.exists():
        return 0
    payload = json.loads(answer_results_path.read_text())
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        details = payload.get("details") or payload.get("rows") or []
        return len(details)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path, help="hw_probe sidecar CSV")
    ap.add_argument("--answer-results", type=Path, default=None,
                    help="answer_results_*.json for the same cell (used to count n_queries)")
    ap.add_argument("--n-queries", type=int, default=0,
                    help="explicit n_queries (overrides --answer-results)")
    ap.add_argument("--outfile", required=True, type=Path,
                    help="Output JSON: {per_cell: {...}, source_csv: \"...\", per_query: []}")
    args = ap.parse_args()

    if args.n_queries > 0:
        n_q = args.n_queries
    elif args.answer_results is not None:
        n_q = count_queries(args.answer_results)
    else:
        n_q = 0

    samples = load_samples(args.csv)
    cell = per_cell_stats(samples, n_q)

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile.write_text(json.dumps({
        "source_csv": str(args.csv),
        "n_queries": n_q,
        "per_cell": cell,
        "per_query": [],
    }, indent=2))
    print(f"[hw_agg_simple] wrote {args.outfile} (n_samples={cell['n_samples']}, "
          f"elapsed_s={cell['elapsed_s']}, J/query={cell['mean_energy_j_per_query']})")


if __name__ == "__main__":
    main()
