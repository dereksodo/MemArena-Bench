#!/usr/bin/env python3
"""Aggregate hw_probe samples + markers into per-query metrics.

Usage:
    python scripts/hw_aggregate.py \
        --csv      spark_results_s1/hw/hw_probe_qwen3_8b_vanilla_answerA.csv \
        --markers  spark_results_s1/vanilla/hw_markers_vanilla_qwen3_0_6b_spark8.jsonl \
        --idle-baseline spark_results_s1/hw/hw_idle_baseline_qwen3_0_6b.json \
        --out      spark_results_s1/hw/hw_agg_qwen3_8b_vanilla.json

For each marker (instance_id, call_idx) with ts_start_ms..ts_end_ms:
    samples = rows in CSV with ts_start_ms <= ts < ts_end_ms
    if len(samples) < 2: warn, fall back to point samples at ts_start and ts_end
    mean_power_w        = mean(samples.power_w)
    peak_power_w        = max(samples.power_w)
    peak_sys_mem_gib    = max(samples.sys_mem_used_gib)
    peak_sglang_rss_gib = max(samples.sglang_rss_gib)
    peak_temp_c         = max(samples.temp_c)
    mean_util_pct       = mean(samples.gpu_util_pct)
    elapsed_s           = (ts_end_ms - ts_start_ms) / 1000
    energy_j_gross      = mean_power_w * elapsed_s
    energy_j_net        = (mean_power_w - idle_baseline_w) * elapsed_s

Output schema:
    { "per_query": [ {instance_id, call_idx, elapsed_s, energy_j_net, energy_j_gross,
                       peak_power_w, peak_sys_mem_gib, peak_sglang_rss_gib,
                       peak_temp_c, mean_util_pct, n_samples, ... } ],
      "per_cell":  { mean_energy_j_per_query, median_energy_j, p95_energy_j,
                      peak_sys_mem_gib, peak_sglang_rss_gib,
                      peak_power_w, peak_temp_c, mean_util_pct } }
"""

import argparse
import csv
import json
import os
import statistics
import sys
from typing import Any, Dict, List


def load_csv(path: str) -> List[Dict[str, float]]:
    """Load hw_probe CSV into list of dicts with float values."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = 0.0
            rows.append(parsed)
    return rows


def load_markers(path: str) -> List[Dict[str, Any]]:
    """Load JSONL marker file."""
    markers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                markers.append(json.loads(line))
    return markers


def load_idle_baseline(path: str) -> float:
    """Load idle baseline mean power (W)."""
    with open(path) as f:
        data = json.load(f)
    return float(data.get("mean_power_w", 0.0))


def aggregate_marker(samples: List[Dict[str, float]], marker: Dict, idle_w: float) -> Dict[str, Any]:
    """Compute per-query metrics for one marker."""
    ts_start = marker["ts_start_ms"]
    ts_end = marker["ts_end_ms"]
    elapsed_s = (ts_end - ts_start) / 1000.0

    # Filter samples within the marker window
    window = [s for s in samples if ts_start <= s.get("ts_epoch_ms", 0) < ts_end]

    if len(window) < 2:
        # Warn and use boundary point samples
        print(f"[hw_aggregate] WARN: only {len(window)} samples for "
              f"{marker.get('instance_id', '?')} call_idx={marker.get('call_idx', 0)} "
              f"({elapsed_s:.3f}s window)", file=sys.stderr)

    if not window:
        return {
            "instance_id": marker.get("instance_id", ""),
            "call_idx": marker.get("call_idx", 0),
            "elapsed_s": round(elapsed_s, 4),
            "energy_j_net": 0.0,
            "energy_j_gross": 0.0,
            "mean_power_w": 0.0,
            "peak_power_w": 0.0,
            "peak_sys_mem_gib": 0.0,
            "peak_sglang_rss_gib": 0.0,
            "peak_temp_c": 0.0,
            "mean_util_pct": 0.0,
            "n_samples": 0,
        }

    powers = [s.get("power_w", 0.0) for s in window]
    mean_power = statistics.mean(powers)
    peak_power = max(powers)
    peak_sys_mem = max(s.get("sys_mem_used_gib", 0.0) for s in window)
    peak_sglang_rss = max(s.get("sglang_rss_gib", 0.0) for s in window)
    peak_temp = max(s.get("temp_c", 0.0) for s in window)
    utils = [s.get("gpu_util_pct", 0.0) for s in window]
    mean_util = statistics.mean(utils) if utils else 0.0

    energy_gross = mean_power * elapsed_s
    energy_net = max(0.0, (mean_power - idle_w) * elapsed_s)

    return {
        "instance_id": marker.get("instance_id", ""),
        "call_idx": marker.get("call_idx", 0),
        "elapsed_s": round(elapsed_s, 4),
        "energy_j_net": round(energy_net, 4),
        "energy_j_gross": round(energy_gross, 4),
        "mean_power_w": round(mean_power, 2),
        "peak_power_w": round(peak_power, 2),
        "peak_sys_mem_gib": round(peak_sys_mem, 3),
        "peak_sglang_rss_gib": round(peak_sglang_rss, 3),
        "peak_temp_c": round(peak_temp, 1),
        "mean_util_pct": round(mean_util, 1),
        "n_samples": len(window),
    }


def compute_cell_summary(per_query: List[Dict]) -> Dict[str, Any]:
    """Compute per-cell aggregate metrics."""
    if not per_query:
        return {}

    energies_net = [q["energy_j_net"] for q in per_query]
    energies_gross = [q["energy_j_gross"] for q in per_query]

    return {
        "n_queries": len(per_query),
        "mean_energy_j_per_query": round(statistics.mean(energies_net), 4),
        "median_energy_j": round(statistics.median(energies_net), 4),
        "p95_energy_j": round(sorted(energies_net)[int(len(energies_net) * 0.95)], 4) if len(energies_net) >= 20 else None,
        "mean_energy_j_gross": round(statistics.mean(energies_gross), 4),
        "peak_sys_mem_gib": round(max(q["peak_sys_mem_gib"] for q in per_query), 3),
        "peak_sglang_rss_gib": round(max(q["peak_sglang_rss_gib"] for q in per_query), 3),
        "peak_power_w": round(max(q["peak_power_w"] for q in per_query), 2),
        "peak_temp_c": round(max(q["peak_temp_c"] for q in per_query), 1),
        "mean_util_pct": round(statistics.mean(q["mean_util_pct"] for q in per_query), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="hw_probe CSV file")
    parser.add_argument("--markers", required=True, help="hw_markers JSONL file")
    parser.add_argument("--idle-baseline", required=True, help="hw_idle_baseline JSON file")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    samples = load_csv(args.csv)
    markers = load_markers(args.markers)
    idle_w = load_idle_baseline(args.idle_baseline)

    print(f"[hw_aggregate] {len(samples)} CSV samples, {len(markers)} markers, idle={idle_w:.1f}W",
          file=sys.stderr)

    per_query = [aggregate_marker(samples, m, idle_w) for m in markers]
    cell_summary = compute_cell_summary(per_query)

    result = {
        "idle_baseline_w": idle_w,
        "per_query": per_query,
        "per_cell": cell_summary,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[hw_aggregate] wrote {args.out} ({len(per_query)} queries)", file=sys.stderr)
    if cell_summary:
        print(f"[hw_aggregate] cell: mean_energy={cell_summary.get('mean_energy_j_per_query', '?')}J/q "
              f"peak_power={cell_summary.get('peak_power_w', '?')}W "
              f"peak_temp={cell_summary.get('peak_temp_c', '?')}°C",
              file=sys.stderr)


if __name__ == "__main__":
    main()
