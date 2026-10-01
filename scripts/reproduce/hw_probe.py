#!/usr/bin/env python3
"""Background hardware telemetry sampler for Spark edge-deployment measurements.

Usage:
    python scripts/hw_probe.py --outfile spark_results_s1/hw/hw_probe_qwen3_8b_vanilla_answerA.csv \
                               --interval-ms 100

Writes one CSV row per sample. Columns:
    ts_epoch_ms, power_w, sys_mem_used_gib, sys_mem_avail_gib, gpu_util_pct,
    temp_c, sm_mhz, cpu_pct, sglang_rss_gib, memobase_rss_gib, memos_rss_gib

Flushes after every row so readers (hw_aggregate.py) can tail concurrently.
Terminates on SIGTERM / SIGINT; writes a final row and closes cleanly.
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time

import psutil

# NVML setup — GB10 supports power, temp, util, clock. NOT memory_info.
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:
    _NVML_OK = False
    _GPU_HANDLE = None

_RUNNING = True


def _signal_handler(signum, frame):
    global _RUNNING
    _RUNNING = False


def _get_power_w():
    """GPU power in watts via NVML."""
    if not _NVML_OK:
        return 0.0
    try:
        return pynvml.nvmlDeviceGetPowerUsage(_GPU_HANDLE) / 1000.0
    except Exception:
        return 0.0


def _get_temp_c():
    if not _NVML_OK:
        return 0.0
    try:
        return float(pynvml.nvmlDeviceGetTemperature(_GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return 0.0


def _get_gpu_util():
    if not _NVML_OK:
        return 0.0
    try:
        return float(pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE).gpu)
    except Exception:
        return 0.0


def _get_sm_mhz():
    if not _NVML_OK:
        return 0
    try:
        return int(pynvml.nvmlDeviceGetClockInfo(_GPU_HANDLE, pynvml.NVML_CLOCK_SM))
    except Exception:
        return 0


def _get_container_rss(name_prefix):
    """Get container RSS in GiB via docker stats. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith(name_prefix):
                mem_str = parts[1]  # e.g. "1.234GiB"
                if "GiB" in mem_str:
                    return float(mem_str.replace("GiB", ""))
                elif "MiB" in mem_str:
                    return float(mem_str.replace("MiB", "")) / 1024.0
                elif "KiB" in mem_str:
                    return float(mem_str.replace("KiB", "")) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outfile", required=True, help="Output CSV path")
    parser.add_argument("--interval-ms", type=int, default=100, help="Sample interval in ms (default: 100 = 10Hz)")
    parser.add_argument("--skip-docker", action="store_true",
                        help="Skip docker stats calls (faster sampling, no per-container RSS)")
    args = parser.parse_args()

    interval_s = args.interval_ms / 1000.0

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

    fieldnames = [
        "ts_epoch_ms", "power_w", "sys_mem_used_gib", "sys_mem_avail_gib",
        "gpu_util_pct", "temp_c", "sm_mhz", "cpu_pct",
        "sglang_rss_gib", "memobase_rss_gib", "memos_rss_gib",
    ]

    with open(args.outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        print(f"[hw_probe] sampling at {args.interval_ms}ms to {args.outfile}", file=sys.stderr)
        n_samples = 0

        while _RUNNING:
            t0 = time.monotonic()

            vm = psutil.virtual_memory()
            row = {
                "ts_epoch_ms": int(time.time() * 1000),
                "power_w": round(_get_power_w(), 2),
                "sys_mem_used_gib": round(vm.used / (1024**3), 3),
                "sys_mem_avail_gib": round(vm.available / (1024**3), 3),
                "gpu_util_pct": round(_get_gpu_util(), 1),
                "temp_c": round(_get_temp_c(), 1),
                "sm_mhz": _get_sm_mhz(),
                "cpu_pct": round(psutil.cpu_percent(interval=None), 1),
                "sglang_rss_gib": 0.0,
                "memobase_rss_gib": 0.0,
                "memos_rss_gib": 0.0,
            }

            # Docker stats are slow (~1s); only sample every 10th iteration
            if not args.skip_docker and n_samples % 10 == 0:
                row["sglang_rss_gib"] = round(_get_container_rss("sglang"), 3)
                row["memobase_rss_gib"] = round(_get_container_rss("memobase"), 3)
                row["memos_rss_gib"] = round(_get_container_rss("memos"), 3)

            writer.writerow(row)
            f.flush()
            n_samples += 1

            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval_s - elapsed)
            if sleep_time > 0 and _RUNNING:
                time.sleep(sleep_time)

    print(f"[hw_probe] stopped after {n_samples} samples", file=sys.stderr)

    if _NVML_OK:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
