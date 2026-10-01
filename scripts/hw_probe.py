#!/usr/bin/env python3
"""Background hardware telemetry sampler for Spark edge-deployment measurements.

Usage:
    python scripts/hw_probe.py --outfile spark_results_s1/hw/hw_probe_qwen3_8b_vanilla_answerA.csv \
                               --interval-ms 100

Writes one CSV row per sample. Columns:

  GPU (NVML, GB10)
    gpu_power_w, gpu_temp_c, gpu_util_pct, sm_mhz
    (alias `power_w` kept for back-compat with old aggregators)

  CPU (psutil + sysfs)
    cpu_pct, cpu_freq_mhz, cpu_temp_c

  Memory (psutil + docker stats)
    sys_mem_used_gib, sys_mem_avail_gib,
    sglang_rss_gib, memobase_rss_gib, memos_rss_gib

  SoC power rails (sysfs hwmon, optional — auto-discovered, columns
  follow the rail's `name` attribute).
    rail_<name>_power_w  e.g. rail_VDD_CPU_CV_power_w, rail_VDD_GPU_SOC_power_w,
                              rail_VDD_DRAM_power_w
  Combined fields:
    rails_total_power_w   — sum of all discovered rails (instantaneous SoC W)
    rails_cpu_power_w     — sum of rails whose name contains "CPU"
    rails_gpu_power_w     — sum of rails whose name contains "GPU"
    rails_dram_power_w    — sum of rails whose name contains "DRAM" or "MEM"

Flushes after every row so readers can tail concurrently. Terminates on
SIGTERM / SIGINT; writes a final row and closes cleanly.
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


# ---------------------------------------------------------------------------
# CPU temperature (sysfs thermal_zone)
# ---------------------------------------------------------------------------

# Auto-discover the first non-trivial CPU thermal zone. On Spark / Jetson
# /sys/class/thermal/thermal_zoneX/type names are 'cpu-thermal', 'CPU-therm',
# 'soc0-cpu', etc. We pick the first whose 'type' contains 'cpu' (case-
# insensitive); fall back to the hottest non-GPU zone.
_CPU_TEMP_PATH: str | None = None


def _discover_cpu_temp_path() -> str | None:
    base = "/sys/class/thermal"
    if not os.path.isdir(base):
        return None
    cpu_zone = None
    fallback = None
    for entry in sorted(os.listdir(base)):
        if not entry.startswith("thermal_zone"):
            continue
        zd = os.path.join(base, entry)
        try:
            ztype = open(os.path.join(zd, "type")).read().strip().lower()
        except OSError:
            continue
        if "cpu" in ztype:
            cpu_zone = os.path.join(zd, "temp")
            break
        if "gpu" not in ztype and fallback is None:
            fallback = os.path.join(zd, "temp")
    return cpu_zone or fallback


def _get_cpu_temp_c() -> float:
    global _CPU_TEMP_PATH
    if _CPU_TEMP_PATH is None:
        _CPU_TEMP_PATH = _discover_cpu_temp_path() or ""
    if not _CPU_TEMP_PATH:
        return 0.0
    try:
        # sysfs thermal zones report milli-celsius integers
        return float(open(_CPU_TEMP_PATH).read().strip()) / 1000.0
    except (OSError, ValueError):
        return 0.0


def _get_cpu_freq_mhz() -> float:
    """Average CPU clock across cores (psutil.cpu_freq returns per-core list
    on most platforms, scalar on macOS/some ARM)."""
    try:
        freqs = psutil.cpu_freq(percpu=True)
        if isinstance(freqs, list) and freqs:
            return round(sum(f.current for f in freqs) / len(freqs), 1)
        if freqs:
            return round(float(freqs.current), 1)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# SoC power rails (sysfs hwmon)
# ---------------------------------------------------------------------------
#
# Spark / Jetson exposes per-rail power via /sys/class/hwmon/hwmonN with
# files:
#   name         -- rail label, e.g. 'VDD_CPU_CV', 'VDD_GPU_SOC', 'VDD_DRAM'
#   power1_input -- instantaneous power in microwatts
# We discover all rails once at startup; for each subsequent sample we read
# every power*_input under each hwmon and emit one column per rail name.
#
# Rails are auto-categorised:
#   * name contains 'CPU' -> contributes to rails_cpu_power_w
#   * name contains 'GPU' -> rails_gpu_power_w
#   * name contains 'DRAM' or 'MEM' -> rails_dram_power_w
# Unmatched rails still appear as `rail_<name>_power_w` and roll into
# rails_total_power_w.

_RAILS: list[tuple[str, str]] = []  # (column_name, sysfs_path)


def _discover_rails() -> list[tuple[str, str]]:
    rails: list[tuple[str, str]] = []
    base = "/sys/class/hwmon"
    if not os.path.isdir(base):
        return rails
    for entry in sorted(os.listdir(base)):
        hwmon_dir = os.path.join(base, entry)
        try:
            hwname = open(os.path.join(hwmon_dir, "name")).read().strip()
        except OSError:
            continue
        # Each hwmon may expose multiple powerN_input files. When there are
        # also powerN_label files we use those for finer-grained naming.
        try:
            files = os.listdir(hwmon_dir)
        except OSError:
            continue
        power_inputs = sorted(f for f in files if f.startswith("power") and f.endswith("_input"))
        for fname in power_inputs:
            stem = fname[:-len("_input")]
            label_path = os.path.join(hwmon_dir, f"{stem}_label")
            try:
                label = open(label_path).read().strip() if os.path.exists(label_path) else stem
            except OSError:
                label = stem
            col = f"rail_{hwname}_{label}_power_w".replace(" ", "_")
            rails.append((col, os.path.join(hwmon_dir, fname)))
    return rails


def _read_rail_uw(path: str) -> int:
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return 0


def _categorise_rail(col_name: str) -> str:
    upper = col_name.upper()
    if "CPU" in upper:
        return "cpu"
    if "GPU" in upper:
        return "gpu"
    if "DRAM" in upper or "MEM" in upper:
        return "dram"
    return "other"


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

    # Discover SoC power rails once. On hosts without hwmon this stays empty
    # and the rails_* columns simply read 0.0 every sample.
    global _RAILS
    _RAILS = _discover_rails()
    rail_cols = [col for col, _ in _RAILS]
    rail_categories = {col: _categorise_rail(col) for col, _ in _RAILS}
    if _RAILS:
        print(f"[hw_probe] discovered {len(_RAILS)} SoC power rails: "
              f"{[col for col,_ in _RAILS]}", file=sys.stderr)
    else:
        print("[hw_probe] no /sys/class/hwmon power rails — "
              "rails_* columns will be 0", file=sys.stderr)

    fieldnames = [
        "ts_epoch_ms",
        # Back-compat aliases (so legacy aggregators keep working).
        "power_w", "temp_c",
        # GPU
        "gpu_power_w", "gpu_temp_c", "gpu_util_pct", "sm_mhz",
        # CPU
        "cpu_pct", "cpu_freq_mhz", "cpu_temp_c",
        # Memory
        "sys_mem_used_gib", "sys_mem_avail_gib",
        "sglang_rss_gib", "memobase_rss_gib", "memos_rss_gib",
        # Combined SoC rail buckets
        "rails_total_power_w", "rails_cpu_power_w",
        "rails_gpu_power_w", "rails_dram_power_w",
        # Per-rail (variable, depends on platform).
    ] + rail_cols

    with open(args.outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        print(f"[hw_probe] sampling at {args.interval_ms}ms to {args.outfile}", file=sys.stderr)
        n_samples = 0

        while _RUNNING:
            t0 = time.monotonic()

            vm = psutil.virtual_memory()
            gpu_p = _get_power_w()
            gpu_t = _get_temp_c()
            row = {
                "ts_epoch_ms": int(time.time() * 1000),
                "power_w": round(gpu_p, 2),  # alias for gpu_power_w
                "temp_c": round(gpu_t, 1),    # alias for gpu_temp_c
                "gpu_power_w": round(gpu_p, 2),
                "gpu_temp_c": round(gpu_t, 1),
                "gpu_util_pct": round(_get_gpu_util(), 1),
                "sm_mhz": _get_sm_mhz(),
                "cpu_pct": round(psutil.cpu_percent(interval=None), 1),
                "cpu_freq_mhz": _get_cpu_freq_mhz(),
                "cpu_temp_c": round(_get_cpu_temp_c(), 1),
                "sys_mem_used_gib": round(vm.used / (1024**3), 3),
                "sys_mem_avail_gib": round(vm.available / (1024**3), 3),
                "sglang_rss_gib": 0.0,
                "memobase_rss_gib": 0.0,
                "memos_rss_gib": 0.0,
                "rails_total_power_w": 0.0,
                "rails_cpu_power_w": 0.0,
                "rails_gpu_power_w": 0.0,
                "rails_dram_power_w": 0.0,
            }

            # SoC rails (cheap; sysfs reads are <100us each).
            total = cpu = gpu = dram = 0.0
            for col, path in _RAILS:
                w = _read_rail_uw(path) / 1_000_000.0  # uW -> W
                row[col] = round(w, 3)
                total += w
                cat = rail_categories[col]
                if cat == "cpu":
                    cpu += w
                elif cat == "gpu":
                    gpu += w
                elif cat == "dram":
                    dram += w
            row["rails_total_power_w"] = round(total, 3)
            row["rails_cpu_power_w"] = round(cpu, 3)
            row["rails_gpu_power_w"] = round(gpu, 3)
            row["rails_dram_power_w"] = round(dram, 3)

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
