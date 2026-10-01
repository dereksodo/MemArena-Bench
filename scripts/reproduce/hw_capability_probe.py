#!/usr/bin/env python3
"""Hardware capability probe for DGX Spark edge-deployment measurements.

Writes a JSON report to stdout and to sanity_reports/spark_hw_capability_<date>.json.
Checks NVML power/temp/mem/util, tegrastats, power_supply sysfs, ipmitool, nvidia-smi.
"""

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

def run_cmd(cmd, timeout=10):
    """Run a shell command, return (stdout, stderr, returncode)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except Exception as e:
        return "", str(e), -1

def main():
    report = {}
    report["probe_utc"] = datetime.now(timezone.utc).isoformat()
    report["hostname"], _, _ = run_cmd("hostname")
    report["arch"], _, _ = run_cmd("uname -m")

    # NVML probing
    nvml_report = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml_report["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        nvml_report["device_name"] = pynvml.nvmlDeviceGetName(handle)

        # Power
        try:
            pw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
            nvml_report["power_usage_works"] = True
            nvml_report["power_usage_mw_sample"] = pw
        except Exception as e:
            nvml_report["power_usage_works"] = False
            nvml_report["power_usage_error"] = str(e)

        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            nvml_report["temperature_works"] = True
            nvml_report["temperature_c_sample"] = temp
        except Exception as e:
            nvml_report["temperature_works"] = False
            nvml_report["temperature_error"] = str(e)

        # Memory
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            nvml_report["memory_info_works"] = True
            nvml_report["memory_total_gib"] = round(mem.total / (1024**3), 2)
            nvml_report["memory_used_gib"] = round(mem.used / (1024**3), 2)
            nvml_report["memory_free_gib"] = round(mem.free / (1024**3), 2)
        except Exception as e:
            nvml_report["memory_info_works"] = False
            nvml_report["memory_info_error"] = str(e)

        # Utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            nvml_report["utilization_works"] = True
            nvml_report["gpu_util_pct_sample"] = util.gpu
            nvml_report["mem_util_pct_sample"] = util.memory
        except Exception as e:
            nvml_report["utilization_works"] = False
            nvml_report["utilization_error"] = str(e)

        # Max query rate for power: sample 100 times in tight loop
        try:
            t0 = time.monotonic()
            errors = 0
            for _ in range(100):
                try:
                    pynvml.nvmlDeviceGetPowerUsage(handle)
                except Exception:
                    errors += 1
            elapsed_ms = (time.monotonic() - t0) * 1000
            nvml_report["power_100_samples_elapsed_ms"] = round(elapsed_ms, 2)
            nvml_report["power_100_samples_errors"] = errors
            nvml_report["power_10hz_feasible"] = (elapsed_ms < 10000 and errors == 0)
        except Exception as e:
            nvml_report["power_rate_test_error"] = str(e)

        pynvml.nvmlShutdown()
    except Exception as e:
        nvml_report["init_error"] = str(e)

    report["nvml"] = nvml_report

    # tegrastats
    report["tegrastats_on_path"] = shutil.which("tegrastats") is not None

    # /sys/class/power_supply/
    ps_path = "/sys/class/power_supply/"
    if os.path.isdir(ps_path):
        entries = os.listdir(ps_path)
        report["power_supply_sysfs"] = entries if entries else "empty"
    else:
        report["power_supply_sysfs"] = "not_found"

    # ipmitool
    if shutil.which("ipmitool"):
        stdout, stderr, rc = run_cmd("ipmitool sdr type 'Current'", timeout=5)
        report["ipmitool"] = {
            "found": True,
            "returncode": rc,
            "stdout_preview": stdout[:500] if stdout else "",
            "stderr_preview": stderr[:500] if stderr else ""
        }
    else:
        report["ipmitool"] = {"found": False}

    # nvidia-smi
    stdout, stderr, rc = run_cmd("nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader")
    report["nvidia_smi"] = {
        "output": stdout,
        "returncode": rc,
        "stderr": stderr[:200] if stderr else ""
    }

    # Model checkpoints
    models_dir = None
    for d in ["/models", os.path.expanduser("~/models")]:
        if os.path.isdir(d):
            models_dir = d
            break
    if models_dir:
        report["models_dir"] = models_dir
        report["models_present"] = sorted(os.listdir(models_dir))
    else:
        report["models_dir"] = None
        report["models_present"] = []

    # Required models check
    required = ["Qwen3-0.6B", "Llama-3.2-3B", "Qwen3-8B", "Qwen3-32B-AWQ"]
    present = set(report["models_present"])
    report["required_models"] = {}
    for m in required:
        if m in present:
            report["required_models"][m] = "FOUND"
        else:
            # Check for close matches
            close = [p for p in present if m.lower().replace("-", "") in p.lower().replace("-", "")]
            if close:
                report["required_models"][m] = f"MISSING (close match: {close})"
            else:
                report["required_models"][m] = "MISSING"

    # Output
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = Path("sanity_reports") / f"spark_hw_capability_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nWritten to {out_path}")

if __name__ == "__main__":
    main()
