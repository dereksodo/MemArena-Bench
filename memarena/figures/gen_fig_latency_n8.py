"""Latency overview figure for the N_EGOS=8 spark-s2 trial.

Six panels in a 2x3 grid:
  Row 1 (per-query, from answer_results JSON):
    (a) Median TTFT (ms)
    (b) Median total answer time (ms)
    (c) Median prompt tokens
  Row 2 (per-cell hardware, from hw_agg JSON):
    (d) Mean GPU energy per query (J)
    (e) Peak GPU temperature (deg C)
    (f) Mean GPU power (W)

5 readers on the x-axis; backends as grouped colored bars. Cells with no
data for a given panel are simply omitted (e.g., memobase has no hw probe;
memsearch has no answer_results JSON in this trial).

Output:
  out/reproduced_figures/figures/fig_latency_n8.{pdf,png}
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paths import figure_path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LATENCY_ROOT = REPO_ROOT / "out"

OUT_PDF = figure_path("fig_latency_n8.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")

READERS = ["0_6b", "llama3b", "7b", "8b", "32b"]
READER_LABEL = {
    "0_6b":    "Qwen3-0.6B",
    "llama3b": "Llama-3.2-3B",
    "7b":      "Mistral-7B",
    "8b":      "Qwen3-8B",
    "32b":     "Qwen3-32B-AWQ",
}

# Backend → (label, color, has_perquery, has_hw)
BACKENDS = [
    ("vanilla",   "Vanilla",   "#7f7f7f", True,  True),
    ("inmem",     "RAG",       "#1f77b4", True,  True),
    ("oracle",    "Oracle",    "#2ca02c", True,  True),
    ("memobase",  "Memobase",  "#d62728", True,  False),
    ("memsearch", "MemSearch", "#ff7f0e", False, True),
]


def _per_query_path(reader: str, backend: str) -> Path | None:
    """Locate the per-query answer_results JSON for one (reader, backend)."""
    base = LATENCY_ROOT / f"latency_spark_{reader}_s2" / "eval_results_s2"
    if backend == "vanilla":
        p = base / "vanilla" / f"answer_results_vanilla_{reader}_s2.json"
    elif backend == "oracle":
        p = base / "oracle" / f"answer_results_oracle_{reader}_s2.json"
    elif backend == "inmem":
        p = base / "inmem" / f"answer_results_baseline_simplerag_{reader}_s2.json"
    elif backend == "memobase":
        p = base / "memory_cache" / f"answer_results_memobase_{reader}_s2.json"
    else:
        return None
    return p if p.exists() else None


def _hw_path(reader: str, backend: str) -> Path | None:
    base = LATENCY_ROOT / f"latency_spark_{reader}_s2" / "hw"
    name = "baseline_simplerag" if backend == "inmem" else backend
    p = base / f"hw_agg_{name}_{reader}_s2.json"
    return p if p.exists() else None


def _load_per_query(reader: str, backend: str) -> list[dict]:
    p = _per_query_path(reader, backend)
    if p is None:
        return []
    return json.loads(p.read_text())


def _load_hw(reader: str, backend: str) -> dict:
    p = _hw_path(reader, backend)
    if p is None:
        return {}
    return json.loads(p.read_text()).get("per_cell", {})


def _median(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return statistics.median(xs) if xs else float("nan")


def _grouped_bars(ax, values, ylabel, title, log=False):
    """values[backend_key] = [val per reader]"""
    n_back = len(values)
    width = 0.16
    x = np.arange(len(READERS))
    for i, (key, label, color) in enumerate(values["__order"]):
        ys = values[key]
        offset = (i - (n_back - 1) / 2) * width
        bars = ax.bar(x + offset, ys, width, color=color, label=label,
                      edgecolor="white", linewidth=0.4)
        # Annotate bar tops where data exists, but only when n_back ≤ 5 to
        # avoid clutter
        for b, y in zip(bars, ys):
            if not (isinstance(y, float) and np.isnan(y)):
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height(),
                        f"{y:.0f}" if y >= 10 else f"{y:.1f}",
                        ha="center", va="bottom", fontsize=6, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([READER_LABEL[r] for r in READERS],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def _collect_perquery():
    """Return {metric: {backend_key: [median per reader]}}."""
    metrics = {"ttft_ms": {}, "answer_time_ms": {}, "prompt_tokens": {}}
    order = []
    for backend, label, color, has_pq, has_hw in BACKENDS:
        if not has_pq:
            continue
        order.append((backend, label, color))
        for m in metrics:
            metrics[m].setdefault(backend, [])
        for r in READERS:
            rows = _load_per_query(r, backend)
            for m in metrics:
                vals = [x.get(m) for x in rows if x.get("timing_reliable", True)]
                metrics[m][backend].append(_median(vals))
    for m in metrics:
        metrics[m]["__order"] = order
    return metrics


def _collect_hw():
    metrics = {
        "mean_gpu_energy_j_per_query": {},
        "peak_temp_c": {},
        "mean_power_w": {},
    }
    order = []
    for backend, label, color, has_pq, has_hw in BACKENDS:
        if not has_hw:
            continue
        order.append((backend, label, color))
        for m in metrics:
            metrics[m].setdefault(backend, [])
        for r in READERS:
            d = _load_hw(r, backend)
            for m in metrics:
                metrics[m][backend].append(d.get(m, float("nan")))
    for m in metrics:
        metrics[m]["__order"] = order
    return metrics


def main():
    pq = _collect_perquery()
    hw = _collect_hw()

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))

    _grouped_bars(axes[0, 0], pq["ttft_ms"],
                  "Median TTFT (ms)",
                  "(a) Time to first token  [per query]",
                  log=True)
    _grouped_bars(axes[0, 1], pq["answer_time_ms"],
                  "Median answer time (ms)",
                  "(b) Total answer time  [per query]",
                  log=True)
    _grouped_bars(axes[0, 2], pq["prompt_tokens"],
                  "Median prompt tokens",
                  "(c) Prompt token count  [per query]",
                  log=False)
    _grouped_bars(axes[1, 0], hw["mean_gpu_energy_j_per_query"],
                  "GPU energy / query (J)",
                  "(d) Per-query GPU energy  [hw probe]",
                  log=True)
    _grouped_bars(axes[1, 1], hw["peak_temp_c"],
                  "Peak GPU temp (°C)",
                  "(e) Peak GPU temperature  [hw probe]",
                  log=False)
    _grouped_bars(axes[1, 2], hw["mean_power_w"],
                  "Mean GPU power (W)",
                  "(f) Mean GPU power  [hw probe]",
                  log=False)

    handles = []
    seen = set()
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); seen.add(l)
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Latency overview — N_EGOS=8 spark-s2 trial",
        fontsize=12, y=1.00)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")

    print()
    print("=== Per-cell latency table ===")
    print(f"{'reader':<14s} {'backend':<10s}  {'TTFT_ms':>8s} {'total_ms':>9s} "
          f"{'prompt_tk':>9s}  {'J/query':>8s} {'peak_C':>7s} {'pwr_W':>6s}")
    for r in READERS:
        for backend, label, color, has_pq, has_hw in BACKENDS:
            ttft = pq["ttft_ms"].get(backend, [float("nan")] * len(READERS))[READERS.index(r)] if has_pq else float("nan")
            tot  = pq["answer_time_ms"].get(backend, [float("nan")] * len(READERS))[READERS.index(r)] if has_pq else float("nan")
            ptk  = pq["prompt_tokens"].get(backend, [float("nan")] * len(READERS))[READERS.index(r)] if has_pq else float("nan")
            j    = hw["mean_gpu_energy_j_per_query"].get(backend, [float("nan")] * len(READERS))[READERS.index(r)] if has_hw else float("nan")
            t    = hw["peak_temp_c"].get(backend, [float("nan")] * len(READERS))[READERS.index(r)] if has_hw else float("nan")
            pw   = hw["mean_power_w"].get(backend, [float("nan")] * len(READERS))[READERS.index(r)] if has_hw else float("nan")
            print(f"{r:<14s} {label:<10s}  "
                  f"{('--' if np.isnan(ttft) else f'{ttft:8.0f}'):>8s} "
                  f"{('--' if np.isnan(tot)  else f'{tot:9.0f}'):>9s} "
                  f"{('--' if np.isnan(ptk)  else f'{ptk:9.0f}'):>9s}  "
                  f"{('--' if np.isnan(j)    else f'{j:8.2f}'):>8s} "
                  f"{('--' if np.isnan(t)    else f'{t:7.1f}'):>7s} "
                  f"{('--' if np.isnan(pw)   else f'{pw:6.1f}'):>6s}")
        print()


if __name__ == "__main__":
    main()
