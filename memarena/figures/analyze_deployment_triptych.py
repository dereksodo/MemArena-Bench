#!/usr/bin/env python3
"""Generate a 3-panel deployment figure for Section 7.3.

Panels:
    (a) Online serving cost on representative tiers.
    (b) Answer-time energy / thermal behaviour.
    (c) Ingest overhead for Qwen3-8B backends.

The figure is intentionally opinionated: each panel is optimized to communicate
one deployment finding at a glance, rather than exhaustively plotting every
available metric.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

try:
    from .paths import figure_path
except ImportError:  # pragma: no cover - direct script execution
    from paths import figure_path


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent


def _resolve_spark_root() -> Path:
    override = os.getenv("MEMARENA_SPARK_RUN_DIR") or os.getenv("MEMARENA_DEPLOYMENT_ROOT")
    if override:
        return Path(override).expanduser()
    run_override = os.getenv("MEMARENA_RUN_DIR")
    if run_override:
        return Path(run_override).expanduser() / "spark_results_s1"
    return REPO_ROOT / "MASim" / "runs" / "l_20260408_111046" / "spark_results_s1"


SPARK_S1 = _resolve_spark_root()
HW_DIR = SPARK_S1 / "hw"
MEMCACHE_DIR = SPARK_S1 / "paperhook" / "memory_cache"
OUT_PDF = figure_path("fig_deployment_triptych.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")


BACKEND_ORDER = ["vanilla", "inmem", "oracle", "memobase", "memos"]
BACKEND_LABEL = {
    "vanilla": "Vanilla",
    "inmem": "RAG",
    "oracle": "Oracle",
    "memobase": "Memobase",
    "memos": "MemOS",
}
BACKEND_COLOR = {
    "vanilla": "#7f7f7f",
    "inmem": "#1f77b4",
    "oracle": "#2ca02c",
    "memobase": "#d62728",
    "memos": "#ff7f0e",
}


def _load_json(path: Path):
    return json.loads(path.read_text())


def _answer_rows(backend: str, model: str) -> list[dict]:
    if backend in {"vanilla", "inmem", "oracle"}:
        path = SPARK_S1 / backend / model / "answer_results_run.json"
    else:
        path = MEMCACHE_DIR / f"answer_results_paperhook_{backend}_{model}_s1.json"
    return _load_json(path)


def _answer_stats(backend: str, model: str) -> dict[str, float]:
    rows = _answer_rows(backend, model)
    return {
        "total_s": median(r["answer_time_ms"] / 1000.0 for r in rows),
        "ttft_ms": median(r["ttft_ms"] for r in rows),
        "prompt_toks": median(r["prompt_tokens"] for r in rows),
    }


def _hw_answer_stats(backend: str, model: str) -> dict[str, float] | None:
    if backend in {"vanilla", "inmem", "oracle"}:
        path = HW_DIR / f"hw_agg_{model}_{backend}.json"
    else:
        path = HW_DIR / f"hw_agg_{model}_{backend}_answerB.json"
    if not path.exists():
        return None
    d = _load_json(path)
    per_cell = d["per_cell"]
    return {
        "mean_j": float(per_cell.get("mean_energy_j_per_query") or per_cell.get("mean_energy_j")),
        "peak_temp_c": float(per_cell["peak_temp_c"]),
    }


def _ingest_stats(backend: str, model: str) -> dict[str, float]:
    d = _load_json(HW_DIR / f"hw_agg_{model}_{backend}_ingest.json")
    ingest_rows = [row for row in d["per_query"] if "ingest" in row["instance_id"]]
    summary = _load_json(SPARK_S1 / "memory_cache" / f"memcache_{backend}_{model}_spark8.summary.json")
    return {
        "time_min": float(summary["elapsed_seconds"]) / 60.0,
        "energy_kj": sum(row["energy_j_net"] for row in ingest_rows) / 1000.0,
    }


# Main-table Avg values for the Qwen3-8B row in memarena/figures/tables/main_L.tex.
QWEN8_AVG = {
    "inmem": 50.1,
    "memobase": 74.7,
    "memos": 74.8,
}


def _style_bar(bar, backend: str) -> None:
    bar.set_edgecolor("black")
    bar.set_linewidth(0.7)
    if backend == "oracle":
        bar.set_alpha(0.6)
        bar.set_hatch("//")


def _panel_a(ax) -> None:
    models = [
        ("qwen3_0.6b", "Qwen3-0.6B"),
        ("qwen3_8b", "Qwen3-8B"),
    ]
    width = 0.14
    xs = np.arange(len(models))

    for i, backend in enumerate(BACKEND_ORDER):
        ys = []
        stats = []
        for model, _ in models:
            s = _answer_stats(backend, model)
            ys.append(s["total_s"])
            stats.append(s)
        offset = (i - (len(BACKEND_ORDER) - 1) / 2.0) * width
        bars = ax.bar(xs + offset, ys, width=width, color=BACKEND_COLOR[backend], label=BACKEND_LABEL[backend])
        for bar in bars:
            _style_bar(bar, backend)
        for bar, s in zip(bars, stats):
            label = f"{s['ttft_ms']:.0f}ms\n{s['prompt_toks'] / 1000.0:.1f}K"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.06,
                label,
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in models], fontsize=9)
    ax.set_ylabel("Median total latency (s)", fontsize=10)
    ax.set_title("Online Serving Cost", fontsize=11, pad=10)
    ax.set_ylim(0, 4.95)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.5,
        -0.24,
        "(a)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.0,
        0.95,
        "Labels: first-token ms / prompt K",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
    )


def _panel_b(ax) -> None:
    models = [
        ("qwen3_8b", "Qwen3-8B"),
        ("qwen3_32b_awq", "Qwen3-32B AWQ"),
    ]
    width = 0.14
    xs = np.arange(len(models))

    for i, backend in enumerate(BACKEND_ORDER):
        offset = (i - (len(BACKEND_ORDER) - 1) / 2.0) * width
        for j, (model, _) in enumerate(models):
            s = _hw_answer_stats(backend, model)
            x = xs[j] + offset
            if s is None:
                ax.text(x, 35, "n/a", ha="center", va="bottom", fontsize=7, color="#666666", rotation=90)
                continue
            bar = ax.bar(x, s["mean_j"], width=width, color=BACKEND_COLOR[backend])[0]
            _style_bar(bar, backend)
            x_text = x
            if model == "qwen3_32b_awq" and backend == "oracle":
                x_text = x - 0.035
            if model == "qwen3_32b_awq" and backend == "memobase":
                x_text = x + 0.035
            ax.text(
                x_text,
                s["mean_j"] * 1.08,
                f"{s['mean_j']:.0f} J\n{s['peak_temp_c']:.0f}°C",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in models], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(30, 4200)
    ax.set_ylabel("Mean answer energy (J/query)", fontsize=10)
    ax.set_title("Answer-Time Energy and Thermals", fontsize=11, pad=10)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.text(
        0.5,
        -0.24,
        "(b)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.0,
        0.95,
        "Labels: mean J/query / peak temp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
    )


def _panel_c(ax) -> None:
    memobase = _ingest_stats("memobase", "qwen3_8b")
    memos = _ingest_stats("memos", "qwen3_8b")
    points = [
        ("memobase", "Memobase", memobase["time_min"], memobase["energy_kj"], QWEN8_AVG["memobase"], "o"),
        ("memos", "MemOS", memos["time_min"], memos["energy_kj"], QWEN8_AVG["memos"], "s"),
    ]

    for backend, label, time_min, energy_kj, avg, marker in points:
        ax.scatter(
            time_min,
            energy_kj,
            s=140,
            marker=marker,
            color=BACKEND_COLOR[backend],
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        if backend == "memobase":
            x_text, y_text, ha = time_min * 1.03, energy_kj * 1.01, "left"
        else:
            x_text, y_text, ha = time_min / 1.07, energy_kj * 1.02, "right"
        ax.text(
            x_text,
            y_text,
            f"{label}\n{time_min:.2f} min, {energy_kj:.2f} kJ\nAvg {avg:.1f}",
            fontsize=8,
            ha=ha,
            va="bottom",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(50, 550)
    ax.set_ylim(100, 320)
    ax.set_xlabel("Ingest / prep time (min)", fontsize=10)
    ax.set_ylabel("Ingest / prep energy (kJ)", fontsize=10)
    ax.set_title("Qwen3-8B Ingest Overhead", fontsize=11, pad=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(
        0.5,
        -0.24,
        "(c)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )


def main() -> None:
    if not SPARK_S1.exists():
        for path in (OUT_PDF, OUT_PNG):
            if path.exists():
                path.unlink()
        print(
            f"[fig:deployment-triptych] skipped: Spark timing root not found: {SPARK_S1}. "
            "Set MEMARENA_SPARK_RUN_DIR to reproduce this optional deployment figure."
        )
        return

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
    try:
        _panel_a(axes[0])
        _panel_b(axes[1])
        _panel_c(axes[2])
    except FileNotFoundError as exc:
        plt.close(fig)
        for path in (OUT_PDF, OUT_PNG):
            if path.exists():
                path.unlink()
        print(f"[fig:deployment-triptych] skipped: missing deployment input: {exc}")
        return

    handles = []
    labels = []
    for backend in BACKEND_ORDER:
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=BACKEND_COLOR[backend], edgecolor="black", linewidth=0.7)
        if backend == "oracle":
            patch.set_alpha(0.6)
            patch.set_hatch("//")
        handles.append(patch)
        labels.append(BACKEND_LABEL[backend])
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig:deployment-triptych] wrote {OUT_PDF}")
    print(f"[fig:deployment-triptych] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
