"""TTFT decomposition (search + LLM prefill) per simulated day for the two
on-device readers we have raw Spark GB10 measurements for: Qwen3-0.6B and
Llama-3.2-3B.

For each (reader, day) cell we pool inmem (BM25-RAG) per-question records and
take the median of search_time_ms and prefill_ms (= ttft_ms - search_time_ms).
inmem is the canonical backend for this view because it exercises BOTH a
non-trivial search step and a non-trivial prefill step.

Output: paper/figures/fig_ttft_per_day.{pdf,png}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from memarena.figures.paper_data import RESULTS_ROOT as ROOT  # noqa: E402  (result files live under $MEMARENA_RESULTS_DIR)
OUT_DIR = ROOT / "paper" / "figures"

QA_FILE = (ROOT / "out/memobase_writer32_7b/eval_results_s2/memory_cache/"
           "masim_qa_memobase_writer32_7b_s2_judge_remote.json")

READERS = [
    ("0_6b",     "Qwen3-0.6B"),
    ("llama3b",  "Llama-3.2-3B"),
]


def load_day_map() -> dict[str, int]:
    """question_id → simulated day (int from instance_query_timestamp)."""
    d = json.load(open(QA_FILE))
    out = {}
    for x in d["qars"]:
        ts = x.get("meta", {}).get("instance_query_timestamp")
        if ts is not None:
            out[x["id"]] = int(ts)
    return out


def load_lat(reader: str) -> list[dict]:
    """Per-question latency records for inmem backend on Spark GB10."""
    f = (ROOT / f"out/latency_spark_{reader}_s2/eval_results_s2/inmem/"
         f"answer_results_baseline_simplerag_{reader}_s2.json")
    if not f.exists():
        print(f"MISSING: {f}")
        return []
    d = json.load(open(f))
    return d if isinstance(d, list) else d.get("details", [])


def per_day_stats(rows: list[dict], day_map: dict[str, int]):
    """Pool by day, return (days, search_med, prefill_med, n)."""
    by_day = defaultdict(lambda: {"search": [], "prefill": [], "n": 0})
    for r in rows:
        qid = r.get("question_id")
        if qid not in day_map:
            continue
        ttft = r.get("ttft_ms")
        s = r.get("search_time_ms")
        if ttft is None:
            continue
        s_val = float(s) if s is not None else 0.0
        p_val = max(0.0, float(ttft) - s_val)
        day = day_map[qid]
        by_day[day]["search"].append(s_val)
        by_day[day]["prefill"].append(p_val)
        by_day[day]["n"] += 1
    days = sorted(by_day)
    s_med = [float(np.median(by_day[d]["search"])) for d in days]
    p_med = [float(np.median(by_day[d]["prefill"])) for d in days]
    n_q = [by_day[d]["n"] for d in days]
    return days, s_med, p_med, n_q


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig_ttft_per_day.pdf"
    out_png = OUT_DIR / "fig_ttft_per_day.png"

    day_map = load_day_map()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.0), sharey=False)

    for ax, (reader, label) in zip(axes, READERS):
        rows = load_lat(reader)
        days, s_med, p_med, n_q = per_day_stats(rows, day_map)
        if not days:
            ax.text(0.5, 0.5, f"no data for {label}", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        x = np.array(days)
        s = np.array(s_med)
        p = np.array(p_med)
        ax.bar(x, s, width=0.78, color="#1f77b4", edgecolor="white", lw=0.6,
               label="Memory search")
        ax.bar(x, p, bottom=s, width=0.78, color="#ff7f0e", edgecolor="white",
               lw=0.6, label="LLM prefill")
        for xi, total in zip(x, s + p):
            ax.text(xi, total + max(s + p) * 0.015, f"{total:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Simulated day", fontsize=10)
        if reader == "0_6b":
            ax.set_ylabel("Median TTFT (ms)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in x], fontsize=9)
        ax.set_ylim(0, max(s + p) * 1.18)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        ax.grid(axis="y", alpha=0.18)
        print(f"  {reader} per-day median: search={s_med}")
        print(f"  {reader} per-day median: prefill={p_med}")
        print(f"  {reader} per-day n_q: {n_q}")

    fig.suptitle("Per-day TTFT decomposition — BM25-RAG (inmem), Spark GB10 s2",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_png, bbox_inches="tight", dpi=160, pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
