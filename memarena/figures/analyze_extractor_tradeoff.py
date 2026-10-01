#!/usr/bin/env python3
"""Generate the Config A vs Config B extractor-trade-off scatter figure.

Produces ``paper/figures/fig_extractor_tradeoff.pdf`` (and .png): a two-cluster
scatter of the six (backend, extractor-config) cells we have for structured
memory at MemArena-L scale:

* Config A -- on-device Qwen3-0.6B extractor -- tends to preserve surface forms
  (names, dates, numbers) so D1 Cloze is high but the distilled facts are noisier
  for multi-hop reasoning.
* Config B -- closed-API gpt-4.1-mini extractor -- paraphrases session content
  into condensed facts; surface-form recall (D1) crashes but multi-hop reasoning
  (D3) sharpens.

The figure renders D1 (x-axis) vs D3 (y-axis) so the trade-off is a single
eye-magnet, and annotates each of the three backends (Memobase, MemOS, Mem0)
twice -- once per config. The two clusters sit along a near-vertical rotation
line (Config A in the lower-right, Config B in the upper-left), which is the
visual point: the extractor moves you along this axis, and the backend choice
only jitters you within your cluster.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
try:
    from .paths import figure_path
except ImportError:  # pragma: no cover - direct script execution
    from paths import figure_path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent
sys.path.insert(0, str(PAPER_DIR))

from paper_data import _load_cell_json, NEW_FROM_OLD, DIM_KEYS_PAPER  # noqa: E402


def _resolve_source_dir() -> Path:
    override = os.getenv("MEMARENA_EXTRACTOR_TRADEOFF_SRC")
    if override:
        return Path(override).expanduser()
    run_override = os.getenv("MEMARENA_RUN_DIR")
    if run_override:
        run_dir = Path(run_override).expanduser()
        candidates = [
            run_dir / "eval_results" / "memory_cache" / "memory_cache",
            run_dir / "eval_results_s2" / "memory_cache" / "memory_cache",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return REPO_ROOT / "MASim" / "runs" / "l_20260408_111046" / "eval_results" / "memory_cache" / "memory_cache"


SRC = _resolve_source_dir()
OUT_PDF = figure_path("fig_extractor_tradeoff.pdf")

# Paper-dim id -> display label. DIM_KEYS_PAPER is ["d1_cloze", "d2_metadata",
# "d3_factual_qa", "d4_cross_session", "d5_abstention", "d6_permission", "d7_exception"].
DIM_PAPER_TO_LABEL = {
    "d1_cloze":           "D1",
    "d2_metadata":        "D2",
    "d3_factual_qa":      "D3",
    "d4_cross_session":   "D4",
    "d5_abstention":      "D5",
    "d6_permission":      "D6",
}

BACKENDS = ["memobase", "memos", "mem0"]
BACKEND_LABEL = {"memobase": "Memobase", "memos": "MemOS", "mem0": "Mem0"}
BACKEND_COLOR = {
    "memobase": "#d62728",  # red
    "memos":    "#ff7f0e",  # orange
    "mem0":     "#9467bd",  # purple
}

# Extractor tier -> (display label, marker, size).
# Config A extractors (verbatim-copy on-device) get geometric markers ordered by capacity;
# Config B (paraphrasing closed-API) gets a star to stand visually apart.
EXTRACTORS = {
    "qwen3_0_6b":    ("Qwen3-0.6B",       "o", 70),
    "llama3_2_3b":   ("Llama-3.2-3B",     "s", 75),
    "qwen3_8b_awq":  ("Qwen3-8B (AWQ)",   "^", 85),
    "qwen3_32b_awq": ("Qwen3-32B (AWQ)",  "D", 95),
    "gpt41mini":     ("gpt-4.1-mini (B)", "*", 160),
}
EXTRACTOR_ORDER = ["qwen3_0_6b", "llama3_2_3b", "qwen3_8b_awq", "qwen3_32b_awq", "gpt41mini"]

# (backend, extractor) -> filename.  Mem0 only has one on-device cell (0.6B)
# plus its gpt-4.1-mini comparison; the SDK crashes on Config-A ingest past
# that tier (see \S app:memsys-impl).
CELL_FILES: dict[tuple[str, str], str] = {
    ("memobase", "qwen3_0_6b"):    "evaluation_results_memcache_memobase_A_paired_qwen3_0_6b.json",
    ("memobase", "llama3_2_3b"):   "evaluation_results_memcache_memobase_A_paired_llama3_2_3b.json",
    ("memobase", "qwen3_8b_awq"):  "evaluation_results_memcache_memobase_A_paired_qwen3_8b_awq.json",
    ("memobase", "qwen3_32b_awq"): "evaluation_results_memcache_memobase_A_paired_qwen3_32b_awq.json",
    ("memos",    "qwen3_0_6b"):    "evaluation_results_memcache_memos_A_paired_qwen3_0_6b.json",
    ("memos",    "llama3_2_3b"):   "evaluation_results_memcache_memos_A_paired_llama3_2_3b.json",
    ("memos",    "qwen3_8b_awq"):  "evaluation_results_memcache_memos_A_paired_qwen3_8b_awq.json",
    ("memos",    "qwen3_32b_awq"): "evaluation_results_memcache_memos_A_paired_qwen3_32b_awq.json",
    ("mem0",     "qwen3_0_6b"):    "evaluation_results_memcache_mem0_A_paired_qwen3_0_6b_vanilla.json",
    ("memobase", "gpt41mini"):     "evaluation_results_memcache_memobase_B_remote_gpt41mini.json",
    ("memos",    "gpt41mini"):     "evaluation_results_memcache_memos_B_remote_gpt41mini.json",
    ("mem0",     "gpt41mini"):     "evaluation_results_memcache_mem0_B_remote_gpt41mini_production.json",
}


def _per_dim_accuracy(path: Path) -> dict[str, float]:
    """Pool sub-dim correct/total via paper_data's loader, then aggregate to paper dims."""
    per_sub_dim = _load_cell_json(path)  # {sub_dim_name -> (correct, total)}
    out: dict[str, float] = {}
    for paper_dim, subs in NEW_FROM_OLD.items():
        label = DIM_PAPER_TO_LABEL.get(paper_dim)
        if label is None:
            continue
        c = sum(per_sub_dim.get(s, (0, 0))[0] for s in subs)
        n = sum(per_sub_dim.get(s, (0, 0))[1] for s in subs)
        out[label] = c / n if n else float("nan")
    return out


def _render(points: dict, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Connect the four on-device A cells per backend with a faint line showing
    # the extractor-capacity ladder from 0.6B to 32B. This makes the monotonic
    # climb visible at a glance without cluttering the legend.
    for b in ("memobase", "memos"):
        xs, ys = [], []
        for ex in ("qwen3_0_6b", "llama3_2_3b", "qwen3_8b_awq", "qwen3_32b_awq"):
            if (b, ex) not in points:
                continue
            xs.append(points[(b, ex)]["D1"] * 100)
            ys.append(points[(b, ex)]["D3"] * 100)
        ax.plot(xs, ys, color=BACKEND_COLOR[b], alpha=0.35, lw=1.2, zorder=1)

    # Draw each cell. Marker = extractor; color = backend.
    for (b, ex), per_dim in points.items():
        label, marker, size = EXTRACTORS[ex]
        ax.scatter(
            per_dim["D1"] * 100, per_dim["D3"] * 100,
            marker=marker, s=size,
            facecolor=BACKEND_COLOR[b], edgecolor="black", linewidth=0.6,
            zorder=3,
        )

    # Manual two-legend layout: backend (color) and extractor (marker).
    from matplotlib.lines import Line2D
    backend_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=BACKEND_COLOR[b], markersize=10,
               markeredgecolor="black", markeredgewidth=0.6,
               label=BACKEND_LABEL[b])
        for b in BACKENDS
    ]
    extractor_handles = [
        Line2D([0], [0], marker=EXTRACTORS[ex][1], color="w",
               markerfacecolor="#777777", markersize=(9 if EXTRACTORS[ex][1] != "*" else 12),
               markeredgecolor="black", markeredgewidth=0.6,
               label=EXTRACTORS[ex][0])
        for ex in EXTRACTOR_ORDER
    ]
    leg1 = ax.legend(handles=backend_handles, title="Backend",
                     loc="upper left", bbox_to_anchor=(0.02, 0.98),
                     fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=extractor_handles, title="Extractor",
              loc="upper left", bbox_to_anchor=(0.02, 0.72),
              fontsize=8, title_fontsize=9, frameon=False)

    ax.set_xlabel("D1 Cloze accuracy (%)")
    ax.set_ylabel("D3 Factual-QA accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(30, 90)
    ax.grid(True, alpha=0.25)
    ax.set_title("On-device extractor scaling (A) vs closed-API paraphrasing (B)", fontsize=10)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SRC.exists():
        for path in (OUT_PDF, OUT_PDF.with_suffix(".png")):
            if path.exists():
                path.unlink()
        print(
            f"[fig:extractor-tradeoff] skipped: source directory not found: {SRC}. "
            "Set MEMARENA_EXTRACTOR_TRADEOFF_SRC to reproduce this optional comparison."
        )
        return

    points: dict[tuple[str, str], dict[str, float]] = {}
    for (b, ex), fname in CELL_FILES.items():
        path = SRC / fname
        if not path.exists():
            continue
        per_dim = _per_dim_accuracy(path)
        if math.isfinite(per_dim.get("D1", float("nan"))) and math.isfinite(per_dim.get("D3", float("nan"))):
            points[(b, ex)] = per_dim

    if not points:
        for path in (OUT_PDF, OUT_PDF.with_suffix(".png")):
            if path.exists():
                path.unlink()
        print(f"[fig:extractor-tradeoff] skipped: no complete cells found under {SRC}")
        return

    # Print table: Backend, Extractor, D1..D6 so downstream consumers can sanity-check.
    print(f"{'Backend':<10}{'Extractor':<22}{'D1':>6}{'D2':>6}{'D3':>6}{'D4':>6}{'D5':>6}{'D6':>6}")
    for b in BACKENDS:
        for ex in EXTRACTOR_ORDER:
            if (b, ex) not in points:
                continue
            p = points[(b, ex)]
            row = f"{BACKEND_LABEL[b]:<10}{EXTRACTORS[ex][0]:<22}"
            for dim in ("D1", "D2", "D3", "D4", "D5", "D6"):
                row += f"{p[dim]*100:6.1f}"
            print(row)

    _render(points, OUT_PDF)
    print(f"[fig:extractor-tradeoff] wrote {OUT_PDF}")
    print(f"[fig:extractor-tradeoff] wrote {OUT_PDF.with_suffix('.png')}")


if __name__ == "__main__":
    main()
