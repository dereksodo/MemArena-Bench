"""Generate the two §7.3 ablation figures.

Figure F1 (anti-policy disclosure is intrinsic):
    Anti-policy gap (DENY leak rate − ALLOW DISCLOSE_CORRECT rate)
    vs reader scale, pooled across the three Oracle-style baseline backends
    (in-mem text sessions / Oracle + curated distractor / Oracle + random
    distractor) and three seeds. Bigger readers leak more on DENY, not less.

Figure F2 (matched-evidence is the bottleneck for cross-session reasoning):
    D4 accuracy across nine evidence pathways, pooled over readers and
    seeds. Oracle row reaches 79.8\\% while every retrieval / no-retrieval /
    omniscient variant lands ≤30\\%, isolating matched-evidence provenance as
    the lever.

Outputs:
  paper/figures/fig_ablation_f1.{pdf,png}
  paper/figures/fig_ablation_f2.{pdf,png}
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_CSV = REPO_ROOT / "experiments_index.csv"
OUT_DIR = REPO_ROOT / "paper" / "figures"


# qid prefix → paper dim (for D4 cross-session reasoning).
PAPER_DIM = {
    "d6":  "D2", "d7":  "D3", "d8":  "D3", "d10": "D3",
    "d1":  "D4", "d2":  "D4", "d3":  "D5",
}
READER_ORDER = ["0_6b", "llama3b", "7b", "8b", "32b"]
READER_LABEL = {"0_6b": "Qwen3-0.6B", "llama3b": "Llama-3.2-3B",
                "7b": "Mistral-7B", "8b": "Qwen3-8B", "32b": "Qwen3-32B"}


def _file_d6_d4(p: Path):
    """Return per-dim {n_total, n_correct} dict + D6 accumulators."""
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    out = {"D4_t": 0, "D4_c": 0,
           "ALLOW": 0, "ALLOW_DC": 0,
           "DENY": 0, "DENY_LEAK": 0}
    for r in d.get("details", []):
        qid = str(r.get("question_id", ""))
        if qid.startswith("d4_perm"):
            mode = str(r.get("expected_answer_mode") or "").lower()
            cat = str(r.get("policy_category") or "").upper()
            leaked = bool(r.get("leaked_fact_in_output", False))
            if mode == "disclose":
                out["ALLOW"] += 1
                if cat == "DISCLOSE_CORRECT":
                    out["ALLOW_DC"] += 1
            elif mode in ("deny", "abstain"):
                out["DENY"] += 1
                if leaked:
                    out["DENY_LEAK"] += 1
            continue
        prefix = qid.split("_", 1)[0] if "_" in qid else qid
        if PAPER_DIM.get(prefix) != "D4":
            continue
        out["D4_t"] += 1
        if r.get("correct"):
            out["D4_c"] += 1
    return out


def _parse_ablation_path(p: Path):
    """Return (group, reader, backend) for an ablations_l_* file."""
    parts = p.relative_to(REPO_ROOT / "out").parts
    g = parts[0].removeprefix("ablations_l_")
    rdr = None
    for r in ["_0_6b", "_llama3b", "_7b", "_8b", "_32b"]:
        if g.endswith(r):
            rdr = r[1:]
            g = g[:-len(r)]
            break
    if rdr is None and g in set(READER_ORDER):
        rdr = g
        g = "baseline"
    backend = parts[2] if len(parts) > 2 else "?"
    return g, rdr, backend


# --------------------------------------------------------------------------
# Figure F1: anti-policy gap vs reader scale
# --------------------------------------------------------------------------
def figure_f1(out_pdf: Path):
    """Bar chart: DENY-leak − ALLOW-DC per reader, pooled across the three
    baseline-Oracle backends (in-mem text sessions, Oracle+distractor variants)."""
    rows = list(csv.DictReader(INDEX_CSV.open()))
    main = [r for r in rows if r["table"] == "main_5x5x3"
            and r["trial"] in {"s2", "s3", "s4"}]

    # Use ablations_l_<reader>/{inmem_text_sessions, oracle_with_distractors,
    # oracle_with_random_distractors} cells — exactly the 45-cell baseline pool.
    agg = defaultdict(lambda: [0, 0, 0, 0])  # ALLOW, ALLOW_DC, DENY, DENY_LEAK
    for adir in (REPO_ROOT / "out").glob("ablations_l_*"):
        for p in adir.rglob("evaluation_results*.json"):
            if p.stem.endswith("_legacy"): continue
            if "/runs/" in str(p): continue
            g, rdr, _ = _parse_ablation_path(p)
            if g != "baseline" or rdr not in READER_ORDER:
                continue
            m = _file_d6_d4(p)
            if not m: continue
            v = agg[rdr]
            v[0] += m["ALLOW"]; v[1] += m["ALLOW_DC"]
            v[2] += m["DENY"];  v[3] += m["DENY_LEAK"]

    gaps, allows, leaks = [], [], []
    for rdr in READER_ORDER:
        a, adc, d, dlk = agg[rdr]
        adc_r = 100 * adc / a if a else 0
        leak_r = 100 * dlk / d if d else 0
        allows.append(adc_r)
        leaks.append(leak_r)
        gaps.append(leak_r - adc_r)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(READER_ORDER))
    bar_w = 0.36
    b1 = ax.bar(x - bar_w/2, allows, width=bar_w,
                color="#2ca02c", edgecolor="white", lw=0.6,
                label="ALLOW DISCLOSE_CORRECT rate")
    b2 = ax.bar(x + bar_w/2, leaks, width=bar_w,
                color="#d62728", edgecolor="white", lw=0.6,
                label="DENY leak rate")
    ax.axhline(0, color="black", lw=0.5)
    # Gap annotation above each pair.
    for i, gap in enumerate(gaps):
        y = max(allows[i], leaks[i]) + 4
        ax.text(i, y, f"Δ = +{gap:.1f} pp", ha="center", va="bottom",
                fontsize=8.5, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([READER_LABEL[r] for r in READER_ORDER],
                       rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Rate on D6 (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=1)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print("  per-reader DENY leak − ALLOW DC gap:")
    for rdr, gap in zip(READER_ORDER, gaps):
        print(f"    {rdr:<10s} {gap:+.1f} pp")


# --------------------------------------------------------------------------
# Figure F2: TTFT decomposition (search vs prefill) across 2 readers × 4 backends
# --------------------------------------------------------------------------
def figure_f2(out_pdf: Path):
    """Stacked-bar TTFT decomposition: 8 cells (Qwen3-0.6B / Qwen3-32B-AWQ)
       × (Vanilla / RAG / Memobase / MemSearch). Search/retrieval time is the
       bottom segment, LLM prefill the top segment. Numbers from Table 3
       (main_SML.tex) and the T_search row (ttft_main.tex)."""
    # (reader, backend, T_search_ms, T_prefill_ms)
    cells = [
        ("0.6B", "Vanilla",     0,   41),
        ("0.6B", "RAG",        87,   74),
        ("0.6B", "Memobase",    7,   25),
        ("0.6B", "MemSearch",  48,  165),
        ("32B",  "Vanilla",     0,  290),
        ("32B",  "RAG",        87, 2437),
        ("32B",  "Memobase",    7, 1482),
        ("32B",  "MemSearch",  48, 3582),
    ]
    x = np.array([0, 1, 2, 3, 4.7, 5.7, 6.7, 7.7])
    search = np.array([c[2] for c in cells])
    prefill = np.array([c[3] for c in cells])
    total = search + prefill
    labels = [c[1] for c in cells]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(x, search,  width=0.85, color="#1f77b4", edgecolor="white", lw=0.7,
           label="Memory search")
    ax.bar(x, prefill, width=0.85, bottom=search, color="#ff7f0e", edgecolor="white", lw=0.7,
           label="LLM prefill")
    for xi, t in zip(x, total):
        ax.text(xi, t + max(total) * 0.012, f"{t}", ha="center", va="bottom",
                fontsize=8.2, color="#333")
    ymax = max(total) * 1.18
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.8)
    ax.text(1.5, ymax * 0.78, "Qwen3-0.6B", ha="center", va="top",
            fontsize=10, fontweight="bold", color="#222")
    ax.text(6.2, ymax * 0.78, "Qwen3-32B-AWQ", ha="center", va="top",
            fontsize=10, fontweight="bold", color="#222")
    ax.axvline(4.0 - 0.15, color="#bbbbbb", lw=0.7, ls="--", alpha=0.8)
    ax.set_ylabel("End-to-end TTFT (ms)", fontsize=10)
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    for r, b, s, p in cells:
        print(f"  {r:<5} {b:<10} search={s:>3} prefill={p:>5} total={s+p:>5}")
    return  # skip the old D4-pooling code below

    # OLD D4-pooling code retained below for reference (unreachable)
    rows = list(csv.DictReader(INDEX_CSV.open()))
    main = [r for r in rows if r["table"] == "main_5x5x3"
            and r["trial"] in {"s2", "s3", "s4"}]
    rows = list(csv.DictReader(INDEX_CSV.open()))
    main = [r for r in rows if r["table"] == "main_5x5x3"
            and r["trial"] in {"s2", "s3", "s4"}]

    # Pool main_5x5x3 by backend.
    main_agg = defaultdict(lambda: [0, 0])
    for r in main:
        p = REPO_ROOT / r["json_path"]
        if not p.exists(): continue
        m = _file_d6_d4(p)
        if not m: continue
        v = main_agg[r["backend"]]
        v[0] += m["D4_t"]; v[1] += m["D4_c"]

    # Pool ablation cells by intent.
    abl_agg = defaultdict(lambda: [0, 0])
    for adir in (REPO_ROOT / "out").glob("ablations_l_*"):
        for p in adir.rglob("evaluation_results*.json"):
            if p.stem.endswith("_legacy"): continue
            if "/runs/" in str(p): continue
            g, rdr, bk = _parse_ablation_path(p)
            m = _file_d6_d4(p)
            if not m: continue
            if g == "baseline" and bk in {"inmem_text_sessions",
                                          "oracle_with_distractors",
                                          "oracle_with_random_distractors"}:
                # Already covered by main_agg's oracle row; skip.
                continue
            if g == "extra":
                abl_agg[bk][0] += m["D4_t"]; abl_agg[bk][1] += m["D4_c"]
            elif g == "omniscient":
                abl_agg["omniscient"][0] += m["D4_t"]
                abl_agg["omniscient"][1] += m["D4_c"]

    # Order bars: matched-evidence baselines on the left, retrieval/omniscient
    # on the right, all separated by a thin vertical guide.
    order = [
        ("Oracle",          "oracle",            main_agg, "#2ca02c"),
        ("BM25-RAG",        "inmem",             main_agg, "#1f77b4"),
        ("MemSearch",       "memsearch",         main_agg, "#ff7f0e"),
        ("Memobase",        "memobase",          main_agg, "#d62728"),
        ("Vanilla",         "vanilla",           main_agg, "#7f7f7f"),
        ("dense E5",        "dense_e5",          abl_agg,  "#888888"),
        ("dense BGE-M3",    "dense_bge_m3",      abl_agg,  "#888888"),
        ("hybrid+rerank",   "hybrid_bm25rerank", abl_agg,  "#888888"),
        ("temporal-prior",  "temporal",          abl_agg,  "#888888"),
        ("Omniscient",      "omniscient",        abl_agg,  "#a0522d"),
    ]
    labels, accs, colors = [], [], []
    for name, key, src, col in order:
        v = src.get(key, [0, 0])
        if v[0] == 0:
            continue
        accs.append(100 * v[1] / v[0])
        labels.append(name)
        colors.append(col)

    # Landscape layout: vertical bars, conditions left→right.
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, accs, color=colors, edgecolor="white", lw=0.7)
    # Highlight Oracle bar.
    if labels and labels[0] == "Oracle":
        bars[0].set_edgecolor("#1c4e9b")
        bars[0].set_linewidth(1.6)
    # Reference: Oracle ceiling.
    if labels and labels[0] == "Oracle":
        ax.axhline(accs[0], color="#1c4e9b", lw=0.6, ls=":", alpha=0.7)
        ax.text(len(labels) - 0.6, accs[0] + 1.5,
                f"Oracle ceiling {accs[0]:.1f}%",
                color="#1c4e9b", fontsize=8.0, ha="right", va="bottom")
    # Numeric labels above each bar.
    for xi, v in zip(x, accs):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
                fontsize=8.0, color="#333")
    # Group divider between main backends and retriever-side variants.
    ax.axvline(4.5, color="#bbbbbb", lw=0.6, ls="--", alpha=0.7)
    ax.text(2.0, 92, "main backends",
            ha="center", va="center", fontsize=8.0, color="#666", style="italic")
    ax.text(7.0, 92, "retrieval-side ablations / omniscient",
            ha="center", va="center", fontsize=8.0, color="#666", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.6)
    ax.set_ylabel("D4 cross-session reasoning accuracy (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print("  pooled D4 accuracy:")
    for name, v in zip(labels, accs):
        print(f"    {name:<22s} {v:5.1f}%")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_f1(OUT_DIR / "fig_ablation_f1.pdf")
    print()
    figure_f2(OUT_DIR / "fig_ablation_f2.pdf")


if __name__ == "__main__":
    main()
