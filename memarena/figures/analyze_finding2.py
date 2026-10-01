#!/usr/bin/env python3
"""Generate Figure (``fig:finding2``): memory organization closes the gap that
retrieval alone does not.

Finding 2 of MemArena: "The dominant bottleneck is memory organization and
reading, not retrieval alone." Two complementary claims:

* **Panel (a): retrieval alone leaves the gap open; organization closes it.**
  At a single Qwen3-8B anchor, side-by-side bars show Vanilla, RAG (BM25),
  Memobase, MemOS, and Oracle. The Memobase/MemOS bars reach the Oracle
  reference while RAG closes only part of the Vanilla-Oracle gap.

* **Panel (b): lift-vs-scale asymmetry.** Grouped bars per reader tier of
  (RAG - Vanilla), (Memobase - Vanilla), (MemOS - Vanilla) Avg lift. RAG
  can actually be negative at small tiers; the two structured-memory
  families grow monotonically with reader scale, contradicting the
  "memory democratizes small models" reading.

Output
------
``paper/figures/fig_finding2.pdf`` (and ``.png``).

By default this script reads the current figure input pointed to by
``MEMARENA_RUN_DIR`` via ``paper_data.DEFAULT_RUN``. Set
``MEMARENA_FINDING2_SOURCE=published`` to reproduce the archived paper
numbers that are kept below as a compatibility fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

from paper_data import (  # noqa: E402
    MODEL_TEX,
    BACKEND_TEX,
    DEFAULT_RUN,
    aggregate_seeds, category_accuracy, load_all_cells,
)
from paths import figure_path  # noqa: E402

OUT_PDF = figure_path("fig_finding2.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")

BACKENDS_A_DYNAMIC = ["vanilla", "rag", "memobase", "memsearch", "memos", "oracle"]
BACKENDS_A_PUBLISHED = ["vanilla", "rag", "dense_rag", "memobase", "memsearch", "memos", "oracle"]
BACKENDS_B = ["rag", "memobase", "memsearch", "memos"]          # lifts-over-vanilla
ANCHOR = "8b"                                       # Qwen3-8B
TIERS = ["0_6b", "llama3b", "7b", "8b", "32b"]     # all 5 readers; Mistral row included to expose the schema-compliance regression flagged in App. memsys-impl
DENSE_RAG_A = ("BGE-M3", 50.9)                      # Appendix Table p1c, Qwen3-8B, k=5

# Colors matching main-table conventions. Vanilla gray, oracle green, structured
# red/orange, RAG blue.
COLOR = {
    "vanilla":  "#7f7f7f",
    "rag":      "#1f77b4",
    "dense_rag": "#8ec7ff",
    "memobase": "#d62728",
    "memos":    "#ff7f0e",
    "oracle":   "#2ca02c",
}

LABEL_A = {
    "vanilla": BACKEND_TEX["vanilla"],
    "rag": BACKEND_TEX["rag"],
    "dense_rag": "Dense RAG",
    "memobase": BACKEND_TEX["memobase"],
    "memos": BACKEND_TEX["memos"],
    "oracle": BACKEND_TEX["oracle"],
}


# Archived from the paper-era memarena/figures/tables/main_L.tex Avg row
# (mean +- std). These values are intentionally *not* used by default anymore:
# they exist only for explicit ``MEMARENA_FINDING2_SOURCE=published`` rerenders.
PUBLISHED_AVG = {
    # model      : {backend: (mean_pp, std_pp)}
    "0_6b":    {"vanilla": (35.3, 1.0), "oracle": (50.9, 0.3), "rag": (32.8, 0.2),
                "memobase": (58.4, 6.3), "memos": (50.6, 0.5)},
    "llama3b": {"vanilla": (34.8, 0.7), "oracle": (66.3, 0.6), "rag": (12.3, 0.2),
                "memobase": (75.8, 8.6), "memos": (66.4, 0.1)},
    "7b":      {"vanilla": (34.1, 0.8), "oracle": (59.8, 0.8), "rag": (33.1, 0.8),
                "memobase": (41.4, 0.5), "memos": (58.9, 0.3)},
    "8b":      {"vanilla": (33.8, 0.2), "oracle": (74.7, 0.3), "rag": (50.1, 0.1),
                "memobase": (74.7, 0.2), "memos": (74.8, 0.5)},
    "32b":     {"vanilla": (36.1, 0.4), "oracle": (80.4, 0.2), "rag": (48.0, 0.1),
                "memobase": (87.6, 6.1), "memos": (78.9, 3.3)},
}


def _delete_outputs() -> None:
    for path in (OUT_PDF, OUT_PNG):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _cell_overall_avg(cell):
    """Macro-mean over D1..D5, matching the main-table Avg definition."""
    parts = [category_accuracy(cell, [i]) for i in [0, 1, 2, 3, 4]]
    parts = [p for p in parts if p is not None]
    return sum(parts) / len(parts) if parts else None


def _use_published_values() -> bool:
    source = os.getenv("MEMARENA_FINDING2_SOURCE", "").strip().lower()
    legacy = os.getenv("MEMARENA_FINDING2_USE_PUBLISHED", "").strip().lower()
    return source in {"published", "paper", "legacy"} or legacy in {"1", "true", "yes"}


def _load_avg_points() -> tuple[dict[str, dict[str, tuple[float, float]]], str, bool]:
    """Return Avg values in percentage points: data[model][backend] = (mean, std)."""
    if _use_published_values():
        return PUBLISHED_AVG, "published main-table Avg", True

    grid = load_all_cells(DEFAULT_RUN)
    data: dict[str, dict[str, tuple[float, float]]] = {}
    for model in TIERS:
        data[model] = {}
        for backend in BACKENDS_A_DYNAMIC:
            st = aggregate_seeds(grid, backend, model, _cell_overall_avg)
            if st.n:
                data[model][backend] = (st.mean * 100, st.std * 100)
    return data, f"current run Avg ({DEFAULT_RUN})", False


def _avg_pp(
    points: dict[str, dict[str, tuple[float, float]]],
    model: str,
    backend: str,
    *,
    published: bool,
) -> tuple[float, float] | None:
    if backend == "dense_rag":
        return (DENSE_RAG_A[1], 0.0) if published else None
    return points.get(model, {}).get(backend)


def _panel_a_backends(
    points: dict[str, dict[str, tuple[float, float]]],
    *,
    published: bool,
) -> list[str]:
    candidates = BACKENDS_A_PUBLISHED if published else BACKENDS_A_DYNAMIC
    return [bk for bk in candidates if _avg_pp(points, ANCHOR, bk, published=published) is not None]


def _panel_b_tiers(
    points: dict[str, dict[str, tuple[float, float]]],
    *,
    published: bool,
) -> list[str]:
    out = []
    for tier in TIERS:
        if _avg_pp(points, tier, "vanilla", published=published) is None:
            continue
        if any(_avg_pp(points, tier, bk, published=published) is not None for bk in BACKENDS_B):
            out.append(tier)
    return out


def _render(
    out_pdf: Path,
    points: dict[str, dict[str, tuple[float, float]]],
    *,
    published: bool,
) -> bool:
    backends_a = _panel_a_backends(points, published=published)
    tiers_b = _panel_b_tiers(points, published=published)
    if not backends_a and not tiers_b:
        return False

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # ================== Panel (a): Qwen3-8B anchor waterfall ==================
    bars_a = []
    for bk in backends_a:
        point = _avg_pp(points, ANCHOR, bk, published=published)
        if point is None:
            continue
        bars_a.append((bk, point[0], point[1]))

    if bars_a:
        xs = list(range(len(bars_a)))
        idx = {bk: i for i, (bk, _, _) in enumerate(bars_a)}
        for i, (bk, v, s) in enumerate(bars_a):
            hatch = "//" if bk == "dense_rag" else None
            axA.bar(i, v, 0.68, color=COLOR[bk], edgecolor="black", linewidth=0.6,
                    yerr=s if s > 0 else None, capsize=3, ecolor="#333", hatch=hatch)
            axA.text(i, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
                     fontsize=9, color="#222")

        oracle_point = _avg_pp(points, ANCHOR, "oracle", published=published)
        vanilla_point = _avg_pp(points, ANCHOR, "vanilla", published=published)
        rag_point = _avg_pp(points, ANCHOR, "rag", published=published)
        memobase_point = _avg_pp(points, ANCHOR, "memobase", published=published)
        memos_point = _avg_pp(points, ANCHOR, "memos", published=published)

        if oracle_point is not None:
            oracle_v = oracle_point[0]
            axA.axhline(oracle_v, color=COLOR["oracle"], lw=0.9, ls="--", alpha=0.55)
            axA.text(-0.45, oracle_v + 0.8, "Oracle ceiling",
                     ha="left", va="bottom", fontsize=8, color=COLOR["oracle"],
                     style="italic")

        if oracle_point is not None and vanilla_point is not None and rag_point is not None:
            oracle_v = oracle_point[0]
            vanilla_v = vanilla_point[0]
            rag_v = rag_point[0]
            gap_total = oracle_v - vanilla_v
            if abs(gap_total) > 1e-9 and "rag" in idx:
                rag_closes_pct = 100 * (rag_v - vanilla_v) / gap_total
                axA.annotate(
                    f"RAG closes\n{rag_closes_pct:.0f}% of the gap",
                    xy=(idx["rag"], rag_v - 2), xytext=(0.0, max(4, vanilla_v - 18)),
                    fontsize=8.5, ha="left", color="#222",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#666"),
                )

        dense_point = _avg_pp(points, ANCHOR, "dense_rag", published=published)
        if dense_point is not None and "dense_rag" in idx:
            axA.annotate(
                "Dense RAG\nhelps, but still\ntrails Oracle",
                xy=(idx["dense_rag"], dense_point[0] - 1.5),
                xytext=(idx["dense_rag"] + 0.55, 56),
                fontsize=8.2,
                ha="left",
                color="#222",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#666"),
            )

        structured = [(bk, point) for bk, point in (("memobase", memobase_point), ("memos", memos_point)) if point]
        if structured:
            bk, point = max(structured, key=lambda item: item[1][0])
            axA.annotate(
                "Structured memory\napproaches Oracle",
                xy=(idx[bk], point[0] - 3), xytext=(idx[bk] + 0.45, max(10, point[0] - 30)),
                fontsize=8.5, ha="left", color="#222",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#666"),
            )

        axA.set_xticks(xs)
        axA.set_xticklabels([LABEL_A[bk] for bk, _, _ in bars_a], rotation=15, ha="right")
    else:
        axA.text(0.5, 0.5, f"No {MODEL_TEX[ANCHOR]} cells", ha="center", va="center",
                 transform=axA.transAxes, fontsize=10, color="#555")
    axA.set_ylabel("Avg  (pp)")
    axA.set_ylim(0, 100)
    axA.set_title("(a) At Qwen3-8B: retrieval alone leaves the gap open")
    axA.grid(True, axis="y", alpha=0.3)

    # ================== Panel (b): lift-over-Vanilla per tier ==================
    x_tiers = list(range(len(tiers_b)))
    width = 0.26
    offsets = {"rag": -width, "memobase": 0.0, "memos": +width}

    all_ys = []
    for bk in BACKENDS_B:
        xs_bk = []
        lifts = []
        for i, tier in enumerate(tiers_b):
            van = _avg_pp(points, tier, "vanilla", published=published)
            cur = _avg_pp(points, tier, bk, published=published)
            if van is None or cur is None:
                continue
            xs_bk.append(i + offsets[bk])
            lifts.append(cur[0] - van[0])
        all_ys.extend(lifts)
        if not lifts:
            continue
        axB.bar(xs_bk, lifts, width,
                color=COLOR[bk], edgecolor="black", linewidth=0.6,
                label=f"{BACKEND_TEX[bk]} - Vanilla")
        for x, v in zip(xs_bk, lifts):
            axB.text(x, v + 1.0 if v >= 0 else v - 1.6,
                     f"{v:+.1f}", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=6.5, color="#222")

    axB.axhline(0, color="#555", lw=0.7)
    axB.set_xticks(x_tiers)
    axB.set_xticklabels([MODEL_TEX[t] for t in tiers_b], rotation=15, ha="right")
    axB.set_xlabel("Extractor / reader model")
    axB.set_ylabel("Avg lift over Vanilla  (pp)")
    axB.set_title("(b) Memory lift over Vanilla, by reader")
    axB.grid(True, axis="y", alpha=0.3)
    if all_ys:
        axB.legend(loc="upper left", fontsize=8.5, frameon=False)
        low = min(all_ys)
        high = max(all_ys)
        pad = max(5.0, (high - low) * 0.15)
        axB.set_ylim(min(-5, low - pad), max(5, high + pad))
    else:
        axB.text(0.5, 0.5, "No lift cells", ha="center", va="center",
                 transform=axB.transAxes, fontsize=10, color="#555")
        axB.set_ylim(-5, 5)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    points, source_label, published = _load_avg_points()
    if not any(points.get(model) for model in TIERS):
        _delete_outputs()
        print(f"[fig:finding2] skipped: no Avg cells under {DEFAULT_RUN}")
        return 0

    # Stdout table for paper-reference.
    print(f"\n(a) {MODEL_TEX[ANCHOR]} anchor ({source_label}):")
    for bk in _panel_a_backends(points, published=published):
        point = _avg_pp(points, ANCHOR, bk, published=published)
        if point is None:
            continue
        m, s = point
        print(f"  {LABEL_A[bk]:>10} = {m:.1f} +- {s:.1f}")
    print(f"\n(b) Avg lift over Vanilla (pp) per tier ({source_label}):")
    hdr = f"  {'tier':<14}" + " ".join(f"{BACKEND_TEX[bk]+' − V':>14}" for bk in BACKENDS_B)
    print(hdr)
    for t in _panel_b_tiers(points, published=published):
        van_point = _avg_pp(points, t, "vanilla", published=published)
        if van_point is None:
            continue
        van_m, _ = van_point
        row = f"  {MODEL_TEX[t]:<14}"
        for bk in BACKENDS_B:
            bv_point = _avg_pp(points, t, bk, published=published)
            row += f"{bv_point[0] - van_m:>14.1f}" if bv_point else f"{'--':>14}"
        print(row)

    if not _render(OUT_PDF, points, published=published):
        _delete_outputs()
        print("[fig:finding2] skipped: no renderable cells")
        return 0
    print(f"\n[fig:finding2] wrote {OUT_PDF}")
    print(f"[fig:finding2] wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    main()
