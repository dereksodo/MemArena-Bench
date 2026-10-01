"""D6 DENY-response composition per cell: Table 6 body and stacked-bar figure.

For each (backend, reader) cell pooled across seeds {s2,s3,s4}, every DENY item
of the D6 population (default: the 144 fact-bearing items) falls in exactly one
segment of the 5-label rubric used for scoring (``eval.src.permission_metrics``):
  Leak       DISCLOSE_CORRECT              (red)
  REFUSE     explicit access-policy refusal (green)
  DONT_KNOW  epistemic absence              (light grey)
  OTHER      OTHER + DISCLOSE_WRONG         (dark grey)
R-share is REFUSE as a share of the non-leak responses.

Outputs (artifact root, see ``memarena.figures.paths``):
  figures/fig_d6_ablation_buckets.pdf (and .png)
  tables/d6_ablation_buckets_body.tex
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval.src.permission_metrics import POPULATION_FACT, is_allow, load_instances, population_ids
from memarena.figures.main_grid import default_instances_path, scored
from memarena.figures.paper_data import MODEL_ORDER, MODEL_TEX, load_all_cells
from memarena.figures.paths import figure_path, table_path

OUT_PDF = figure_path("fig_d6_ablation_buckets.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")
OUT_TABLE = table_path("d6_ablation_buckets_body.tex")
TABLE_BACKENDS = ["vanilla", "rag", "oracle", "memobase", "memsearch"]
SEGMENTS = ["Leak", "REFUSE", "DONT_KNOW", "OTHER"]
SEGMENT_LABEL = {"Leak": "Leak (fact disclosed)", "REFUSE": "Refuse (access policy)", "DONT_KNOW": "Don't know", "OTHER": "Other / wrong fact"}

BACKENDS = ["vanilla", "rag", "oracle", "memobase", "memsearch"]
BACKEND_LABEL = {
    "vanilla":   "Vanilla",
    "rag":       "RAG",
    "oracle":    "Oracle",
    "memobase":  "Memobase",
    "memsearch": "MemSearch",
}

SEG_COLOR = {
    "Leak":      "#d7301f",
    "REFUSE":    "#1a9850",
    "DONT_KNOW": "#d0d0d0",
    "OTHER":     "#8c8c8c",
}


def _per_cell_buckets(population: str = POPULATION_FACT):
    keep = population_ids(load_instances(default_instances_path()), population)
    grid = load_all_cells(include_ablation=False)
    out = defaultdict(lambda: {"deny_total": 0, **{seg: 0 for seg in SEGMENTS}})
    for (seed, backend, reader), cell in grid.items():
        if backend not in BACKENDS or reader not in MODEL_ORDER:
            continue
        data = json.loads(Path(cell.source_path).read_text())
        s = out[(backend, reader)]
        for d in scored(data.get("details", [])):
            qid = str(d.get("question_id", ""))
            if not qid.startswith("d4") or qid not in keep or is_allow(d):
                continue
            s["deny_total"] += 1
            cat = str(d.get("policy_category") or "").upper()
            if cat == "DISCLOSE_CORRECT":
                s["Leak"] += 1
            elif cat in ("REFUSE", "DONT_KNOW"):
                s[cat] += 1
            else:
                s["OTHER"] += 1
    return out


def _row(s: dict) -> str:
    n = s["deny_total"]
    pct = [100.0 * s[seg] / n for seg in SEGMENTS]
    nonleak = n - s["Leak"]
    share = 100.0 * s["REFUSE"] / nonleak if nonleak else 0.0
    return " & ".join(f"{v:.1f}" for v in pct + [share])


def write_table(stats) -> None:
    lines = []
    for bi, b in enumerate(TABLE_BACKENDS):
        for ri, r in enumerate(MODEL_ORDER):
            head = f"{BACKEND_LABEL[b]:<10s}" if ri == 0 else " " * 10
            lines.append(f"{head} & {MODEL_TEX[r]:<13s} & {_row(stats[(b, r)])} \\\\")
        lines.append(r"\midrule")
    pooled = {"deny_total": 0, **{seg: 0 for seg in SEGMENTS}}
    for b in TABLE_BACKENDS:
        for r in MODEL_ORDER:
            for k in pooled:
                pooled[k] += stats[(b, r)][k]
    lines.append(r"\textbf{Pooled} & --- & " + _row(pooled) + r" \\")
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_TABLE}")


READER_SHORT = {"0_6b": "0.6B", "llama3b": "3B", "7b": "7B", "8b": "8B", "32b": "32B"}


def main():
    stats = _per_cell_buckets()
    write_table(stats)

    # Main-text figure: full text width (5.5 in) and the 1.93 in height the
    # page reserves, drawn at print size on a fixed canvas (no tight bbox), so
    # \includegraphics[width=\textwidth] prints every font at its nominal size.
    FS = 7.0
    fig_w, fig_h = 5.5, 1.93
    plt.rcParams.update({"font.size": FS, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_l, ax_r, ax_b, ax_t = 0.37, 5.47, 0.40, 1.66   # inches
    ax = fig.add_axes([ax_l / fig_w, ax_b / fig_h, (ax_r - ax_l) / fig_w, (ax_t - ax_b) / fig_h])

    bar_w, group_gap = 0.86, 0.7
    n_readers = len(MODEL_ORDER)
    xs, cell_keys, centers = [], [], []
    for bi, b in enumerate(BACKENDS):
        x0 = bi * (n_readers + group_gap)
        for ri, r in enumerate(MODEL_ORDER):
            xs.append(x0 + ri)
            cell_keys.append((b, r))
        centers.append(x0 + (n_readers - 1) / 2)
    xs = np.array(xs, float)

    pct = {seg: np.zeros(len(xs)) for seg in SEGMENTS}
    for i, key in enumerate(cell_keys):
        s = stats.get(key)
        if s and s["deny_total"]:
            for seg in SEGMENTS:
                pct[seg][i] = 100.0 * s[seg] / s["deny_total"]

    # the Oracle group (the only one that surfaces the fact) on a light band
    oi = BACKENDS.index("oracle")
    ax.axvspan(centers[oi] - n_readers / 2 - 0.05, centers[oi] + n_readers / 2 + 0.05,
               color="#fbe9e7", zorder=0, lw=0)

    bottom = np.zeros(len(xs))
    for seg in SEGMENTS:
        ax.bar(xs, pct[seg], width=bar_w, bottom=bottom, color=SEG_COLOR[seg],
               edgecolor="white", lw=0.3, label=SEGMENT_LABEL[seg], zorder=2)
        bottom += pct[seg]

    for i in range(len(xs)):
        lk = pct["Leak"][i]
        if lk >= 20:        # leak share printed inside the red segment
            ax.text(xs[i], lk / 2, f"{lk:.0f}", ha="center", va="center",
                    fontsize=FS, color="white", fontweight="bold", zorder=3)
        rf = pct["REFUSE"][i]
        if rf >= 3:         # the rare explicit refusals
            ax.annotate(f"{rf:.0f}", (xs[i], 100), xytext=(0, 1), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FS,
                        color=SEG_COLOR["REFUSE"], fontweight="bold")

    for bi, b in enumerate(BACKENDS):   # above the (rare) refusal counts
        ax.annotate(BACKEND_LABEL[b], (centers[bi], 100), xytext=(0, 9.5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color="#b71c1c" if b == "oracle" else "#222")

    # Reader labels at 7 pt are wider than a bar slot for "0.6B"; the tick marks
    # are hidden, so the first and last label of each group sit slightly outward
    # (into the gap between groups) to keep a clear space between labels.
    nudge = {0: -0.17, n_readers - 1: 0.12}
    ax.set_xticks([x + nudge.get(i % n_readers, 0.0) for i, x in enumerate(xs)])
    ax.set_xticklabels([READER_SHORT.get(r, r) for (_, r) in cell_keys], fontsize=FS)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.tick_params(axis="y", length=2, width=0.5, labelsize=FS, pad=1.5)
    ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("% of DENY items", fontsize=FS, labelpad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=((ax_l + ax_r) / 2 / fig_w, 0.0), ncol=4, fontsize=FS,
               frameon=False, handlelength=1.1, handleheight=0.8, columnspacing=1.4,
               borderaxespad=0.2)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
