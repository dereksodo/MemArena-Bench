"""Generate D6 Finding-1 figure.

Thesis: no method does well on permission-aware access control. Two causes:
  Cause 1 — non-Oracle: failed retrieval (privacy by amnesia, not gating).
  Cause 2 — Oracle: model is disclosure-biased; leaks even when given the
            secret AND the access marker in the same context.

Panel (a) — Cause 1 vs Cause 2 quadrant scatter (25 cells).
  x = ALLOW DC rate (third-party probe). Proxy for "did the system surface
      relevant info?" Low x = retrieval failure.
  y = DENY leak rate (third-party probe). Direct measure of "failed to gate".

  Top-left  = ideal (knows but withholds)         — empty in our data.
  Top-right = retrieves AND leaks                  → Cause 2 (Oracle cluster).
  Bot-left  = doesn't retrieve, can't leak         → Cause 1 (non-Oracle).
  Bot-right = leaks without retrieval (impossible).

Panel (b) — Asker-identity contrast on the Oracle row (Cause 2 close-up).
  Per reader, paired bars showing TP leak rate vs EGO leak rate on DENY.
  Both bars high => indiscriminate disclosure (the model leaks regardless of
  who asks). The d4_perm_L_005 case study is highlighted in caption text.

Output:
  memarena/figures/figures/fig_d6_finding1.pdf  (and .png)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval.src.permission_metrics import POPULATION_FACT, is_allow, is_leak, load_instances, population_ids
from memarena.figures.main_grid import default_instances_path, main_grid_stats, scored
from memarena.figures.paper_data import MODEL_ORDER, MODEL_TEX, load_all_cells
from memarena.figures.paths import figure_path

OUT_PDF = figure_path("fig_d6_finding1.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")
F1PU_PDF = figure_path("fig_d6_f1pu_heatmap.pdf")
F1PU_PNG = F1PU_PDF.with_suffix(".png")

REPO_ROOT = Path(__file__).resolve().parents[2]
SELF_PROBE_ROOT = REPO_ROOT / "data" / "d6_self_probe"

BACKENDS = ["vanilla", "rag", "oracle", "memobase", "memsearch"]
BACKEND_LABEL = {
    "vanilla":   "Vanilla",
    "rag":       "RAG",
    "oracle":    "Oracle",
    "memobase":  "Memobase",
    "memsearch": "MemSearch",
}
BACKEND_COLOR = {
    "vanilla":   "#7f7f7f",
    "rag":       "#1f77b4",
    "oracle":    "#2ca02c",
    "memobase":  "#d62728",
    "memsearch": "#ff7f0e",
}
READER_MARKER = {
    "0_6b":    "o",
    "llama3b": "s",
    "7b":      "^",
    "8b":      "D",
    "32b":     "P",
}

SELF_PROBE_TOP = {
    "vanilla":   "vanilla",
    "rag":       "baseline_simplerag",
    "oracle":    "oracle",
    "memobase":  "memobase",
    "memsearch": "memsearch",
}
SEEDS = ["s2", "s3", "s4"]


def _per_cell_tp_stats(population: str = POPULATION_FACT):
    """Pool TP records per (backend, reader) over seeds. Return dict of dict.

    Only the D6 population (default: the 144 fact-bearing items) is counted,
    and a DENY leak is a DISCLOSE_CORRECT label (``eval.src.permission_metrics``).
    """
    keep = population_ids(load_instances(default_instances_path()), population)
    grid = load_all_cells(include_ablation=False)
    out = defaultdict(lambda: {
        "deny_total": 0, "deny_leak": 0,
        "deny_dc": 0, "deny_refuse": 0, "deny_dk": 0,
        "allow_total": 0, "allow_dc": 0,
    })
    for (seed, backend, model), cell in grid.items():
        if backend not in BACKENDS or model not in MODEL_ORDER:
            continue
        if seed not in SEEDS:
            continue
        try:
            data = json.loads(Path(cell.source_path).read_text())
        except Exception:
            continue
        for d in scored(data.get("details", [])):
            qid = str(d.get("question_id", ""))
            if not qid.startswith("d4") or qid not in keep:
                continue
            cat = str(d.get("policy_category") or "").upper()
            leaked = is_leak(d)
            stats = out[(backend, model)]
            if not is_allow(d):
                stats["deny_total"] += 1
                if cat == "DISCLOSE_CORRECT":
                    stats["deny_dc"] += 1
                elif cat == "REFUSE":
                    stats["deny_refuse"] += 1
                elif cat == "DONT_KNOW":
                    stats["deny_dk"] += 1
                if leaked:
                    stats["deny_leak"] += 1
            else:
                stats["allow_total"] += 1
                if cat == "DISCLOSE_CORRECT":
                    stats["allow_dc"] += 1
    return out


def _discover_self_probe_files():
    """Yield (path, backend, reader, seed) for self-probe judged outputs."""
    if not SELF_PROBE_ROOT.exists():
        return
    for be in BACKENDS:
        be_top = SELF_PROBE_TOP[be]
        for reader in MODEL_ORDER:
            top = SELF_PROBE_ROOT / f"{be_top}_{reader}"
            if not top.exists():
                continue
            for seed in SEEDS:
                seed_dir = top / seed / f"eval_results_{seed}"
                if not seed_dir.exists():
                    continue
                cands = list(seed_dir.rglob("evaluation_results_run.json"))
                if cands:
                    yield cands[0], be, reader, seed


def _per_cell_ego_stats():
    out = defaultdict(lambda: {
        "deny_total": 0, "deny_leak": 0, "deny_refuse": 0, "deny_dk": 0,
    })
    for path, backend, reader, _seed in _discover_self_probe_files():
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for d in data.get("details", []):
            qid = str(d.get("question_id", ""))
            if not qid.startswith("d4_perm"):
                continue
            mode = str(d.get("expected_answer_mode") or "").lower()
            if mode not in ("deny", "abstain"):
                continue
            cat = str(d.get("policy_category") or "").upper()
            leaked = bool(d.get("leaked_fact_in_output", False))
            s = out[(backend, reader)]
            s["deny_total"] += 1
            if leaked:
                s["deny_leak"] += 1
            if cat == "REFUSE":
                s["deny_refuse"] += 1
            elif cat == "DONT_KNOW":
                s["deny_dk"] += 1
    return out


def _frac(num, den):
    return num / den if den > 0 else float("nan")


def _draw_panel_a(ax, tp, compact: bool = False):
    """Panel (a): Cause-1 vs Cause-2 quadrant scatter.

    ``compact`` draws it for a figure made at its printed size (Fig. 2 of the
    paper, ~2.3 in wide): 7 pt text, wrapped region labels, smaller markers.
    The default keeps the layout of the standalone 6.7 in figure.
    """
    AX_MAX = 1.0
    THRESH = 0.30  # quadrant boundary
    if compact:
        fs = 7.0
        label_cause2 = dict(x=0.32, y=0.985, s="Cause 2: surfaces\nAND leaks\n(Oracle cluster)", ha="left")
        label_cause1 = dict(x=0.15, y=0.53, s="Cause 1:\nfact not\nsurfaced", ha="center")
        label_ideal = dict(x=0.56, y=0.275, s="ideal: surfaces\nAND gates\n(EMPTY)", ha="center")
        diag = dict(x=0.47, y=0.43, s=r"$y=x$ (policy-blind)", fontsize=fs, rotation=45,
                    transform_rotates_text=True)
        star_note = dict(xytext=(0.885, 0.125), fontsize=fs)
        pt_size, star_size, line_lw = 22, 55, 0.6
    else:
        fs = None
        label_cause2 = dict(x=0.65, y=0.93, s="Cause 2: surfaces AND leaks\n(Oracle cluster)", ha="center", fontsize=12)
        label_cause1 = dict(x=0.15, y=0.36, s="Cause 1:\nfact not surfaced", ha="center", fontsize=11)
        label_ideal = dict(x=0.65, y=0.27, s="ideal: surfaces AND gates\n(EMPTY)", ha="center", fontsize=12)
        diag = dict(x=0.55, y=0.50, s=r"$y=x$ (policy-blind)", fontsize=11, rotation=38)
        star_note = dict(xytext=(0.85, 0.10), fontsize=11)
        pt_size, star_size, line_lw = 90, 200, 0.8
    region_fs = {} if fs is None else {"fontsize": fs}

    # Quadrant shading. Axes: x = ALLOW DC (retrieval success), y = DENY leak.
    # Ideal corner is BOTTOM-RIGHT (retrieves AND gates), not top-left.
    cause2_box = plt.Rectangle((THRESH, THRESH), AX_MAX - THRESH, AX_MAX - THRESH,
                               facecolor="#ffe6e6", edgecolor="none", alpha=0.5, zorder=0)
    cause1_box = plt.Rectangle((0, 0), THRESH, THRESH,
                               facecolor="#e6f0ff", edgecolor="none", alpha=0.5, zorder=0)
    ideal_box = plt.Rectangle((THRESH, 0), AX_MAX - THRESH, THRESH,
                              facecolor="#fff4d6", edgecolor="none", alpha=0.5, zorder=0)
    ax.add_patch(cause2_box)
    ax.add_patch(cause1_box)
    ax.add_patch(ideal_box)
    ax.text(**label_cause2, **region_fs, color="#9b1c1c", va="top", fontweight="bold")
    ax.text(**label_cause1, **region_fs, color="#1c4e9b", va="top", fontweight="bold")
    ax.text(**label_ideal, **region_fs, color="#7a5a00", va="top", fontweight="bold")

    # Diagonal y=x (policy-blind reference)
    diag_xy = np.linspace(0, AX_MAX, 40)
    ax.plot(diag_xy, diag_xy, color="black", lw=line_lw, ls=":", alpha=0.55)
    ax.text(**diag, color="#444", rotation_mode="anchor", ha="left", va="top")

    # Plot 25 cells
    for backend in BACKENDS:
        for reader in MODEL_ORDER:
            s_tp = tp.get((backend, reader))
            if not s_tp or s_tp["allow_total"] == 0 or s_tp["deny_total"] == 0:
                continue
            x = _frac(s_tp["allow_dc"], s_tp["allow_total"])
            y = _frac(s_tp["deny_leak"], s_tp["deny_total"])
            ax.scatter([x], [y], s=pt_size, marker=READER_MARKER[reader],
                       color=BACKEND_COLOR[backend], edgecolor="white",
                       lw=0.4 if compact else 0.7, alpha=0.95, zorder=4)

    # Star at ideal (1, 0) — high retrieval, zero leak
    ax.scatter([1.0], [0.0], marker="*", s=star_size, color="#d4af37",
               edgecolor="black", lw=0.5 if compact else 0.8, zorder=5)
    ax.annotate("ideal\n(1, 0)", xy=(1.0, 0.0), **star_note,
                color="#7a5a00", ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#7a5a00"))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Correct-disclosure rate on ALLOW (utility $\rightarrow$)",
                  fontsize=fs or 12, labelpad=2 if compact else None)
    # The compact panel is too short for the y label on one line.
    ax.set_ylabel(("Fact-leak rate on DENY\n" if compact else "Fact-leak rate on DENY ")
                  + r"(failed to gate $\rightarrow$)",
                  fontsize=fs or 12, labelpad=2 if compact else None)
    if compact:
        ax.tick_params(labelsize=fs, length=2, width=0.5, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    else:
        ax.tick_params(labelsize=11)
    ax.grid(alpha=0.2)


HEATMAP_FS = 7.0  # the heatmap is drawn at its printed size (0.7\textwidth)


def _draw_f1pu_heatmap(ax, stats):
    """Standalone F1_PU heatmap: seed-mean D6 per cell, identical to the D6
    column of tab:results-L-full (``memarena.figures.main_grid``)."""
    fs = HEATMAP_FS
    grid_arr = np.full((len(BACKENDS), len(MODEL_ORDER)), np.nan)
    for i, backend in enumerate(BACKENDS):
        for j, reader in enumerate(MODEL_ORDER):
            grid_arr[i, j] = stats[(backend, reader)]["D6"].mean
    im = ax.imshow(grid_arr, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_TEX[m] for m in MODEL_ORDER], rotation=30, ha="right",
                       rotation_mode="anchor", fontsize=fs)
    ax.set_yticks(range(len(BACKENDS)))
    ax.set_yticklabels([BACKEND_LABEL[b] for b in BACKENDS], fontsize=fs)
    ax.tick_params(length=2, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    for i in range(len(BACKENDS)):
        for j in range(len(MODEL_ORDER)):
            v = grid_arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", color="white", fontsize=fs + 0.5)
            else:
                color = "white" if v < 55 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", color=color, fontsize=fs + 0.5)
    oracle_i = BACKENDS.index("oracle")
    n_cols = len(MODEL_ORDER)
    ax.add_patch(plt.Rectangle((-0.5, oracle_i - 0.5), n_cols, 1,
                               fill=False, edgecolor="black", lw=1.2, zorder=5))
    ax.set_title(r"$\mathrm{F1}_\mathrm{PU}$ per cell  (Oracle row boxed)",
                 fontsize=8, pad=3)
    return im


def main():
    tp = _per_cell_tp_stats()
    ego = _per_cell_ego_stats()

    # ====================================================================
    # Standalone panel (a) — for main text wrapfigure
    # ====================================================================
    fig_a, ax_a = plt.subplots(figsize=(6.7, 4.0))
    _draw_panel_a(ax_a, tp)
    backend_handles = [plt.Line2D([0], [0], marker="o", lw=0, markersize=8,
                                  markerfacecolor=BACKEND_COLOR[b],
                                  markeredgecolor="white", markeredgewidth=0.5,
                                  label=BACKEND_LABEL[b]) for b in BACKENDS]
    reader_handles = [plt.Line2D([0], [0], marker=READER_MARKER[r], lw=0,
                                 markersize=8, color="black",
                                 markerfacecolor="white",
                                 label=MODEL_TEX[r]) for r in MODEL_ORDER]
    # Side legends — Backend on the left, Reader on the right (outside axes)
    leg_b = ax_a.legend(handles=backend_handles, loc="center right",
                        bbox_to_anchor=(-0.20, 0.5), fontsize=10,
                        frameon=False, title="Backend", title_fontsize=10)
    ax_a.add_artist(leg_b)
    leg_r = ax_a.legend(handles=reader_handles, loc="center left",
                        bbox_to_anchor=(1.03, 0.5), fontsize=10,
                        frameon=False, title="Reader", title_fontsize=10)
    fig_a.tight_layout(rect=(0.14, 0, 0.88, 1))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig_a.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.15,
                  bbox_extra_artists=[leg_b, leg_r])
    fig_a.savefig(OUT_PNG, bbox_inches="tight", dpi=160, pad_inches=0.15,
                  bbox_extra_artists=[leg_b, leg_r])
    plt.close(fig_a)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")

    # ====================================================================
    # Standalone F1_PU heatmap — for appendix
    # ====================================================================
    # Fixed canvas at the printed size (0.7\textwidth = 3.85 in, 2.55 in tall),
    # so the 7 pt labels print at 7 pt.
    hm_w, hm_h = 3.85, 2.55
    fig_c = plt.figure(figsize=(hm_w, hm_h))
    ax_c = fig_c.add_axes([0.64 / hm_w, 0.47 / hm_h, 2.64 / hm_w, 1.86 / hm_h])
    cax = fig_c.add_axes([3.37 / hm_w, 0.47 / hm_h, 0.11 / hm_w, 1.86 / hm_h])
    im = _draw_f1pu_heatmap(ax_c, main_grid_stats())
    cbar = fig_c.colorbar(im, cax=cax)
    cbar.set_label(r"$\mathrm{F1}_\mathrm{PU}$ (%)", fontsize=HEATMAP_FS, labelpad=2)
    cbar.ax.tick_params(labelsize=HEATMAP_FS, length=2, width=0.5, pad=1.5)
    cbar.outline.set_linewidth(0.6)
    fig_c.savefig(F1PU_PDF)
    fig_c.savefig(F1PU_PNG, dpi=300)
    plt.close(fig_c)
    print(f"Wrote {F1PU_PDF}")
    print(f"Wrote {F1PU_PNG}")

    # Print numbers for inspection (panel a)
    print()
    print("=== Panel (a) coordinates per cell ===")
    print(f"  {'cell':<22s} {'ALLOW_DC%':>9s} {'DENY_leak%':>10s}")
    for backend in BACKENDS:
        for reader in MODEL_ORDER:
            s = tp.get((backend, reader))
            if not s or s["allow_total"] == 0 or s["deny_total"] == 0:
                continue
            x = s["allow_dc"] / s["allow_total"] * 100
            y = s["deny_leak"] / s["deny_total"] * 100
            print(f"  {backend}/{reader:<10s}  {x:8.1f}  {y:9.1f}")
    return


if __name__ == "__main__":
    main()
