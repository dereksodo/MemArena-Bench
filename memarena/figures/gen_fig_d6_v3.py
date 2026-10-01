"""Generate D6 access-control figure (v3, three panels).

Panel (a): Policy-conditioning scatter — x=REFUSAL on ALLOW, y=REFUSAL on DENY.
           Diagonal y=x is the "policy-blind" null line. Reference points:
           always-refuse (1,1), always-allow (0,0), perfect (0,1).
Panel (b): Failure-mode dialectic — x=fact-leak rate on DENY, y=NONE rate on DENY.
           Empty upper-left = "ideal refusal region (no observed cell)".
Panel (c): F1_PU heatmap — 5 backends x 5 readers.
           P = 1 - leak_rate on DENY (fact-based, judge-independent),
           U = COMPLY rate on ALLOW.

Output:
  memarena/figures/figures/fig_d6_v3.pdf
  memarena/figures/figures/fig_d6_v3.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paths import figure_path
from paper_data import load_all_cells, MODEL_ORDER, MODEL_TEX

OUT_PDF = figure_path("fig_d6_v3.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")

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


def _per_cell_stats(grid):
    """Pool s2/s3/s4 for each (backend, reader); return per-cell fractions.

    Uses the new 5-label vocabulary (DISCLOSE_CORRECT / DISCLOSE_WRONG /
    DONT_KNOW / REFUSE / OTHER). For backward-compat with old field names
    used in the figure code:
      ``deny_refusal``  ← REFUSE label on DENY (the rare "true access control")
      ``deny_comply``   ← DISCLOSE_CORRECT on DENY (== leak by definition)
      ``deny_none``     ← DONT_KNOW + OTHER + DISCLOSE_WRONG on DENY
      ``deny_leaked``   ← deterministic leaked_fact_in_output substring match
      same naming pattern for ALLOW
    """
    out = defaultdict(lambda: {
        "deny_total": 0, "deny_refusal": 0, "deny_comply": 0,
        "deny_none": 0, "deny_leaked": 0,
        "allow_total": 0, "allow_refusal": 0, "allow_comply": 0, "allow_none": 0,
    })
    for (seed, backend, model), cell in grid.items():
        if backend not in BACKENDS or model not in MODEL_ORDER:
            continue
        data = json.loads(Path(cell.source_path).read_text())
        for d in data.get("details", []):
            qid = d.get("question_id", "")
            if not qid.startswith("d4"):
                continue
            if d.get("answer_scored") is False:
                continue
            mode = str(d.get("expected_answer_mode") or "").lower()
            cat = str(d.get("policy_category") or "")
            leaked = bool(d.get("leaked_fact_in_output", False))
            key = (backend, model)
            stats = out[key]
            if mode in ("deny", "abstain"):
                stats["deny_total"] += 1
                if cat == "REFUSE":
                    stats["deny_refusal"] += 1
                elif cat == "DISCLOSE_CORRECT":
                    stats["deny_comply"] += 1
                else:  # DONT_KNOW, OTHER, DISCLOSE_WRONG, PARSE_ERROR
                    stats["deny_none"] += 1
                if leaked:
                    stats["deny_leaked"] += 1
            elif mode == "disclose":
                stats["allow_total"] += 1
                if cat == "REFUSE":
                    stats["allow_refusal"] += 1
                elif cat == "DISCLOSE_CORRECT":
                    stats["allow_comply"] += 1
                else:
                    stats["allow_none"] += 1
    return out


def _frac(num, den):
    return num / den if den > 0 else float("nan")


def _f1(p, u):
    if p + u <= 0 or np.isnan(p) or np.isnan(u):
        return 0.0
    return 2 * p * u / (p + u)


def main():
    grid = load_all_cells()
    stats = _per_cell_stats(grid)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})

    # ---------- Panel (a) — Policy-conditioning scatter (zoomed) ----------
    ax = axes[0]
    AX_MAX = 0.05  # 5-label rubric: REFUSE rate is below 4% on every cell
    # diagonal "policy-blind" null line
    ax.plot([0, AX_MAX], [0, AX_MAX], color="black", lw=1.0, ls="--",
            alpha=0.55, label="policy-blind ($y{=}x$)")

    for (backend, reader), s in stats.items():
        x = _frac(s["allow_refusal"], s["allow_total"])
        y = _frac(s["deny_refusal"], s["deny_total"])
        ax.scatter([x], [y], s=70, marker=READER_MARKER[reader],
                   color=BACKEND_COLOR[backend], edgecolor="white", lw=0.6,
                   alpha=0.9, zorder=4)
    ax.set_xlim(-0.005, AX_MAX); ax.set_ylim(-0.005, AX_MAX)
    ax.set_xlabel("REFUSAL rate on ALLOW (over-refusal)")
    ax.set_ylabel("REFUSAL rate on DENY (correct refusal)")
    ax.set_title("(a) Policy-conditioning scatter", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ---------- Panel (b) — Leak vs recall-failure on DENY ----------
    ax = axes[1]
    # ideal refusal region: tight box hugging the corner
    ax.add_patch(plt.Rectangle((0, 0), 0.08, 0.08, fill=True,
                               facecolor="gold", alpha=0.22, zorder=0,
                               edgecolor="#b8860b", lw=0.6))
    ax.text(0.005, 0.072, "ideal refusal\n(empty)",
            fontsize=7, color="#7a5a00", va="top", ha="left",
            fontweight="bold")
    # antidiag x+y=1 — label rotated along the line itself
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, 1 - xs, color="black", lw=0.8, ls=":", alpha=0.45)
    ax.text(0.55, 0.47, "leak $+$ NONE $=1$",
            fontsize=8, color="#444", rotation=-45,
            rotation_mode="anchor", ha="center", va="bottom")

    for (backend, reader), s in stats.items():
        x = _frac(s["deny_leaked"], s["deny_total"])
        y = _frac(s["deny_none"], s["deny_total"])
        ax.scatter([x], [y], s=70, marker=READER_MARKER[reader],
                   color=BACKEND_COLOR[backend], edgecolor="white", lw=0.6,
                   alpha=0.9, zorder=4)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Leak rate on DENY (fact-level)")
    ax.set_ylabel("NONE rate on DENY (recall-failure proxy)")
    ax.set_title("(b) Failure-mode dialectic on DENY", fontsize=11)
    ax.grid(alpha=0.25)

    # ---------- Panel (c) — F1_PU heatmap ----------
    ax = axes[2]
    grid_arr = np.full((len(BACKENDS), len(MODEL_ORDER)), np.nan)
    for i, backend in enumerate(BACKENDS):
        for j, reader in enumerate(MODEL_ORDER):
            s = stats.get((backend, reader))
            if s is None or s["deny_total"] == 0 or s["allow_total"] == 0:
                continue
            p = 1 - _frac(s["deny_leaked"], s["deny_total"])  # fact-based privacy
            u = _frac(s["allow_comply"], s["allow_total"])
            grid_arr[i, j] = _f1(p, u) * 100
    im = ax.imshow(grid_arr, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_TEX[m] for m in MODEL_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(BACKENDS)))
    # Row labels: Memobase gets a star (highlighted as rule-out case in F1 ¶3),
    # Oracle gets a downward arrow indicating its monotone-decreasing trend
    # (highlighted in F1 ¶2 as the privacy-disposition probe).
    yticklabels = []
    for b in BACKENDS:
        lbl = BACKEND_LABEL[b]
        if b == "memobase":
            lbl = lbl + r"$^{\star}$"
        if b == "oracle":
            lbl = lbl + r" $\searrow$"
        yticklabels.append(lbl)
    ax.set_yticklabels(yticklabels)
    for i in range(len(BACKENDS)):
        for j in range(len(MODEL_ORDER)):
            v = grid_arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center",
                        color="white", fontsize=9)
            else:
                color = "white" if v < 55 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=color, fontsize=9)
    # Bold border around Oracle row (the privacy-disposition probe row in F1 ¶2)
    oracle_i = BACKENDS.index("oracle")
    n_cols = len(MODEL_ORDER)
    ax.add_patch(plt.Rectangle((-0.5, oracle_i - 0.5), n_cols, 1,
                               fill=False, edgecolor="black", lw=1.8, zorder=5))
    ax.set_title(r"(c) F1$_\mathrm{PU}$", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(r"F1$_\mathrm{PU}$ (\%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ---------- Shared legend (backends + readers) ----------
    backend_handles = [plt.Line2D([0], [0], marker="o", lw=0, markersize=8,
                                  markerfacecolor=BACKEND_COLOR[b],
                                  markeredgecolor="white", markeredgewidth=0.5,
                                  label=BACKEND_LABEL[b]) for b in BACKENDS]
    reader_handles = [plt.Line2D([0], [0], marker=READER_MARKER[r], lw=0,
                                 markersize=8, color="black",
                                 markerfacecolor="white",
                                 label=MODEL_TEX[r]) for r in MODEL_ORDER]
    fig.legend(handles=backend_handles + reader_handles,
               loc="lower center", ncol=len(BACKENDS) + len(MODEL_ORDER),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")

    # ---------- Print numbers next to the figure for inspection ----------
    print()
    print("=== Per-cell numbers (sorted by F1_PU desc) ===")
    rows = []
    for i, backend in enumerate(BACKENDS):
        for j, reader in enumerate(MODEL_ORDER):
            s = stats.get((backend, reader))
            if s is None or s["deny_total"] == 0:
                continue
            p_fact = 1 - _frac(s["deny_leaked"], s["deny_total"])
            u_cat = _frac(s["allow_comply"], s["allow_total"])
            f1 = _f1(p_fact, u_cat) * 100
            ref_d = _frac(s["deny_refusal"], s["deny_total"])
            ref_a = _frac(s["allow_refusal"], s["allow_total"])
            spec = ref_d / ref_a if ref_a > 0.001 else float("inf")
            rows.append((f1, backend, reader, p_fact, u_cat, ref_d, ref_a, spec,
                         _frac(s["deny_leaked"], s["deny_total"]),
                         _frac(s["deny_none"], s["deny_total"])))
    rows.sort(reverse=True)
    print(f"{'backend':10s} {'reader':8s}  {'F1_PU':>5s}  {'P_fact':>6s}  {'U_cat':>6s}  "
          f"{'refD':>5s}  {'refA':>5s}  {'spec':>6s}  {'leakD':>5s}  {'noneD':>5s}")
    for f1, b, r, pf, uc, rd, ra, sp, ld, nd in rows:
        sp_s = f"{sp:6.2f}" if np.isfinite(sp) else "  inf "
        print(f"{b:10s} {r:8s}  {f1:5.1f}  {pf*100:5.1f}   {uc*100:5.1f}   "
              f"{rd*100:4.1f}  {ra*100:4.1f}  {sp_s}   {ld*100:4.1f}  {nd*100:4.1f}")


if __name__ == "__main__":
    main()
