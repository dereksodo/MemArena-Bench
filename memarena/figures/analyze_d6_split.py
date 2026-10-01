#!/usr/bin/env python3
"""Emit a TikZ fragment for the D6 refusal-rate split figure.

Writes ``paper/figures/fig_d6_split.tex`` as a ``\\begin{figure}`` block that
contains a raw ``tikzpicture`` (no ``\\includegraphics``, no external PDF).
Two side-by-side panels:

* (a) **Social** context refusal rate — 159 refuse-expected instances
  (permission_compliance refuse-mode + known_requester + anonymous_querier).
* (b) **Autonomous** privacy refusal rate — 41 refuse-expected instances
  (autonomous_privacy).

Each bar is split into a solid lower block (actual privacy-motivated
refusals) and a 55%-alpha upper block stacked on top (info_absent refusals
— the retrieval-miss artefact that makes naive policy_compliant look too
good). Error bars sit on the *total* refusal rate (compliant total),
matching the number we reported before the rationale split.

The y-axis runs from 0 to 1 in data space, but the upper segment
``[0.5, 1]`` is visually compressed 10× (Python applies the piecewise
linear transform so the TikZ source carries raw cm coordinates — no
``\\pgfmathparse`` acrobatics). A zigzag break marker on the left spine
tells readers the axis is folded.

Main.tex picks this file up via ``\\input{figures/fig_d6_split}`` in §6.3.
Compilation requires ``\\usepackage{tikz}`` in the preamble (no pgfplots
needed; the whole figure is manual TikZ).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import policy_compliant_from_armB  # noqa: E402
from paper_data import (  # noqa: E402
    BACKEND_TEX, MODEL_ORDER, MODEL_TEX,
    SEEDS, DEFAULT_RUN,
    load_all_cells,
)
from d6_refusal_classifier import classify  # noqa: E402
from paths import figure_path  # noqa: E402

OUT_TEX = figure_path("fig_d6_split.tex")

# Backends in display order; matches the main table.
PLOT_BACKENDS = ["vanilla", "rag", "oracle", "memobase", "memos"]

# Per-backend colour — matches the palette used elsewhere in the paper.
# TikZ colour names are defined once in the preamble block below.
BACKEND_COLORS_RGB = {
    "vanilla":  (127, 127, 127),  # #7f7f7f
    "rag":      ( 31, 119, 180),  # #1f77b4
    "oracle":   ( 44, 160,  44),  # #2ca02c
    "memobase": (214,  39,  40),  # #d62728
    "memos":    (255, 127,  14),  # #ff7f0e
}

PANELS = [
    ("social_refuse",     "(I) Social"),
    ("autonomous_refuse", "(II) Autonomous"),
]

SOCIAL_FEATURES = {"permission_compliance", "known_requester", "anonymous_querier"}
AUTONOMOUS_FEATURES = {"autonomous_privacy"}


# ---------------------------------------------------------------------------
# Data loading — identical to the previous matplotlib version.
# ---------------------------------------------------------------------------

# Hardcoded location of the D4 permission ground-truth file. This ships as
# part of the MemArena-L Hugging Face dataset (dereksodo/memarena-l)
# and lands at data/benchmark/eval_instances/ after running
# scripts/download_dataset.py.
D4_PERMISSION_JSONL = (
    Path(__file__).resolve().parents[2]
    / "data" / "benchmark" / "eval_instances" / "d4_permission.jsonl"
)


def _load_qid_taxonomy(run_dir: Path) -> dict[str, str]:
    """Map instance_id -> panel_id."""
    del run_dir  # superseded by the hardcoded D4_PERMISSION_JSONL constant
    out: dict[str, str] = {}
    with open(D4_PERMISSION_JSONL) as f:
        for line in f:
            row = json.loads(line)
            feat = (row.get("metadata") or {}).get("question_feature")
            mode = (row.get("ground_truth") or {}).get("expected_answer_mode")
            is_refuse = mode in ("deny", "abstain")
            if feat in SOCIAL_FEATURES and is_refuse:
                out[row["instance_id"]] = "social_refuse"
            elif feat in AUTONOMOUS_FEATURES and is_refuse:
                out[row["instance_id"]] = "autonomous_refuse"
    return out


def _cell_panel_rates(json_path: Path, qid_to_panel) -> dict[str, tuple[float | None, float | None]]:
    """For one eval JSON, return {panel_id: (compliant_rate, actual_privacy_rate)}."""
    try:
        data = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {p[0]: (None, None) for p in PANELS}
    tallies = defaultdict(lambda: [0, 0, 0])  # [compliant_c, total_n, actual_privacy_c]
    for r in data.get("details", []):
        qid = r.get("question_id") or ""
        if not qid.startswith("d4_"):
            continue
        panel = qid_to_panel.get(qid)
        if panel is None:
            continue
        tallies[panel][1] += 1
        if not policy_compliant_from_armB(r):
            continue
        tallies[panel][0] += 1
        if classify(r.get("prediction", "")) != "info_absent":
            tallies[panel][2] += 1
    out = {}
    for panel_id, _ in PANELS:
        c, n, a = tallies[panel_id]
        out[panel_id] = (c / n, a / n) if n else (None, None)
    return out


def _aggregate(grid, qid_to_panel, backend, model, panel_id):
    """Return (compliant_mean, compliant_std, actual_mean, actual_std, n)."""
    cs_, as_, ns = [], [], []
    for seed in SEEDS:
        cell = grid.get((seed, backend, model))
        if cell is None:
            continue
        cr, ar = _cell_panel_rates(cell.source_path, qid_to_panel).get(panel_id, (None, None))
        if cr is None:
            continue
        cs_.append(cr)
        as_.append(ar if ar is not None else cr)
    if not cs_:
        return None, None, None, None, 0
    cm = statistics.mean(cs_)
    cstd = statistics.stdev(cs_) if len(cs_) > 1 else 0.0
    am = statistics.mean(as_)
    astd = statistics.stdev(as_) if len(as_) > 1 else 0.0
    return cm, cstd, am, astd, len(cs_)


# ---------------------------------------------------------------------------
# TikZ emission
# ---------------------------------------------------------------------------

# Panel geometry (all in cm). A panel is a plot area 7 wide × 5 tall, with
# the top 0.5 cm reserved for the compressed [0.5, 1.0] band.
PANEL_W = 7.0
PANEL_H = 5.0
# Where data y = 0.5 lives in drawn cm (so [0, 0.5] gets 4.5 cm and
# [0.5, 1.0] gets 0.5 cm — a 10× visual compression).
Y_BREAK_CM = 4.5
# Vertical gap between stacked panels (panel (a) on top, panel (b) below).
# Must accommodate panel (a)'s rotated x-tick labels (hanging ~1 cm below
# the axis at 20° rotation for names up to "Mistral-7B") AND panel (b)'s
# title (sitting ~0.1 cm above its top edge). 2.5 cm leaves a comfortable
# margin.
PANEL_SEP = 2.5

BAR_W = 0.15             # cm — width of a single bar
SLOT_W = PANEL_W / 5     # each of 5 models gets an equal slot
# Inside a slot, 5 backends sit side by side with this offset from slot centre.
BACKEND_OFFSETS = [(i - 2) * (BAR_W + 0.02) for i in range(5)]


def _y_cm(y_data: float) -> float:
    """Map data-y ∈ [0, 1] to drawn cm ∈ [0, PANEL_H] via the piecewise
    linear broken-axis transform."""
    if y_data <= 0.5:
        return y_data * (Y_BREAK_CM / 0.5)       # [0, 0.5] → [0, 4.5]
    return Y_BREAK_CM + (y_data - 0.5) * ((PANEL_H - Y_BREAK_CM) / 0.5)  # [0.5,1] → [4.5,5]


def _emit_panel(panel_id: str, title: str, grid, qid_to_panel,
                 x_shift: float, y_shift: float, is_first: bool) -> list[str]:
    """Return a list of TikZ lines that render one panel at the given offsets.

    ``is_first`` is True for the panel that owns the shared y-axis label and
    legend (top panel in the vertically-stacked layout, leftmost panel in
    the historical horizontal layout).
    """
    L: list[str] = []
    L.append(f"% ---- Panel {title} ----")
    L.append(f"\\begin{{scope}}[xshift={x_shift:.3f}cm, yshift={y_shift:.3f}cm]")

    # Axis frame (bottom + left spine).
    L.append(f"\\draw[thick] (0,0) -- ({PANEL_W:.3f},0);")
    L.append(f"\\draw[thick] (0,0) -- (0,{PANEL_H:.3f});")

    # y-ticks at 0, 0.1, 0.2, 0.3, 0.4, 0.5, and 1.0.
    for y_data in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
        y = _y_cm(y_data)
        L.append(f"\\draw[thick] (-0.08,{y:.3f}) -- (0,{y:.3f});")
        label = f"{y_data:.1f}" if y_data != 0 else "0"
        L.append(f"\\node[anchor=east, font=\\small] at (-0.12,{y:.3f}) {{{label}}};")

    # y-label on the left panel only (a standalone extraction can re-enable).
    if is_first:
        L.append(
            f"\\node[rotate=90, anchor=south, font=\\small] at "
            f"(-0.95,{PANEL_H / 2:.3f}) {{Refusal rate}};"
        )

    # Zigzag axis-break symbol on the spine, spanning ~0.1 cm vertically
    # right at the y=0.5 break.  Matches the TikZ idiom the user showed:
    # four points forming `/\/\` along the spine.
    z_cx = 0.0
    z_cy = Y_BREAK_CM  # at the break
    dx = 0.10
    L.append(
        f"\\draw[black, thick] "
        f"({z_cx:.3f},{z_cy + 0.12:.3f}) -- ({z_cx - dx:.3f},{z_cy + 0.06:.3f}) -- "
        f"({z_cx:.3f},{z_cy - 0.06:.3f}) -- ({z_cx - dx:.3f},{z_cy - 0.12:.3f});"
    )

    # x-tick labels — one per model, centred in its slot.
    for mi, model in enumerate(MODEL_ORDER):
        slot_cx = (mi + 0.5) * SLOT_W
        L.append(
            f"\\node[anchor=north east, rotate=20, font=\\small] at "
            f"({slot_cx:.3f},-0.08) {{{MODEL_TEX[model]}}};"
        )

    # Panel title.
    L.append(
        f"\\node[font=\\bfseries, anchor=south] at "
        f"({PANEL_W / 2:.3f},{PANEL_H + 0.10:.3f}) {{{title}}};"
    )

    # Bars.
    for mi, model in enumerate(MODEL_ORDER):
        slot_cx = (mi + 0.5) * SLOT_W
        for bi, backend in enumerate(PLOT_BACKENDS):
            cm_mean, cstd, am_mean, astd, n = _aggregate(
                grid, qid_to_panel, backend, model, panel_id,
            )
            if n == 0 or cm_mean is None:
                continue  # no bar for missing cells
            bar_cx = slot_cx + BACKEND_OFFSETS[bi]
            x0 = bar_cx - BAR_W / 2
            x1 = bar_cx + BAR_W / 2

            color = f"bk{backend}"
            # Solid lower block — actual privacy-motivated refusals.
            y_actual = _y_cm(am_mean)
            if y_actual > 0:
                L.append(
                    f"\\fill[{color}, draw=black, line width=0.2pt] "
                    f"({x0:.3f},0) rectangle ({x1:.3f},{y_actual:.3f});"
                )
            # Translucent upper block — info_absent portion on top.
            y_total = _y_cm(cm_mean)
            if cm_mean - am_mean > 1e-6:
                L.append(
                    f"\\fill[{color}, opacity=0.20, draw=black, "
                    f"draw opacity=0.55, line width=0.2pt] "
                    f"({x0:.3f},{y_actual:.3f}) rectangle ({x1:.3f},{y_total:.3f});"
                )
            # Error bar on the total (compliant) rate.  Drop the bar if the
            # compressed region would smear the whiskers into the zigzag.
            if cstd > 0:
                y_lo = _y_cm(max(cm_mean - cstd, 0.0))
                y_hi = _y_cm(min(cm_mean + cstd, 1.0))
                L.append(
                    f"\\draw[black, line width=0.3pt] "
                    f"({bar_cx:.3f},{y_lo:.3f}) -- ({bar_cx:.3f},{y_hi:.3f});"
                )
                L.append(
                    f"\\draw[black, line width=0.3pt] "
                    f"({bar_cx - 0.05:.3f},{y_lo:.3f}) -- "
                    f"({bar_cx + 0.05:.3f},{y_lo:.3f});"
                )
                L.append(
                    f"\\draw[black, line width=0.3pt] "
                    f"({bar_cx - 0.05:.3f},{y_hi:.3f}) -- "
                    f"({bar_cx + 0.05:.3f},{y_hi:.3f});"
                )

    # Legend — only on panel (a), upper-centre, 3 columns.
    if is_first:
        lx = PANEL_W / 2 - 2.2   # legend block starts 2.2cm left of panel centre
        ly = PANEL_H - 0.55      # inside the compressed band
        cell_w = 1.5             # per-entry horizontal width
        cell_h = 0.35             # per-entry vertical height
        # Row 2 has only {Memobase, MemOS}; push MemOS rightward to clear the
        # longer "Memobase" label.
        ROW2_MEMOS_EXTRA = 0.6
        for bi, backend in enumerate(PLOT_BACKENDS):
            col = bi % 3
            row = bi // 3
            cx = lx + col * cell_w
            if row == 1 and backend == "memos":
                cx += ROW2_MEMOS_EXTRA
            cy = ly - row * cell_h
            L.append(
                f"\\fill[bk{backend}] ({cx:.3f},{cy - 0.08:.3f}) "
                f"rectangle ({cx + 0.22:.3f},{cy + 0.08:.3f});"
            )
            L.append(
                f"\\node[anchor=west, font=\\footnotesize] at "
                f"({cx + 0.28:.3f},{cy:.3f}) {{{BACKEND_TEX[backend]}}};"
            )

    L.append("\\end{scope}")
    return L


def _emit_color_defs() -> list[str]:
    """Emit ``\\definecolor`` lines so the panel code can refer to ``bkvanilla``
    etc. symbolically."""
    lines = []
    for backend, (r, g, b) in BACKEND_COLORS_RGB.items():
        lines.append(f"\\definecolor{{bk{backend}}}{{RGB}}{{{r},{g},{b}}}")
    return lines


def build_tikz(grid, qid_to_panel) -> str:
    """Return a raw ``tikzpicture`` (no figure/caption wrapper).

    The panels are stacked vertically — panel (a) Social on top, panel (b)
    Autonomous directly below — so the whole picture has a narrow aspect
    ratio and can sit beside Figure~3 inside a shared figure block.
    The caller (``paper/main.tex``) owns the ``\\begin{figure}`` block, the
    ``\\captionof{figure}{...}`` line, and the ``\\label{fig:d6-split}``.
    """
    body = []
    body.append("% Auto-generated by analyze_d6_split.py. Do not hand-edit.")
    body.append("% Regenerate via: python paper/analyze_d6_split.py")
    body.extend(_emit_color_defs())
    body.append("\\begin{tikzpicture}[x=1cm, y=1cm]")
    n = len(PANELS)
    for i, (panel_id, title) in enumerate(PANELS):
        # Panel (a) (i=0) goes on top; panel (b) (i=1) below it.
        y_shift = (n - 1 - i) * (PANEL_H + PANEL_SEP)
        body.extend(_emit_panel(panel_id, title, grid, qid_to_panel,
                                 x_shift=0.0, y_shift=y_shift, is_first=(i == 0)))
    body.append("\\end{tikzpicture}")
    return "\n".join(body) + "\n"


def main() -> None:
    if not D4_PERMISSION_JSONL.exists():
        if OUT_TEX.exists():
            OUT_TEX.unlink()
        print(
            f"[fig:d6-split] skipped: D4 permission ground-truth not found: {D4_PERMISSION_JSONL}. "
            "Run scripts/download_dataset.py to fetch the MemArena-L HF dataset."
        )
        return
    grid = load_all_cells(DEFAULT_RUN)
    qid_to_panel = _load_qid_taxonomy(DEFAULT_RUN)
    counts = defaultdict(int)
    for v in qid_to_panel.values():
        counts[v] += 1
    print("[fig:d6-split] instance counts by panel:")
    for panel_id, title in PANELS:
        print(f"    {panel_id:22s} {counts[panel_id]}")
    tex = build_tikz(grid, qid_to_panel)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"[fig:d6-split] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
