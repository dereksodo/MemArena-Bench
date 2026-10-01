#!/usr/bin/env python3
"""Regenerate the paper's main and appendix results tables.

Emits three LaTeX table fragments under ``paper/tables/``:

* ``main_SML.tex``             — Table 3 ``tab:results-L``.  Category-level
                               (Rec / Rea / Conf / Avg), one column block per
                               model, one row per backend, mean ± std across
                               stochastic seeds.
* ``appendix_L.tex``          — Table 7 ``tab:results-L-full``.  Per paper-
                               dimension (D1..D7) accuracy, again as
                               mean ± std.
* ``appendix_L_subdim.tex``   — Table 8 ``tab:results-L-subdim``. Per
                               sub-dimension accuracy (d1..d11), one row per
                               backend × model.

All three read from the canonical run at
``MASim/runs/l_20260408_111046/`` via :mod:`paper.paper_data`, which handles
the directory layout, the schema differences between s1 and s2/s3/s4, and the
structured-memcache filename parsing.

## Why all three in one script

They share the same data load, the same column-max bolding logic, and the same
cell-missing fallback rules.  Splitting them would mean duplicating the
category-averaging or the stat-formatting in three places.

## Seed handling

Each cell reports mean ± sample std across whichever of {s1, s2, s3, s4} are
available.  Baseline cells for Llama-3.2-3B exist only at s1 (H200 did not
re-answer those for stochastic passes), so those render as a plain point
estimate with no ±; a footnote in the caption flags the asymmetry.

Usage::

    python paper/gen_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import DEFAULT_INDEX_CSV  # noqa: E402
from paper_data import (  # noqa: E402
    BACKEND_TEX,
    CAT_RECALL, CAT_REASONING, CAT_TRUST,
    DIM_KEYS_PAPER, DIM_SHORT,
    MODEL_ORDER, MODEL_TEX,
    NEW_FROM_OLD, SEEDS,
    DEFAULT_RUN,
    CellResult, Stat,
    aggregate_seeds, category_accuracy, fmt_pct, fmt_stat,
    load_all_cells,
)
from paths import tables_dir  # noqa: E402

TABLES_DIR = tables_dir()

# Backends in the order we want rows to appear in the main table.
# Main tables include only backends with the full 5-reader sweep.
MAIN_BACKEND_ORDER = ["vanilla", "rag", "memobase", "memsearch", "oracle"]
# Appendix tables include the legacy backends that were in the previous
# paper draft but have been swapped out of the headline 5x5x3 in the May 2026
# revision. They live in the `ablation` block of experiments_index.csv.
APPENDIX_ABLATION_BACKENDS = ["vanilla_full", "memos"]
# Mem0 is intentionally omitted from the main table: at 50-agent L-scale the
# vanilla SDK does not complete Config-A ingestion cleanly (see
# §app:memsys-impl for the Phase-6 probe numbers and the Config-B reference).
# HippoRAG is intentionally omitted from the main table: it was run on
# Qwen3-8B only as a Q3-8B ablation lever; its result is reported in
# tab:ablation-summary (Finding 2 ablation row) rather than in the main
# backend-x-reader matrix.

# Paper dimensions that belong to each of the three category columns.
CATEGORY_INDICES = {
    "Rec":  CAT_RECALL,
    "Rea":  CAT_REASONING,
    "Conf": CAT_TRUST,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_category_selector(cat_key: str) -> Callable[[CellResult], float | None]:
    """Return a function that pulls a category average out of a CellResult.

    Used with :func:`aggregate_seeds` to compute mean±std across seeds.
    """
    indices = CATEGORY_INDICES[cat_key]
    return lambda cell: category_accuracy(cell, indices)


def _cell_overall_avg(cell: CellResult) -> float | None:
    """Macro-mean over the five paper dimensions D1..D5 (D6 excluded).

    Each of D1 (Cloze), D2 (Metadata), D3 (QA sub-types pooled), D4 (Cross-
    Session sub-types pooled), and D5 (Abstention) contributes equal weight
    (1/5). D6 (Permission-Aware Access) is analysed separately in
    \\S\\ref{sec:core-findings} and excluded from Avg to keep the overall
    number an evidence-grounded content score rather than mixing it with a
    deterministic access-control metric on a different scale.
    """
    parts = [category_accuracy(cell, [i]) for i in [0, 1, 2, 3, 4]]
    parts = [p for p in parts if p is not None]
    return sum(parts) / len(parts) if parts else None


def _best_mean(stats: list[Stat]) -> float | None:
    """Return the highest mean among the given stats, ignoring n==0 cells."""
    valid = [s.mean for s in stats if s.n > 0]
    return max(valid) if valid else None


# ---------------------------------------------------------------------------
# Main results table (tab:results-L, written to main_SML.tex)
# ---------------------------------------------------------------------------

def gen_main_table(grid) -> str:
    """Render the category-level main table in *transposed* layout.

    Columns are backends; rows are grouped by model with one sub-row per
    metric (Rec / Rea / Conf / Avg). This fits the single-column NeurIPS
    width far better than the wide-format layout it replaces (5 models x 4
    metrics = 20 columns was unreadable even with adjustbox scaling).

    Bolding: per (model, metric) row, the highest backend mean is bold.
    ``>= row_max - 0.0005`` is the tie threshold so rounded ties both bold.
    """
    # Selectors keyed by the metric label.
    selectors = {
        "Rec":  _cell_category_selector("Rec"),
        "Rea":  _cell_category_selector("Rea"),
        "Conf": _cell_category_selector("Conf"),
        "Avg":  _cell_overall_avg,
    }
    metric_order = ["Rec", "Rea", "Conf", "Avg"]
    metric_label = {"Rec": "Rec.", "Rea": "Rea.", "Conf": "Abst.", "Avg": "Avg"}

    # Precompute stats[(backend, model)][metric] = Stat.
    stats: dict[tuple[str, str], dict[str, Stat]] = {}
    for backend in MAIN_BACKEND_ORDER:
        for model in MODEL_ORDER:
            stats[(backend, model)] = {
                m: aggregate_seeds(grid, backend, model, selectors[m]) for m in metric_order
            }

    # Row-level max for bolding: within each (model, metric) row, best across backends.
    row_max: dict[tuple[str, str], float | None] = {}
    for model in MODEL_ORDER:
        for m in metric_order:
            row_max[(model, m)] = _best_mean([stats[(b, model)][m] for b in MAIN_BACKEND_ORDER])

    # Build the body: one group per model, 4 rows per group (one per metric).
    body_rows: list[str] = []
    for mi, model in enumerate(MODEL_ORDER):
        n_metrics = len(metric_order)
        for ci, metric in enumerate(metric_order):
            if ci == 0:
                model_cell = f"\\multirow{{{n_metrics}}}{{*}}{{{MODEL_TEX[model]}}}"
            else:
                model_cell = ""
            cells: list[str] = []
            best = row_max[(model, metric)]
            vanilla_st = stats[("vanilla", model)][metric]
            vanilla_mean = vanilla_st.mean if vanilla_st.n > 0 else None
            for backend in MAIN_BACKEND_ORDER:
                st = stats[(backend, model)][metric]
                bold = (st.n > 0 and best is not None and abs(st.mean - best) < 5e-4)
                cell_tex = fmt_stat(st, bold=bold)
                if backend != "vanilla" and st.n > 0 and vanilla_mean is not None:
                    delta_pp = (st.mean - vanilla_mean) * 100
                    if abs(delta_pp) >= 0.1:
                        if delta_pp > 0:
                            marker = f"\\,{{\\tiny\\textcolor{{green!55!black}}{{$\\uparrow${delta_pp:.1f}}}}}"
                        else:
                            marker = f"\\,{{\\tiny\\textcolor{{red!70!black}}{{$\\downarrow${abs(delta_pp):.1f}}}}}"
                        cell_tex = cell_tex + marker
                cells.append(cell_tex)
            body_rows.append(f"{model_cell} & {metric_label[metric]} & " + " & ".join(cells) + " \\\\")
        if mi != len(MODEL_ORDER) - 1:
            body_rows.append("\\midrule")

    # Backend column headers.
    backend_headers = " & ".join(
        f"\\textbf{{{BACKEND_TEX[b]}}}" for b in MAIN_BACKEND_ORDER
    )
    header_row = f" & & {backend_headers} \\\\"

    # Place a vertical rule between MemOS and Oracle so Oracle reads as the
    # ceiling reference, separated from the four candidate systems.
    backend_col_spec = ""
    for b in MAIN_BACKEND_ORDER:
        if b == "oracle":
            backend_col_spec += "| c "
        else:
            backend_col_spec += "c "
    tabular_spec = "@{}ll " + backend_col_spec + "@{}"

    caption = (
        "\\benchL{} main results: backend $\\times$ reader accuracy "
        "(mean $\\pm$ sample std across $n=3$ seeds; T$=$0.3). "
        "\\textbf{Bold} = row-local best mean. "
        "{\\tiny\\textcolor{green!55!black}{$\\uparrow$}} / "
        "{\\tiny\\textcolor{red!70!black}{$\\downarrow$}} marker on each non-Vanilla cell "
        "$=$ pp gap to row-local Vanilla (omitted when $|\\Delta| < 0.1$). "
        "Oracle is shown to the right of the vertical rule as a ceiling reference. "
        "Category column definitions are in \\S\\ref{sec:eval-method}."
    )
    label = "tab:results-L"

    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        f"\\begin{{tabular}}{{{tabular_spec}}}\n"
        "\\toprule\n"
        f"{header_row}\n"
        "\\midrule\n" + "\n".join(body_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\vspace{-6pt}\n"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Per-dimension appendix table (tab:results-L-full, appendix_L.tex)
# ---------------------------------------------------------------------------

def gen_appendix_dim_table(grid) -> str:
    """Render per-dimension D1..D7 accuracy with mean ± std.

    We drop D7 from the visible table: it is a legacy sub-dimension ("exception")
    that is not part of the six-dimension story the paper tells, and the
    current data has essentially no D7 instances under the new schema.
    """
    visible_dims = [i for i, key in enumerate(DIM_KEYS_PAPER) if key != "d7_exception"]
    visible_labels = [DIM_SHORT[i] for i in visible_dims]

    def dim_selector(i: int):
        paper_dim = DIM_KEYS_PAPER[i]
        def _get(cell: CellResult) -> float | None:
            return cell.accuracy_by_paper_dim().get(paper_dim)
        return _get

    rows: list[str] = []
    for backend in MAIN_BACKEND_ORDER:
        # Only emit rows for models that have at least one seed for this backend.
        models_for_backend = [
            m for m in MODEL_ORDER
            if any((seed, backend, m) in grid for seed in SEEDS)
        ]
        if not models_for_backend:
            continue
        label = BACKEND_TEX[backend]
        n_rows = len(models_for_backend)
        for i, model in enumerate(models_for_backend):
            prefix = f"\\multirow{{{n_rows}}}{{*}}{{{label}}}" if i == 0 else ""
            cells: list[str] = []
            for di in visible_dims:
                st = aggregate_seeds(grid, backend, model, dim_selector(di))
                cells.append(fmt_stat(st))
            avg_st = aggregate_seeds(grid, backend, model, _cell_overall_avg)
            cells.append(fmt_stat(avg_st))
            rows.append(f"{prefix}\n  & {MODEL_TEX[model]:<14} & " + " & ".join(cells) + " \\\\")
        if backend != MAIN_BACKEND_ORDER[-1]:
            rows.append("\\midrule")

    # Column layout: 2 identifier cols (backend, model) + 6 dim cols + Avg col.
    col_spec = "@{}ll " + "c" * len(visible_dims) + " c@{}"
    header = (
        "& & "
        "\\multicolumn{2}{c}{\\textit{Recall}} & "
        "\\multicolumn{2}{c}{\\textit{Reasoning}} & "
        "\\multicolumn{2}{c}{\\textit{Trustworthiness}} & \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n"
        "Backend & Model & " + " & ".join(visible_labels) + " & Avg \\\\"
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Per paper-dimension accuracy on \\benchL{} with mean $\\pm$ "
        "sample std across stochastic seeds. D1--D2: Memory Recall; "
        "D3--D4: Memory Reasoning; D5--D6: Memory Trustworthiness. Cells "
        "reporting $n = 1$ are s1-only (see Table~\\ref{tab:results-L} note).}\n"
        "\\label{tab:results-L-full}\n"
        "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n" + header + "\n\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Cross-family reader-only appendix table (Llama/Mistral only)
# ---------------------------------------------------------------------------

CROSS_FAMILY_MODELS = ["llama3b", "7b"]

def gen_cross_family_dim_table(grid) -> str:
    """Render the explicit cross-family reader-only filter table.

    Restricts the main per-dimension comparison to reader families disjoint from
    the Qwen3-235B dialogue generator, namely Llama-3.2-3B and Mistral-7B.
    Missing cells are rendered as dashes so system-incompatible configurations
    remain visible rather than silently dropped.
    """
    visible_dims = [i for i, key in enumerate(DIM_KEYS_PAPER) if key != "d7_exception"]
    visible_labels = [DIM_SHORT[i] for i in visible_dims]

    def dim_selector(i: int):
        paper_dim = DIM_KEYS_PAPER[i]
        def _get(cell: CellResult) -> float | None:
            return cell.accuracy_by_paper_dim().get(paper_dim)
        return _get

    rows: list[str] = []
    for backend in MAIN_BACKEND_ORDER:
        label = BACKEND_TEX[backend]
        n_rows = len(CROSS_FAMILY_MODELS)
        for i, model in enumerate(CROSS_FAMILY_MODELS):
            prefix = f"\\multirow{{{n_rows}}}{{*}}{{{label}}}" if i == 0 else ""
            cells: list[str] = []
            for di in visible_dims:
                st = aggregate_seeds(grid, backend, model, dim_selector(di))
                cells.append(fmt_stat(st))
            avg_st = aggregate_seeds(grid, backend, model, _cell_overall_avg)
            cells.append(fmt_stat(avg_st))
            rows.append(f"{prefix}\n  & {MODEL_TEX[model]:<14} & " + " & ".join(cells) + " \\\\")
        if backend != MAIN_BACKEND_ORDER[-1]:
            rows.append("\\midrule")

    col_spec = "@{}ll " + "c" * len(visible_dims) + " c@{}"
    header = (
        "& & "
        "\\multicolumn{2}{c}{\\textit{Recall}} & "
        "\\multicolumn{2}{c}{\\textit{Reasoning}} & "
        "\\multicolumn{2}{c}{\\textit{Trustworthiness}} & \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n"
        "Backend & Model & " + " & ".join(visible_labels) + " & Avg \\\\"
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Cross-family reader-only filter on \\benchL{}, retaining only "
        "Llama-3.2-3B and Mistral-7B readers and excluding all Qwen readers. "
        "Metrics match Table~\\ref{tab:results-L-full}: D1--D2 Recall, D3--D4 "
        "Reasoning, D5--D6 Trustworthiness, and Avg = mean(D1..D5). Dashes denote "
        "system-incompatible cells rather than unrun comparisons.}\n"
        "\\label{tab:cross-family-readers}\n"
        "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n" + header + "\n\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Per-sub-dimension appendix table (tab:results-L-subdim, appendix_L_subdim.tex)
# ---------------------------------------------------------------------------

# Sub-dimension order follows the paper's D1..D6 grouping so rows read left-to-right
# along the same categorical order as the main table.
SUB_DIM_ORDER = [
    ("d1_conflict",        "Confl."),   # part of D4 Cross-Session
    ("d2_anaphora",        "Anaph."),   # part of D4
    ("d7_qa",              "QA"),       # part of D3 Factual QA
    ("d8_temporal",        "Temp."),    # part of D3
    ("d10_counterfactual", "Adv."),     # part of D3 (counterfactual / adversarial)
    ("d5_cloze",           "Cloze"),    # D1
    ("d6_metadata",        "Meta."),    # D2
    ("d3_confabulation",   "Abst."),    # D5
    ("d4_permission",      "Perm."),    # D6
]

def gen_appendix_subdim_table(grid) -> str:
    """Render the detailed d1..d11 sub-dimension table.

    D6 permission-aware access (``d4_permission``) is reported deterministically
    from raw outputs for every backend; no LLM judge is used in the final paper
    numbers.
    """
    def subdim_selector(sub: str):
        def _get(cell: CellResult) -> float | None:
            return cell.accuracy_by_sub_dim().get(sub)
        return _get

    rows: list[str] = []
    for backend in MAIN_BACKEND_ORDER:
        models_for_backend = [
            m for m in MODEL_ORDER
            if any((seed, backend, m) in grid for seed in SEEDS)
        ]
        if not models_for_backend:
            continue
        label = BACKEND_TEX[backend]
        n_rows = len(models_for_backend)
        for i, model in enumerate(models_for_backend):
            prefix = f"\\multirow{{{n_rows}}}{{*}}{{{label}}}" if i == 0 else ""
            cells: list[str] = []
            for sub_key, _ in SUB_DIM_ORDER:
                st = aggregate_seeds(grid, backend, model, subdim_selector(sub_key))
                cells.append(fmt_stat(st))
            rows.append(f"{prefix}\n  & {MODEL_TEX[model]:<14} & " + " & ".join(cells) + " \\\\")
        if backend != MAIN_BACKEND_ORDER[-1]:
            rows.append("\\midrule")

    col_spec = "@{}ll " + "c" * len(SUB_DIM_ORDER) + "@{}"
    sub_labels = [lbl for _, lbl in SUB_DIM_ORDER]
    # Group headers: Cross-Session (2) + Factual QA (3) + Recall (2) + Confab (1) + Perm (1).
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Sub-dimension accuracy on \\benchL{} with mean $\\pm$ std "
        "across stochastic seeds. Internal dimension IDs shown; mapping to "
        "paper dimensions: D1 Cloze $\\to$ d5, D2 Metadata $\\to$ d6, "
        "D3 QA $\\to$ d7/d8/d10, D4 Cross-Session $\\to$ d1/d2, "
        "D5 Abstention $\\to$ d3, D6 Permission-Aware Access $\\to$ d4. "
        "D6 is reported deterministically from raw outputs for all backends; "
        "all other answer-required dimensions use GPT-4o-mini as the primary judge.}\n"
        "\\label{tab:results-L-subdim}\n"
        "\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        "& & \\multicolumn{2}{c}{\\textit{Cross-Sess.}} "
        "& \\multicolumn{3}{c}{\\textit{Factual QA}} "
        "& \\multicolumn{2}{c}{\\textit{Recall}} "
        "& \\textit{Abst.} & \\textit{Perm.} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-9}\n"
        "Backend & Model & " + " & ".join(sub_labels) + " \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def gen_ablation_table(grid) -> str:
    """Render the appendix ablation table (vanilla_full + memos).

    These backends were in the previous paper draft but moved out of the
    headline 5x5x3 in the May 2026 revision (vanilla replaced by
    vanilla-512; memos replaced by memsearch). The ablation table keeps
    them visible side-by-side with the main vanilla / memsearch columns
    so reviewers can compare.
    """
    selectors = {
        "Rec":  _cell_category_selector("Rec"),
        "Rea":  _cell_category_selector("Rea"),
        "Conf": _cell_category_selector("Conf"),
        "Avg":  _cell_overall_avg,
    }
    metric_order = ["Rec", "Rea", "Conf", "Avg"]
    metric_label = {"Rec": "Rec.", "Rea": "Rea.", "Conf": "Abst.", "Avg": "Avg"}

    cols = [
        ("vanilla",      "Vanilla\\textsubscript{512}"),
        ("vanilla_full", "Vanilla\\textsubscript{full}"),
        ("memsearch",    "MemSearch"),
        ("memos",        "MemOS"),
    ]
    col_keys = [c[0] for c in cols]
    col_labels = [c[1] for c in cols]

    stats: dict[tuple[str, str], dict[str, Stat]] = {}
    for backend, _ in cols:
        for model in MODEL_ORDER:
            stats[(backend, model)] = {
                m: aggregate_seeds(grid, backend, model, selectors[m]) for m in metric_order
            }

    body_rows: list[str] = []
    for mi, model in enumerate(MODEL_ORDER):
        for ci, metric in enumerate(metric_order):
            model_cell = f"\\multirow{{{len(metric_order)}}}{{*}}{{{MODEL_TEX[model]}}}" if ci == 0 else ""
            cells: list[str] = []
            for backend in col_keys:
                cells.append(fmt_stat(stats[(backend, model)][metric]))
            body_rows.append(f"{model_cell} & {metric_label[metric]} & " + " & ".join(cells) + " \\\\")
        if mi != len(MODEL_ORDER) - 1:
            body_rows.append("\\midrule")

    backend_headers = " & ".join(f"\\textbf{{{lbl}}}" for lbl in col_labels)
    header_row = f" & & {backend_headers} \\\\"
    tabular_spec = "@{}ll | c c | c c @{}"  # vanilla pair | memory pair

    caption = (
        "\\benchL{} ablation: paired comparison between the main-table "
        "(\\textsubscript{512} = max\\_tokens=512 generation budget; "
        "MemSearch = markdown + Milvus-Lite hybrid retriever) and the "
        "previously-default backends. Mean $\\pm$ sample std across $n=3$ "
        "seeds; T$=$0.3."
    )
    label = "tab:results-L-ablation"

    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        f"\\begin{{tabular}}{{{tabular_spec}}}\n"
        "\\toprule\n"
        f"{header_row}\n"
        "\\midrule\n" + "\n".join(body_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\vspace{-6pt}\n"
        "\\end{table}\n"
    )


def main(run_dir: Path = DEFAULT_RUN) -> None:
    grid = load_all_cells(run_dir)
    print(f"[gen_tables] loaded {len(grid)} cells from {DEFAULT_INDEX_CSV}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Table 3: main results (category level) ----------------------------
    main_tex = gen_main_table(grid)
    (TABLES_DIR / "main_SML.tex").write_text(main_tex)
    print(f"[gen_tables] wrote {TABLES_DIR/'main_SML.tex'}")

    # --- Table 7: per-paper-dim appendix -----------------------------------
    app_tex = gen_appendix_dim_table(grid)
    (TABLES_DIR / "appendix_L.tex").write_text(app_tex)
    print(f"[gen_tables] wrote {TABLES_DIR/'appendix_L.tex'}")

    # --- Cross-family reader-only appendix ----------------------------------
    cf_tex = gen_cross_family_dim_table(grid)
    (TABLES_DIR / "appendix_cross_family.tex").write_text(cf_tex)
    print(f"[gen_tables] wrote {TABLES_DIR/'appendix_cross_family.tex'}")

    # --- Table 8: per-sub-dim appendix -------------------------------------
    sub_tex = gen_appendix_subdim_table(grid)
    (TABLES_DIR / "appendix_L_subdim.tex").write_text(sub_tex)
    print(f"[gen_tables] wrote {TABLES_DIR/'appendix_L_subdim.tex'}")

    # --- Ablation appendix (vanilla_full + memos vs main) -------------------
    abl_tex = gen_ablation_table(grid)
    (TABLES_DIR / "appendix_L_ablation.tex").write_text(abl_tex)
    print(f"[gen_tables] wrote {TABLES_DIR/'appendix_L_ablation.tex'}")


if __name__ == "__main__":
    main()
