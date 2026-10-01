#!/usr/bin/env python3
"""Generate Table 8 (``tab:results-L-f1``) → ``paper/tables/appendix_L_f1.tex``.

Token F1 is the SQuAD-style word-level overlap between ``prediction`` and
``gold_answer``, computed per instance and averaged per cell. Unlike the
binary judge score, Token F1 credits partial correctness (e.g. a prediction
missing one of three expected entities still earns 2/3 recall). We report
Token F1 for the open-ended dimensions only — D1 Cloze uses exact-match, D5
Confabulation is a yes/no rubric, and D6 Permission uses keyword refusal, so
F1 is not informative for them.

Sub-dimensions shown (consistent with the paper's appendix layout):

* D4 Cross-Session: d1_conflict (Confl.), d2_anaphora (Anaph.)
* D3 Factual QA: d7_qa (Std. QA), d8_temporal (Temp.), d10_counterfactual (Adv.)
* D2 Recall: d6_metadata (Meta.)

Aggregation: mean ± sample std across the stochastic seeds in
:data:`paper.paper_data.SEEDS` (default s2/s3/s4). Llama-3.2-3B baselines
are s1-only and therefore do not appear in this table.
"""

from __future__ import annotations

import json
import re
import statistics
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import (  # noqa: E402
    BACKEND_TEX, MODEL_ORDER, MODEL_TEX,
    SEEDS, DEFAULT_RUN,
    load_all_cells,
)
from paths import tables_dir  # noqa: E402

TABLES_DIR = tables_dir()

# Sub-dim column layout in display order.
F1_COLUMNS = [
    ("d1_conflict",        "Confl."),
    ("d2_anaphora",        "Anaph."),
    ("d7_qa",              "Std.\\ QA"),
    ("d8_temporal",        "Temp."),
    ("d10_counterfactual", "Adv."),
    ("d6_metadata",        "Meta."),
]

BACKENDS = ["vanilla", "oracle", "rag", "memobase", "memos"]


# ---------------------------------------------------------------------------
# SQuAD-style token F1
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop articles -- SQuAD v1.1 preprocessing."""
    text = text.lower().translate(_PUNCT_TABLE)
    return [w for w in text.split() if w and w not in _ARTICLES]


def _token_f1(pred: str, gold: str) -> float:
    """Multiset-based token F1 (duplicate tokens count with multiplicity)."""
    p = Counter(_normalize(pred or ""))
    g = Counter(_normalize(gold or ""))
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _sub_dim_f1(path: Path, sub_key: str) -> float | None:
    """Mean Token F1 over rows whose qid prefix matches sub_key's D-number."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    # Map sub_key → qid prefix. e.g. d1_conflict → 'd1'
    prefix = sub_key.split("_", 1)[0]
    f1s: list[float] = []
    for r in data.get("details", []):
        qid = r.get("question_id") or ""
        if not qid.startswith(prefix + "_"):
            continue
        if r.get("answer_scored") is False:
            continue
        f1s.append(_token_f1(r.get("prediction", ""), r.get("gold_answer", "")))
    if not f1s:
        return None
    return sum(f1s) / len(f1s)


def _aggregate(grid, backend, model, sub_key: str):
    vals = []
    for seed in SEEDS:
        cell = grid.get((seed, backend, model))
        if cell is None:
            continue
        v = _sub_dim_f1(cell.source_path, sub_key)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None, 0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s, len(vals)


def _fmt(mean: float | None, std: float | None, n: int) -> str:
    if mean is None or n == 0:
        return "--"
    out = f"{mean * 100:.1f}"
    if n < 2 or std is None:
        return out
    return f"{out}{{\\scriptsize$\\pm${std * 100:.1f}}}"


def main() -> None:
    grid = load_all_cells(DEFAULT_RUN)

    rows: list[str] = []
    for backend in BACKENDS:
        # A model is included if at least one sub-dim yields a value.
        models = []
        for m in MODEL_ORDER:
            if any(_aggregate(grid, backend, m, k)[2] > 0 for k, _ in F1_COLUMNS):
                models.append(m)
        if not models:
            continue
        label = BACKEND_TEX[backend]
        nr = len(models)
        for i, model in enumerate(models):
            prefix = f"\\multirow{{{nr}}}{{*}}{{{label}}}" if i == 0 else ""
            cells = []
            for sub_key, _ in F1_COLUMNS:
                m, s, n = _aggregate(grid, backend, model, sub_key)
                cells.append(_fmt(m, s, n))
            rows.append(f"{prefix}\n  & {MODEL_TEX[model]:<14} & " + " & ".join(cells) + " \\\\")
        if backend != BACKENDS[-1]:
            rows.append("\\midrule")

    labels = " & ".join(lbl for _, lbl in F1_COLUMNS)
    col_spec = "@{}ll " + "c" * len(F1_COLUMNS) + "@{}"
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Token F1 (\\%) on open-ended dimensions of \\benchL{} with "
        "mean $\\pm$ sample std across stochastic seeds (s2/s3/s4). "
        "Token F1 measures SQuAD-style word-level overlap between prediction "
        "and gold answer, crediting partial correctness that binary accuracy "
        "misses. Dimensions not shown (D5 Confabulation, D1 Cloze, D6 Permission) "
        "use non-F1 scoring.}\n"
        "\\label{tab:results-L-f1}\n"
        "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        "& & \\multicolumn{2}{c}{\\textit{Cross-Session}} "
        "& \\multicolumn{3}{c}{\\textit{Factual QA}} "
        "& \\textit{Recall} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-7}\n"
        "Backend & Model & " + labels + " \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "appendix_L_f1.tex"
    out.write_text(tex)
    print(f"[gen_tokenf1] wrote {out}")


if __name__ == "__main__":
    main()
