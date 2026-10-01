#!/usr/bin/env python3
"""Generate Table 9 (``tab:results-L-privacy``) → ``paper/tables/appendix_L_privacy.tex``.

D6 Permission-Aware Access detailed metrics: withhold accuracy, false-refusal
rate, and utility (= 1 - false refusal). Source: per-instance
``policy_compliant`` in the evaluation JSONs (deterministic keyword scoring,
no LLM judge). Aggregated as mean ± sample std across the stochastic seeds
defined in :data:`paper.paper_data.SEEDS` (default: s2/s3/s4).

Rows cover baselines (vanilla/rag/oracle) × {0_6b, 7b, 8b, 32b} plus
structured (memobase/memos) × {0_6b, llama3b, 8b, 32b}. Llama baselines are
s1-only and therefore omitted from the main stochastic table.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import policy_compliant_from_armB  # noqa: E402
from paper_data import (  # noqa: E402
    BACKEND_TEX, MODEL_ORDER, MODEL_TEX,
    SEEDS, DEFAULT_RUN,
    load_all_cells,
)
from paths import tables_dir  # noqa: E402

TABLES_DIR = tables_dir()

# Backend ordering in the appendix table. Same as main table.
BACKENDS = ["vanilla", "oracle", "rag", "memobase", "memos"]


def _cell_privacy(path: Path) -> tuple[int, int, int, int] | None:
    """Return (deny_correct, deny_total, allow_correct, allow_total) for D6."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    dc = dn = ac = an = 0
    for r in data.get("details", []):
        qid = r.get("question_id") or ""
        if not qid.startswith("d4_"):
            continue
        compliant = policy_compliant_from_armB(r)
        exp = r.get("policy_expected")
        if exp == "DENY_NO_ACCESS":
            dn += 1
            dc += int(compliant)
        elif exp == "ALLOW":
            an += 1
            ac += int(compliant)
    return (dc, dn, ac, an)


def _aggregate(grid, backend, model):
    withholds, false_refs = [], []
    for seed in SEEDS:
        cell = grid.get((seed, backend, model))
        if cell is None:
            continue
        parts = _cell_privacy(cell.source_path)
        if parts is None:
            continue
        dc, dn, ac, an = parts
        if dn == 0 or an == 0:
            continue
        withholds.append(dc / dn)
        false_refs.append(1.0 - ac / an)
    if not withholds:
        return None

    def ms(xs):
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return m, s

    w_m, w_s = ms(withholds)
    fr_m, fr_s = ms(false_refs)
    util_m, util_s = ms([1.0 - x for x in false_refs])
    return (w_m, w_s, fr_m, fr_s, util_m, util_s, len(withholds))


def _fmt(mean: float | None, std: float | None, n: int) -> str:
    if mean is None or n == 0:
        return "--"
    m = f"{mean * 100:.1f}"
    if n < 2 or std is None:
        return m
    return f"{m}{{\\scriptsize$\\pm${std * 100:.1f}}}"


def main() -> None:
    grid = load_all_cells(DEFAULT_RUN)

    rows: list[str] = []
    for backend in BACKENDS:
        models = [m for m in MODEL_ORDER
                  if _aggregate(grid, backend, m) is not None]
        if not models:
            continue
        label = BACKEND_TEX[backend]
        nr = len(models)
        for i, model in enumerate(models):
            prefix = f"\\multirow{{{nr}}}{{*}}{{{label}}}" if i == 0 else ""
            agg = _aggregate(grid, backend, model)
            assert agg is not None
            w_m, w_s, fr_m, fr_s, util_m, util_s, n = agg
            rows.append(
                f"{prefix}\n  & {MODEL_TEX[model]:<14} "
                f"& {_fmt(w_m, w_s, n):<24} "
                f"& {_fmt(fr_m, fr_s, n):<24} "
                f"& {_fmt(util_m, util_s, n):<24} \\\\"
            )
        if backend != BACKENDS[-1]:
            rows.append("\\midrule")

    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{D6 Permission-Aware Access detailed metrics on \\benchL{} "
        "(200 instances per seed: 80 DENY, 120 ALLOW). "
        "Withhold Acc.\\ = fraction of DENY queries correctly refused. "
        "False Ref.\\ = fraction of ALLOW queries incorrectly refused. "
        "Utility = $1 -$ False Ref.\\ rate. "
        "Values are mean $\\pm$ sample std across stochastic seeds (s2/s3/s4); "
        "scoring is deterministic from the pre-registered refusal keyword list "
        "on raw model predictions (no LLM judge).}\n"
        "\\label{tab:results-L-privacy}\n"
        "\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}ll ccc@{}}\n"
        "\\toprule\n"
        "Backend & Model & Withhold (\\%) & False Ref.\\ (\\%) & Utility (\\%) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "appendix_L_privacy.tex"
    out.write_text(tex)
    print(f"[gen_privacy_table] wrote {out}")


if __name__ == "__main__":
    main()
