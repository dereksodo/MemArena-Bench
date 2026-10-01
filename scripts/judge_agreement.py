#!/usr/bin/env python3
"""Agreement and Cohen's kappa between two judges over the grid-runner tree.

For every cell judged by both judges
(``<runs>/<backend>/<model>/<seed>/<system>/evaluation_results_*_judge_<tag>.json``,
see memarena/judges.py) the records are paired by ``question_id`` and compared:

* D1-D5 (binary correct / incorrect): only records the LLM judge scored in both
  files (reason ``evidence_judge: ...`` or ``llm_judge: ...``). Deterministically
  scored records (``mcq_exact``, ``refusal_detection``, cloze, abstention rules,
  ``answer_scoring_skipped``, ``pre_scored:*`` ...) are the same under any judge
  and are excluded; the report lists every excluded reason with its count.
* D6 (d4_permission): the 5-label rubric (``policy_category``) and the binary
  leak (label == DISCLOSE_CORRECT, the paper's leak definition). Empty
  predictions are labelled OTHER without a judge call and are excluded.
  Records never scored (``answer_scored`` false, LLM_ERROR) are excluded everywhere.

Output: per cell, per paper dimension and pooled agreement / kappa, as a
markdown report and a CSV.

    python3 scripts/judge_agreement.py --runs-dir out/runs                     # gpt4omini vs deepseek
    python3 scripts/judge_agreement.py --runs-dir out/runs --judges gpt4omini deepseek \\
        --seeds s2,s3,s4 --out-md out/judge_agreement/agreement.md --out-csv out/judge_agreement/agreement.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memarena.judges import (  # noqa: E402
    JUDGES,
    PRIMARY_JUDGE,
    SECONDARY_JUDGE,
    agreement,
    cohen_kappa,
    fmt,
    judge_spec,
)

JUDGE_REASON_PREFIXES = ("evidence_judge:", "llm_judge:")
LEAK_LABEL = "DISCLOSE_CORRECT"
UNSCORED_PREFIXES = ("LLM_ERROR", "TEXT_SESSIONS_ERROR")
SUBDIM = {
    "d1": "d1_conflict", "d2": "d2_anaphora", "d3": "d3_confabulation", "d4": "d4_permission",
    "d5": "d5_cloze", "d6": "d6_metadata", "d7": "d7_qa", "d8": "d8_temporal", "d9": "d9_negation",
    "d10": "d10_counterfactual", "d11": "d11_exception",
}
# sub-dimension -> paper dimension (memarena/figures/main_grid.py PAPER_DIMS; D6 = d4_permission)
PAPER_DIM_OF = {
    "d5_cloze": "D1", "d6_metadata": "D2",
    "d7_qa": "D3", "d8_temporal": "D3", "d10_counterfactual": "D3",
    "d1_conflict": "D4", "d2_anaphora": "D4",
    "d3_confabulation": "D5", "d4_permission": "D6",
}
BACKENDS = ("vanilla", "rag", "memobase", "memsearch", "oracle")
MODELS = ("0_6b", "llama3b", "7b", "8b", "32b")
SEEDS = ("s2", "s3", "s4")


def _truthy(v) -> bool:
    return v is True or str(v) == "True"


def _is_scored(rec: Mapping) -> bool:
    if rec.get("answer_scored") is False or str(rec.get("answer_scored")) == "False":
        return False
    raw = rec.get("raw_response")
    return not (isinstance(raw, str) and raw.startswith(UNSCORED_PREFIXES))


def _subdim(rec: Mapping) -> str:
    return str(rec.get("dimension") or SUBDIM.get(str(rec.get("question_id", "")).split("_")[0], ""))


def _is_d6(rec: Mapping) -> bool:
    return str(rec.get("question_id", "")).startswith("d4_perm")


def _is_judged(rec: Mapping) -> bool:
    return str(rec.get("reason") or "").startswith(JUDGE_REASON_PREFIXES)


def reason_key(reason: str) -> str:
    """``'cloze_3/5'`` -> ``'cloze_k/n'``, ``'pre_scored:LLM_ERROR'`` -> ``'pre_scored'``."""
    key = str(reason or "").split(":", 1)[0].strip() or "(none)"
    return re.sub(r"\d+/\d+", "k/n", key)


@dataclass
class Pairs:
    """Paired verdicts of judge A and judge B, grouped by paper dimension."""

    binary: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)   # dim -> (a, b)
    d6_labels: List[Tuple[str, str]] = field(default_factory=list)
    excluded: Counter = field(default_factory=Counter)                        # reason -> n
    n_unpaired: int = 0
    n_scoring_mismatch: int = 0
    n_d6_empty: int = 0

    def extend(self, other: "Pairs") -> None:
        for dim, rows in other.binary.items():
            self.binary.setdefault(dim, []).extend(rows)
        self.d6_labels.extend(other.d6_labels)
        self.excluded.update(other.excluded)
        self.n_unpaired += other.n_unpaired
        self.n_scoring_mismatch += other.n_scoring_mismatch
        self.n_d6_empty += other.n_d6_empty

    def binary_all(self, dims: Optional[Iterable[str]] = None) -> List[Tuple[int, int]]:
        keep = set(dims) if dims is not None else None
        return [p for d, rows in sorted(self.binary.items()) if keep is None or d in keep for p in rows]


def compare_details(details_a: Sequence[Mapping], details_b: Sequence[Mapping]) -> Pairs:
    """Pair two judges' ``details`` of the same cell by question_id."""
    out = Pairs()
    by_b = {str(r.get("question_id")): r for r in details_b}
    for a in details_a:
        qid = str(a.get("question_id"))
        b = by_b.get(qid)
        if b is None:
            out.n_unpaired += 1
            continue
        if not (_is_scored(a) and _is_scored(b)):
            out.excluded["unscored (answer_scored false / LLM_ERROR)"] += 1
            continue
        if _is_d6(a):
            if not str(a.get("prediction") or "").strip() or not str(b.get("prediction") or "").strip():
                out.n_d6_empty += 1
                out.excluded["D6 empty prediction (OTHER without a judge call)"] += 1
                continue
            out.d6_labels.append((str(a.get("policy_category") or "").upper(),
                                  str(b.get("policy_category") or "").upper()))
            continue
        ja, jb = _is_judged(a), _is_judged(b)
        if not (ja and jb):
            if ja != jb:
                out.n_scoring_mismatch += 1
                out.excluded["judge-scored in one file only"] += 1
            else:
                out.excluded[reason_key(str(a.get("reason") or ""))] += 1
            continue
        dim = PAPER_DIM_OF.get(_subdim(a), _subdim(a) or "?")
        out.binary.setdefault(dim, []).append((int(_truthy(a.get("correct"))), int(_truthy(b.get("correct")))))
    out.n_unpaired += len(set(by_b) - {str(r.get("question_id")) for r in details_a})
    return out


def binary_stats(pairs: Sequence[Tuple[int, int]]) -> dict:
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    n = len(pairs)
    return {"n": n, "agreement": agreement(a, b), "kappa": cohen_kappa(a, b),
            "acc_a": (sum(a) / n) if n else float("nan"), "acc_b": (sum(b) / n) if n else float("nan")}


def d6_stats(pairs: Sequence[Tuple[str, str]]) -> dict:
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    la = [int(x == LEAK_LABEL) for x in a]
    lb = [int(y == LEAK_LABEL) for y in b]
    n = len(pairs)
    return {"n": n, "agreement": agreement(a, b), "kappa": cohen_kappa(a, b),
            "leak_agreement": agreement(la, lb), "leak_kappa": cohen_kappa(la, lb),
            "leak_a": (sum(la) / n) if n else float("nan"), "leak_b": (sum(lb) / n) if n else float("nan")}


def _load_details(path: Path) -> List[Mapping]:
    return json.loads(path.read_text(encoding="utf-8")).get("details") or []


def collect(runs_dir: Path, judge_a: str, judge_b: str, seeds: Sequence[str] = SEEDS,
            models: Sequence[str] = MODELS, backends: Sequence[str] = BACKENDS):
    """[(cell key, Pairs)] for every cell both judges finished, plus the missing cells."""
    from memarena.figures.paper_data import _runs_cell_done, runs_eval_path

    cells: List[Tuple[Tuple[str, str, str], Pairs]] = []
    missing: List[Tuple[Tuple[str, str, str], str]] = []
    for seed in seeds:
        for model in models:
            for backend in backends:
                key = (backend, model, seed)
                pa = runs_eval_path(runs_dir, backend, model, seed, judge_a)
                pb = runs_eval_path(runs_dir, backend, model, seed, judge_b)
                why = []
                for tag, p in ((judge_a, pa), (judge_b, pb)):
                    if not p.exists():
                        why.append(f"no {tag} file")
                    elif not _runs_cell_done(p.parent.parent, tag):
                        why.append(f"{tag} not done")
                if why:
                    missing.append((key, ", ".join(why)))
                    continue
                cells.append((key, compare_details(_load_details(pa), _load_details(pb))))
    return cells, missing


DIM_ORDER = ("D1", "D2", "D3", "D4", "D5")


def build_rows(cells, judge_a: str, judge_b: str) -> Tuple[List[dict], Pairs]:
    """CSV rows: one per cell, per paper dimension and the pooled total."""
    rows: List[dict] = []
    pooled = Pairs()

    def row(scope: str, key: str, p: Pairs, dims: Optional[Sequence[str]] = None, d6: bool = True) -> dict:
        bs = binary_stats(p.binary_all(dims))
        ds = d6_stats(p.d6_labels) if d6 else d6_stats([])
        return {
            "scope": scope, "key": key, "judge_a": judge_a, "judge_b": judge_b,
            "binary_n": bs["n"], "binary_agreement": bs["agreement"], "binary_kappa": bs["kappa"],
            "acc_a": bs["acc_a"], "acc_b": bs["acc_b"],
            "d6_n": ds["n"], "d6_label_agreement": ds["agreement"], "d6_label_kappa": ds["kappa"],
            "d6_leak_agreement": ds["leak_agreement"], "d6_leak_kappa": ds["leak_kappa"],
            "leak_a": ds["leak_a"], "leak_b": ds["leak_b"],
        }

    for (backend, model, seed), p in cells:
        rows.append(row("cell", f"{backend}/{model}/{seed}", p))
        pooled.extend(p)
    present = sorted(pooled.binary, key=lambda d: (d not in DIM_ORDER, d))
    for dim in present:
        rows.append(row("dimension", dim, pooled, dims=[dim], d6=False))
    if pooled.d6_labels:
        rows.append(row("dimension", "D6", Pairs(d6_labels=pooled.d6_labels)))
    rows.append(row("pooled", "all", pooled))
    return rows, pooled


def _md_table(header: Sequence[str], body: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return out


def render_markdown(rows: List[dict], pooled: Pairs, missing, judge_a: str, judge_b: str) -> str:
    ma, mb = judge_spec(judge_a).model, judge_spec(judge_b).model
    lines = [f"# Judge agreement: {judge_a} (A, `{ma}`) vs {judge_b} (B, `{mb}`)", ""]
    n_cells = sum(1 for r in rows if r["scope"] == "cell")
    lines += [f"Cells judged by both: {n_cells}; missing or unfinished: {len(missing)}.",
              "Binary = D1-D5 records the LLM judge scored in both files (correct / incorrect). "
              "D6 = 5-label rubric and binary leak (label DISCLOSE_CORRECT).", ""]
    pr = next(r for r in rows if r["scope"] == "pooled")
    lines += ["## Pooled", ""]
    lines += _md_table(
        ["", "n", "agreement", "Cohen's κ", "A rate", "B rate"],
        [["binary correct (D1-D5)", str(pr["binary_n"]), fmt(pr["binary_agreement"]), fmt(pr["binary_kappa"]),
          fmt(pr["acc_a"]), fmt(pr["acc_b"])],
         ["D6 5-label", str(pr["d6_n"]), fmt(pr["d6_label_agreement"]), fmt(pr["d6_label_kappa"]), "", ""],
         ["D6 binary leak", str(pr["d6_n"]), fmt(pr["d6_leak_agreement"]), fmt(pr["d6_leak_kappa"]),
          fmt(pr["leak_a"]), fmt(pr["leak_b"])]])
    lines += ["", "## Per paper dimension", ""]
    body = []
    for r in rows:
        if r["scope"] != "dimension":
            continue
        if r["key"] == "D6":
            body.append(["D6 (5-label)", str(r["d6_n"]), fmt(r["d6_label_agreement"]), fmt(r["d6_label_kappa"]), "", ""])
            body.append(["D6 (leak)", str(r["d6_n"]), fmt(r["d6_leak_agreement"]), fmt(r["d6_leak_kappa"]),
                         fmt(r["leak_a"]), fmt(r["leak_b"])])
        else:
            body.append([r["key"], str(r["binary_n"]), fmt(r["binary_agreement"]), fmt(r["binary_kappa"]),
                         fmt(r["acc_a"]), fmt(r["acc_b"])])
    lines += _md_table(["dimension", "n", "agreement", "Cohen's κ", "A rate", "B rate"], body)
    lines += ["", "## Per cell", ""]
    body = [[r["key"], str(r["binary_n"]), fmt(r["binary_agreement"]), fmt(r["binary_kappa"]),
             fmt(r["acc_a"]), fmt(r["acc_b"]), str(r["d6_n"]), fmt(r["d6_label_agreement"]),
             fmt(r["d6_label_kappa"]), fmt(r["d6_leak_agreement"]), fmt(r["d6_leak_kappa"])]
            for r in rows if r["scope"] == "cell"]
    lines += _md_table(["cell", "bin n", "bin agr", "bin κ", "A acc", "B acc", "D6 n", "D6 agr", "D6 κ",
                        "leak agr", "leak κ"], body)
    lines += ["", "## Excluded records (pooled over the cells above)", ""]
    lines += _md_table(["reason", "records"], [[k, str(v)] for k, v in pooled.excluded.most_common()] or [["(none)", "0"]])
    if pooled.n_unpaired:
        lines += ["", f"Records present in only one judge's file: {pooled.n_unpaired}."]
    if missing:
        lines += ["", "## Cells left out", ""]
        lines += _md_table(["cell", "why"], [[f"{b}/{m}/{s}", why] for (b, m, s), why in missing])
    return "\n".join(lines) + "\n"


CSV_FIELDS = ["scope", "key", "judge_a", "judge_b", "binary_n", "binary_agreement", "binary_kappa",
              "acc_a", "acc_b", "d6_n", "d6_label_agreement", "d6_label_kappa", "d6_leak_agreement",
              "d6_leak_kappa", "leak_a", "leak_b"]


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and v == v else ("" if isinstance(v, float) else v))
                        for k, v in r.items()})


def _split(value: str) -> List[str]:
    return [x for x in re.split(r"[,\s]+", value or "") if x]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "out" / "runs")
    ap.add_argument("--judges", nargs=2, default=[PRIMARY_JUDGE, SECONDARY_JUDGE], metavar=("A", "B"),
                    choices=sorted(JUDGES))
    ap.add_argument("--seeds", default=",".join(SEEDS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--out-md", type=Path, help="default <runs-dir>/_judge_agreement/agreement_<A>_vs_<B>.md")
    ap.add_argument("--out-csv", type=Path, help="default <runs-dir>/_judge_agreement/agreement_<A>_vs_<B>.csv")
    args = ap.parse_args(argv)

    a, b = args.judges
    if a == b:
        ap.error("pick two different judges")
    runs = args.runs_dir.expanduser().resolve()
    cells, missing = collect(runs, a, b, _split(args.seeds), _split(args.models), _split(args.backends))
    if not cells:
        print(f"[judge_agreement] no cell of {runs} is judged by both {a} and {b}", file=sys.stderr)
        for (bk, m, s), why in missing[:10]:
            print(f"  {bk}/{m}/{s}: {why}", file=sys.stderr)
        return 1
    rows, pooled = build_rows(cells, a, b)
    stem = f"agreement_{a}_vs_{b}"
    out_md = args.out_md or runs / "_judge_agreement" / f"{stem}.md"
    out_csv = args.out_csv or runs / "_judge_agreement" / f"{stem}.csv"
    text = render_markdown(rows, pooled, missing, a, b)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    write_csv(rows, out_csv)
    pr = rows[-1]
    print(f"[judge_agreement] {len(cells)} cells ({len(missing)} left out); "
          f"binary n={pr['binary_n']} agreement={fmt(pr['binary_agreement'])} kappa={fmt(pr['binary_kappa'])}; "
          f"D6 n={pr['d6_n']} 5-label kappa={fmt(pr['d6_label_kappa'])} leak kappa={fmt(pr['d6_leak_kappa'])}")
    print(f"[judge_agreement] excluded: {dict(pooled.excluded.most_common())}")
    print(f"[judge_agreement] wrote {out_md}")
    print(f"[judge_agreement] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
