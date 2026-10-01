"""Judge-vs-human agreement (Cohen's kappa) on the human-calibration sample.

Inputs:
  --labels: labels.json from server.py. Either {"reviewers": {"<rid>": {id: value}}}
            (several annotators; the human label is their strict majority, ties
            dropped) or a flat {"labels": {id: value}} / {id: value}. A value is
            0/1 for binary items, a D6 label (DISCLOSE_CORRECT, ...) for D6
            items, or "skip".
  --pool:   the judged sample: sample.json (the paper's gpt-4o-mini verdicts) or
            sample_judge_<tag>.json from rejudge_sample.py (a re-judge with
            the gold passed, for any judge of memarena/judges.py).

Outputs (default under out/human_calibration/, never into paper/):
  --tex:  LaTeX table
  --json: overall + per-dimension metrics, D6 5-label and leak, per reviewer

Metrics:
  - binary items: agreement and Cohen's kappa of human 0/1 vs judge_correct,
    overall and per dimension;
  - D6 items: 5-label agreement / kappa (human label vs judge_label) and the
    binary leak (label == DISCLOSE_CORRECT);
  - with several reviewers: the same per reviewer, and inter-annotator kappa.

    python3 -m memarena.human_calibration.analyze_judge_human \\
        --labels memarena/human_calibration/labels.json \\
        --pool memarena/human_calibration/sample_judge_deepseek.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memarena.judges import agreement, cohen_kappa as _kappa, majority  # noqa: E402

HERE = Path(__file__).resolve().parent
LEAK_LABEL = "DISCLOSE_CORRECT"


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa (binary or multi-class); nan when undefined."""
    return _kappa(a, b)


DIM_LABELS = {
    "d1": "Conflict", "d2": "Anaphora", "d3": "Confabulation", "d4": "Permission", "d5": "Cloze",
    "d6": "Metadata", "d7": "QA", "d8": "Temporal", "d9": "Negation", "d10": "Counterfactual",
    "d11": "Exception",
    "D1": "D1 Cloze", "D2": "D2 Metadata", "D3": "D3 Factual QA", "D4": "D4 Cross-session",
    "D5": "D5 Abstention", "D6": "D6 Permission",
}


def normalize_dim(dim: str) -> str:
    if dim.startswith("d") and "_" in dim:
        return dim.split("_")[0]
    return dim


def _dim_sort_key(dim: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in dim if ch.isdigit())
    return (int(digits) if digits else 99, dim)


def _clean(value) -> Optional[object]:
    if value is None or value == "skip":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s in ("0", "1"):
        return int(s)
    return s.upper() or None


def reviewer_labels(blob: Mapping) -> Dict[str, Dict[str, object]]:
    """{reviewer: {id: value}} with skips removed (one pseudo-reviewer for a flat file)."""
    if isinstance(blob.get("reviewers"), Mapping):
        raw = blob["reviewers"]
    else:
        raw = {"labels": blob.get("labels", blob)}
    out: Dict[str, Dict[str, object]] = {}
    for rid, labels in raw.items():
        if not isinstance(labels, Mapping):
            continue
        out[str(rid)] = {str(k): v for k, v in ((k, _clean(v)) for k, v in labels.items()) if v is not None}
    return out


def consensus(reviewers: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    ids = set().union(*[set(v) for v in reviewers.values()]) if reviewers else set()
    out = {}
    for iid in ids:
        value = majority(r[iid] for r in reviewers.values() if iid in r)
        if value is not None:
            out[iid] = value
    return out


def _is_d6(row: Mapping) -> bool:
    return row.get("dim_kind") == "d6" or str(row.get("id", "")).startswith("d4_perm")


def metrics(pool: Mapping[str, Mapping], human: Mapping[str, object]) -> dict:
    """Judge-vs-human metrics for one set of human labels."""
    binary: List[Tuple[int, int]] = []
    per_dim: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    d6: List[Tuple[str, str]] = []
    unmatched = 0
    for iid, h in human.items():
        row = pool.get(iid)
        if row is None:
            unmatched += 1
            continue
        if _is_d6(row):
            if isinstance(h, str):
                d6.append((h, str(row.get("judge_label") or "").upper()))
            continue
        if h not in (0, 1):
            continue
        j = int(bool(row.get("judge_correct")))
        binary.append((int(h), j))
        per_dim[normalize_dim(str(row.get("dim") or "?"))].append((int(h), j))

    def bstats(ps: List[Tuple[int, int]]) -> dict:
        h = [p[0] for p in ps]
        j = [p[1] for p in ps]
        n = len(ps)
        return {"n": n, "agreement": agreement(h, j), "cohen_kappa": cohen_kappa(h, j),
                "human_pos_rate": sum(h) / n if n else float("nan"),
                "judge_pos_rate": sum(j) / n if n else float("nan")}

    h6 = [p[0] for p in d6]
    j6 = [p[1] for p in d6]
    hl = [int(x == LEAK_LABEL) for x in h6]
    jl = [int(x == LEAK_LABEL) for x in j6]
    return {
        "binary": bstats(binary),
        "per_dim": [dict(dim=d, name=DIM_LABELS.get(d, d), **bstats(per_dim[d]))
                    for d in sorted(per_dim, key=_dim_sort_key)],
        "d6": {"n": len(d6), "label_agreement": agreement(h6, j6), "label_kappa": cohen_kappa(h6, j6),
               "leak_agreement": agreement(hl, jl), "leak_kappa": cohen_kappa(hl, jl)},
        "unmatched": unmatched,
    }


def inter_annotator(reviewers: Mapping[str, Mapping[str, object]], pool: Mapping[str, Mapping]) -> List[dict]:
    rows = []
    for a, b in itertools.combinations(sorted(reviewers), 2):
        common = [i for i in reviewers[a] if i in reviewers[b] and i in pool]
        bin_ids = [i for i in common if not _is_d6(pool[i])]
        d6_ids = [i for i in common if _is_d6(pool[i])]
        rows.append({
            "pair": f"{a}-{b}",
            "binary_n": len(bin_ids),
            "binary_kappa": cohen_kappa([reviewers[a][i] for i in bin_ids], [reviewers[b][i] for i in bin_ids]),
            "d6_n": len(d6_ids),
            "d6_label_kappa": cohen_kappa([reviewers[a][i] for i in d6_ids], [reviewers[b][i] for i in d6_ids]),
        })
    return rows


def _f(x: float) -> str:
    return "--" if x != x else f"{x:.3f}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(HERE / "labels.json"))
    ap.add_argument("--pool", default=str(HERE / "sample.json"))
    ap.add_argument("--judge-name", help="judge named in the caption (default: the pool's judge_tag, else gpt-4o-mini)")
    ap.add_argument("--tex", help="default out/human_calibration/judge_human_<judge>.tex")
    ap.add_argument("--json", help="default out/human_calibration/judge_human_<judge>.json")
    args = ap.parse_args(argv)

    labels_blob = json.loads(Path(args.labels).read_text())
    pool_blob = json.loads(Path(args.pool).read_text())
    pool = {row["id"]: row for row in pool_blob.get("sample", [])}
    judge = args.judge_name or (pool_blob.get("metadata") or {}).get("judge_tag") or "gpt4omini"
    out_dir = REPO_ROOT / "out" / "human_calibration"
    tex_path = Path(args.tex) if args.tex else out_dir / f"judge_human_{judge}.tex"
    json_path = Path(args.json) if args.json else out_dir / f"judge_human_{judge}.json"

    reviewers = reviewer_labels(labels_blob)
    human = consensus(reviewers) if len(reviewers) > 1 else next(iter(reviewers.values()), {})
    overall = metrics(pool, human)
    if overall["binary"]["n"] == 0 and overall["d6"]["n"] == 0:
        print("No usable pairs; aborting.", file=sys.stderr)
        return 1
    per_reviewer = {rid: metrics(pool, labs) for rid, labs in reviewers.items()} if len(reviewers) > 1 else {}

    result = {
        "meta": {
            "labels_path": str(args.labels), "pool_path": str(args.pool), "judge": judge,
            "judge_model": (pool_blob.get("metadata") or {}).get("judge_model"),
            "reviewers": sorted(reviewers), "human_label": "strict majority" if len(reviewers) > 1 else "single",
            "consensus_items": len(human), "unmatched": overall["unmatched"],
        },
        "overall": overall["binary"],
        "per_dim": overall["per_dim"],
        "d6": overall["d6"],
        "per_reviewer": {rid: {"binary": m["binary"], "d6": m["d6"]} for rid, m in per_reviewer.items()},
        "inter_annotator": inter_annotator(reviewers, pool) if len(reviewers) > 1 else [],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2))

    b, d6 = overall["binary"], overall["d6"]
    lines = [
        "% Auto-generated by memarena/human_calibration/analyze_judge_human.py.",
        "% Source labels: " + Path(args.labels).name + "; judged pool: " + Path(args.pool).name,
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Dimension & $n$ & Agreement & $\\kappa$ \\\\",
        "\\midrule",
    ]
    for r in overall["per_dim"]:
        lines.append(f"{r['name']} & {r['n']} & {_f(r['agreement'])} & {_f(r['cohen_kappa'])} \\\\")
    lines += [
        "\\midrule",
        f"\\textbf{{Binary overall}} & {b['n']} & {_f(b['agreement'])} & {_f(b['cohen_kappa'])} \\\\",
        f"D6 5-label & {d6['n']} & {_f(d6['label_agreement'])} & {_f(d6['label_kappa'])} \\\\",
        f"D6 leak & {d6['n']} & {_f(d6['leak_agreement'])} & {_f(d6['leak_kappa'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n")

    print(f"[ok] judge={judge} reviewers={sorted(reviewers)} consensus items={len(human)}")
    print(f"[ok] binary n={b['n']} agreement={_f(b['agreement'])} kappa={_f(b['cohen_kappa'])}")
    print(f"[ok] D6 n={d6['n']} 5-label kappa={_f(d6['label_kappa'])} leak kappa={_f(d6['leak_kappa'])}")
    for r in overall["per_dim"]:
        print(f"  {r['dim']:>4}  n={r['n']:3d}  agree={_f(r['agreement'])}  kappa={_f(r['cohen_kappa'])}")
    for rid, m in per_reviewer.items():
        print(f"  reviewer {rid}: binary kappa={_f(m['binary']['cohen_kappa'])} "
              f"D6 kappa={_f(m['d6']['label_kappa'])}")
    print(f"[ok] wrote {tex_path}")
    print(f"[ok] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
