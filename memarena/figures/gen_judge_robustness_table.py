"""Generate the cross-family judge-robustness tables (app:judge-robustness).

Both arms re-score the same stratified subset of the main grid (5 readers x
5 backends, seed s2) with the paper's evidence-grounded scorer; only the judge
model differs: ``openai/gpt-4o-mini-2024-07-18`` (the paper's judge) vs
``anthropic/claude-sonnet-5`` (reasoning disabled). A control re-runs the
gpt-4o-mini judge against itself on five cells.

Outputs (<artifact root>/tables/):
  judge_robustness_rho_body.tex        per-reader backend-ranking Spearman + mean [95% CI]
  judge_robustness_agreement_body.tex  item-level agreement / Cohen's kappa (pooled, D6, control)

Inputs: $MEMARENA_RESULTS_DIR/out/g4_ranking/{sample_qids.json, sub/*.json}
(produced by docs/review/claude/g4_ranking/scripts/g4_run.sh). The D6 items
follow the population of ``eval.src.permission_metrics`` (default: only the
fact-bearing items); ``--population all`` reproduces the evidence in
docs/review/claude/g4_ranking/summary_v2.md.

    python -m memarena.figures.gen_judge_robustness_table
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from eval.src.permission_metrics import POPULATION_ALL, POPULATION_FACT, load_instances, population_ids
from memarena.figures.main_grid import default_instances_path
from memarena.figures.paper_data import RESULTS_ROOT
from memarena.figures.paths import table_path

G4 = RESULTS_ROOT / "out" / "g4_ranking"
READERS = ("0_6b", "llama3b", "7b", "8b", "32b")
READER_LABEL = {"0_6b": "Qwen3-0.6B", "llama3b": "Llama-3.2-3B", "7b": "Mistral-7B", "8b": "Qwen3-8B", "32b": "Qwen3-32B-AWQ"}
BACKENDS = ("vanilla", "oracle", "inmem", "memobase", "memsearch")
BASE, CONTRAST, RETEST = "gpt4omini1", "sonnet5g", "gpt4omini2"
CONTROL_CELLS = (("0_6b", "oracle"), ("llama3b", "inmem"), ("7b", "memsearch"), ("8b", "memobase"), ("32b", "vanilla"))


def verdicts(sample: set, reader: str, backend: str, tag: str) -> Dict[str, bool]:
    p = G4 / "sub" / f"evaluation_results_g4-{reader}-{backend}_s2_judge_{tag}.json"
    if not p.exists():
        return {}
    return {x["question_id"]: bool(x.get("correct")) for x in json.loads(p.read_text()).get("details", []) if x.get("question_id") in sample}


def spearman(a: List[float], b: List[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra)) * math.sqrt(sum((x - mb) ** 2 for x in rb))
    return num / den if den else float("nan")


def kappa(pairs: List[Tuple[bool, bool]]) -> Tuple[float, float]:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe)


def compute(population: str, instances: Path | None) -> dict:
    sample = set(json.loads((G4 / "sample_qids.json").read_text()))
    if population == POPULATION_FACT:
        keep = population_ids(load_instances(instances or default_instances_path()), POPULATION_FACT)
        sample = {q for q in sample if not q.startswith("d4") or q in keep}
    acc_g, acc_s, pooled, d6 = {}, {}, [], []
    for r in READERS:
        for b in BACKENDS:
            g, s = verdicts(sample, r, b, BASE), verdicts(sample, r, b, CONTRAST)
            common = sorted(q for q in sample if q in g and q in s)
            if not common:
                continue
            acc_g[(r, b)] = round(100 * sum(g[q] for q in common) / len(common), 1)
            acc_s[(r, b)] = round(100 * sum(s[q] for q in common) / len(common), 1)
            pooled += [(g[q], s[q]) for q in common]
            d6 += [(g[q], s[q]) for q in common if q.startswith("d4")]
    rhos = {r: spearman([acc_g[(r, b)] for b in BACKENDS], [acc_s[(r, b)] for b in BACKENDS]) for r in READERS}
    rng = random.Random(1)
    vals = list(rhos.values())
    boots = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(2000))
    ctrl = []
    for r, b in CONTROL_CELLS:
        a1, a2 = verdicts(sample, r, b, BASE), verdicts(sample, r, b, RETEST)
        ctrl += [(a1[q], a2[q]) for q in sample if q in a1 and q in a2]
    flips = sum(
        ((acc_g[(r, b)] - acc_g[(r, "vanilla")]) > 0) != ((acc_s[(r, b)] - acc_s[(r, "vanilla")]) > 0)
        for r in READERS for b in BACKENDS if b != "vanilla"
    )
    return {
        "n_qids": len(sample), "rhos": rhos, "mean_rho": sum(vals) / len(vals), "ci": (boots[50], boots[1950]),
        "pooled": (len(pooled), *kappa(pooled)), "d6": (len(d6), *kappa(d6)), "control": (len(ctrl), *kappa(ctrl)),
        "flips": flips,
    }


def render(res: dict) -> Dict[str, str]:
    rho = [f"{READER_LABEL[r]:<13s} & ${res['rhos'][r]:.3f}$ \\\\" for r in READERS]
    rho += [r"\midrule", f"Mean ({len(READERS)} readers) & ${res['mean_rho']:.3f}$ $[{res['ci'][0]:.2f}, {res['ci'][1]:.2f}]$ \\\\"]
    rows = []
    for label, key in (("Sonnet-5 vs.\\ \\texttt{gpt-4o-mini}, pooled", "pooled"),
                       ("Sonnet-5 vs.\\ \\texttt{gpt-4o-mini}, \\DSixID{}", "d6"),
                       ("\\texttt{gpt-4o-mini} re-run vs.\\ itself (control)", "control")):
        n, po, k = res[key]
        count = f"{n:,}".replace(",", "{,}")
        rows.append(f"{label} & ${count}$ & ${100 * po:.1f}\\%$ & ${k:.3f}$ \\\\")
    return {"judge_robustness_rho_body.tex": "\n".join(rho) + "\n", "judge_robustness_agreement_body.tex": "\n".join(rows) + "\n"}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the judge-robustness table bodies.")
    ap.add_argument("--population", choices=(POPULATION_FACT, POPULATION_ALL), default=POPULATION_FACT)
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    res = compute(args.population, args.instances)
    for name, body in render(res).items():
        out = (args.out_dir / name) if args.out_dir else table_path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
    print(f"[gen_judge_robustness_table] population={args.population} qids={res['n_qids']} mean_rho={res['mean_rho']:.3f} "
          f"CI=[{res['ci'][0]:.2f},{res['ci'][1]:.2f}] pooled={res['pooled'][1]:.3f}/{res['pooled'][2]:.3f} "
          f"d6={res['d6'][1]:.3f}/{res['d6'][2]:.3f} control={res['control'][1]:.3f}/{res['control'][2]:.3f} flips={res['flips']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
