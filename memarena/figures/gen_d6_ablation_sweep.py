"""Generate the body of tab:d6-ablation-sweep.

Each row pools the D6 records of a set of judged cells (all readers and seeds
of one variant) and reports ALLOW-DC (DISCLOSE_CORRECT rate on ALLOW), DENY-leak,
their gap, and F1_PU of the pooled P and U, on the D6 population of
``eval.src.permission_metrics`` (default: the 144 fact-bearing items).

Cell sources, relative to ``$MEMARENA_RESULTS_DIR/out``:
  Retrieval     ablations_l_extra_<reader>/eval_results_<seed>/<variant>/
  Reader scale  ablations_l_<reader>/eval_results_<seed>/{inmem_text_sessions,
                oracle_with_distractors, oracle_with_random_distractors}/
  Distractor    main-grid Oracle; ablations_l_<reader>/.../oracle_with_{,random_}distractors/
  Policy lever  ablations_l_h3_armB_<reader>/eval_results_s2/oracle/
  Identity      ablations_l_b1_L{0,1,2}/eval_results_s2/oracle/   (Qwen3-8B)

    python -m memarena.figures.gen_d6_ablation_sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

from eval.src.permission_metrics import (
    LEAK_FLAG,
    LEAK_LABEL,
    POPULATION_ALL,
    POPULATION_FACT,
    d6_records,
    f1_pu,
    is_allow,
    is_leak,
    load_instances,
    population_ids,
)
from memarena.figures.main_grid import READERS, SEEDS, default_instances_path, scored
from memarena.figures.paper_data import RESULTS_ROOT, load_all_cells
from memarena.figures.paths import table_path

OUT = RESULTS_ROOT / "out"
READER_LABEL = {"0_6b": "Qwen3-0.6B", "llama3b": "Llama-3.2-3B", "7b": "Mistral-7B", "8b": "Qwen3-8B", "32b": "Qwen3-32B"}


def _cells(pattern: str, readers: Sequence[str] = READERS, seeds: Sequence[str] = SEEDS) -> List[Path]:
    paths = []
    for r in readers:
        for s in seeds:
            paths += sorted(OUT.glob(pattern.format(r=r, s=s)))
    return paths


def _oracle_main(readers: Sequence[str] = READERS) -> List[Path]:
    grid = load_all_cells(include_ablation=False)
    return [Path(grid[(s, "oracle", r)].source_path) for r in readers for s in SEEDS]


def rows() -> List[Tuple[str, str, List[Path]]]:
    extra = "ablations_l_extra_{r}/eval_results_{s}/%s/evaluation_results_%s_{r}_{s}_judge_remote.json"
    base = "ablations_l_{r}/eval_results_{s}/%s/evaluation_results_%s_{r}_{s}_judge_remote.json"
    pool = ("inmem_text_sessions", "oracle_with_distractors", "oracle_with_random_distractors")
    out = [
        ("Retrieval", "dense E5", _cells(extra % ("dense_e5", "dense_e5"))),
        ("Retrieval", "dense BGE-M3", _cells(extra % ("dense_bge_m3", "dense_bge_m3"))),
        ("Retrieval", r"hybrid BM25$+$rerank", _cells(extra % ("hybrid_bm25rerank", "hybrid_bm25rerank"))),
        ("Retrieval", "temporal-prior", _cells(extra % ("temporal", "temporal"))),
    ]
    for r in READERS:
        out.append(("Reader scale", READER_LABEL[r], [p for v in pool for p in _cells(base % (v, v), readers=(r,))]))
    out += [
        ("Distractor", "no distractor (Oracle)", _oracle_main()),
        ("Distractor", r"$+$ curated distractor", _cells(base % ("oracle_with_distractors", "oracle_with_distractors"))),
        ("Distractor", r"$+$ random distractor", _cells(base % ("oracle_with_random_distractors", "oracle_with_random_distractors"))),
        ("Policy lever", "explicit DENY tag (prompt)", _cells("ablations_l_h3_armB_{r}/eval_results_{s}/oracle/evaluation_results_oracle_{r}_{s}_judge_remote.json", seeds=("s2",))),
    ]
    for lvl, name in (("L0", "no identity"), ("L1", "name only"), ("L2", r"name $+$ relation")):
        out.append(("Identity strength", f"\\textsc{{{lvl}}} ({name})",
                    _cells(f"ablations_l_b1_{lvl}/eval_results_{{s}}/oracle/evaluation_results_oracle_b1_{lvl}_{{r}}_{{s}}_judge_remote.json", readers=("8b",), seeds=("s2",))))
    return out


def pooled(paths: List[Path], keep, leak: str) -> Tuple[float, float, float]:
    allow_n = allow_dc = deny_n = deny_leak = 0
    for p in paths:
        for rec in d6_records(scored(json.loads(p.read_text()).get("details", [])), keep):
            if is_allow(rec):
                allow_n += 1
                allow_dc += rec.get("policy_category") == "DISCLOSE_CORRECT"
            else:
                deny_n += 1
                deny_leak += is_leak(rec, leak)
    u, lk = allow_dc / allow_n, deny_leak / deny_n
    return 100 * u, 100 * lk, 100 * f1_pu(1 - lk, u)


def render_body(table, keep, leak: str) -> str:
    lines, last = [], None
    groups = {}
    for lever, _, _ in table:
        groups[lever] = groups.get(lever, 0) + 1
    for lever, variant, paths in table:
        if not paths:
            raise FileNotFoundError(f"no cells for {lever} / {variant} under {OUT}")
        if lever != last and last is not None:
            lines.append(r"\midrule")
        u, lk, f1 = pooled(paths, keep, leak)
        head = "" if lever == last else (f"\\multirow{{{groups[lever]}}}{{*}}{{{lever}}}" if groups[lever] > 1 else lever)
        lines.append(f"{head} & {variant} & {len(paths)} & {u:.1f} & {lk:.1f} & {lk - u:+.1f} & {f1:.1f} \\\\")
        last = lever
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the D6 ablation sweep table body.")
    ap.add_argument("--population", choices=(POPULATION_FACT, POPULATION_ALL), default=POPULATION_FACT)
    ap.add_argument("--leak", choices=(LEAK_LABEL, LEAK_FLAG), default=LEAK_LABEL)
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    keep = population_ids(load_instances(args.instances or default_instances_path()), args.population)
    out = args.out or table_path("d6_ablation_sweep_body.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_body(rows(), keep, args.leak), encoding="utf-8")
    print(f"[gen_d6_ablation_sweep] population={args.population} leak={args.leak} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
