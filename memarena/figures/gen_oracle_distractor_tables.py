"""Generate the bodies of tab:review4-oracle-dist and tab:review4-oracle-dist-random.

Scores are the paper's six-dimension Avg (``memarena.figures.main_grid``) per
seed s2/s3/s4 for three Oracle contexts:
  plain Oracle         main-grid Oracle cells
  + recency distractors  out/ablations_l_<reader>/eval_results_<seed>/oracle_with_distractors/
  + random distractors   out/ablations_l_<reader>/eval_results_<seed>/oracle_with_random_distractors/

    python -m memarena.figures.gen_oracle_distractor_tables
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict, List

from eval.src.permission_metrics import LEAK_FLAG, LEAK_LABEL, POPULATION_ALL, POPULATION_FACT, load_instances, population_ids
from memarena.figures.main_grid import READERS, SEEDS, default_instances_path, seed_scores
from memarena.figures.paper_data import RESULTS_ROOT, load_all_cells
from memarena.figures.paths import table_path

READER_LABEL = {"0_6b": "Qwen3-0.6B", "llama3b": "Llama-3.2-3B", "7b": "Mistral-7B", "8b": "Qwen3-8B", "32b": "Qwen3-32B-AWQ"}
VARIANT = "ablations_l_{r}/eval_results_{s}/{v}/evaluation_results_{v}_{r}_{s}_judge_remote.json"


def collect(population: str, leak: str, instances: Path | None, metric: str = "Avg") -> Dict[str, Dict[str, List[float]]]:
    keep = population_ids(load_instances(instances or default_instances_path()), population)
    grid = load_all_cells(include_ablation=False)

    def score(path: Path) -> float:
        return seed_scores(json.loads(path.read_text())["details"], keep, leak)[metric]

    out = {}
    for r in READERS:
        out[r] = {
            "plain": [score(Path(grid[(s, "oracle", r)].source_path)) for s in SEEDS],
            "recency": [score(RESULTS_ROOT / "out" / VARIANT.format(r=r, s=s, v="oracle_with_distractors")) for s in SEEDS],
            "random": [score(RESULTS_ROOT / "out" / VARIANT.format(r=r, s=s, v="oracle_with_random_distractors")) for s in SEEDS],
        }
    return out


def _ms(xs: List[float]) -> str:
    return f"${st.mean(xs):.1f} \\pm {st.stdev(xs):.1f}$"


def render(data) -> Dict[str, str]:
    dist, rand = [], []
    for r in READERS:
        d = data[r]
        dist.append(
            f"{READER_LABEL[r]:<13s} & {_ms(d['plain'])} & {_ms(d['recency'])} & "
            f"${st.mean(d['recency']) - st.mean(d['plain']):+.1f}$ \\\\"
        )
        rand.append(
            f"{READER_LABEL[r]:<13s} & ${st.mean(d['recency']):.1f}$ & ${st.mean(d['random']):.1f}$ & "
            f"${st.mean(d['random']) - st.mean(d['recency']):+.1f}$ \\\\"
        )
    return {"review4_oracle_dist_body.tex": "\n".join(dist) + "\n", "review4_oracle_dist_random_body.tex": "\n".join(rand) + "\n"}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the Oracle-with-distractors table bodies.")
    ap.add_argument("--population", choices=(POPULATION_FACT, POPULATION_ALL), default=POPULATION_FACT)
    ap.add_argument("--leak", choices=(LEAK_LABEL, LEAK_FLAG), default=LEAK_LABEL)
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    for name, body in render(collect(args.population, args.leak, args.instances)).items():
        out = (args.out_dir / name) if args.out_dir else table_path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        print(f"[gen_oracle_distractor_tables] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
