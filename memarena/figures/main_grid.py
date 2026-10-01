"""Per-cell statistics of the 5x5x3 main grid, shared by every main-grid table.

For each (backend, reader) the paper reports mean ± sample SD over seeds
s2/s3/s4 of per-seed scores:

  D1..D5  accuracy on the paper dimension (sub-dimensions pooled)
  D6      F1_PU (``eval.src.permission_metrics``)
  Rec     pooled accuracy over D1 + D2 items
  Rea     pooled accuracy over D3 + D4 items
  Trust   (D5 + D6) / 2
  Avg     (D1 + ... + D6) / 6

Records the answer stage never scored (``answer_scored`` false, or an
``LLM_ERROR`` / ``TEXT_SESSIONS_ERROR`` response) are excluded everywhere.
"""
from __future__ import annotations

import json
import os
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from eval.src.permission_metrics import (
    LEAK_LABEL,
    POPULATION_FACT,
    cell_scalars,
    load_instances,
    population_ids,
)
from memarena.figures.paper_data import PROJECT_ROOT, load_all_cells, results_source_label

BACKENDS = ("vanilla", "rag", "memobase", "memsearch", "oracle")
READERS = ("0_6b", "llama3b", "7b", "8b", "32b")
READER_LABEL = {"0_6b": "Qwen3-0.6B", "llama3b": "Llama-3.2-3B", "7b": "Mistral-7B", "8b": "Qwen3-8B", "32b": "Qwen3-32B"}
SEEDS = ("s2", "s3", "s4")

SUBDIM = {
    "d1": "d1_conflict", "d2": "d2_anaphora", "d3": "d3_confabulation", "d4": "d4_permission",
    "d5": "d5_cloze", "d6": "d6_metadata", "d7": "d7_qa", "d8": "d8_temporal", "d9": "d9_negation",
    "d10": "d10_counterfactual", "d11": "d11_exception",
}
PAPER_DIMS = {
    "D1": ("d5_cloze",),
    "D2": ("d6_metadata",),
    "D3": ("d7_qa", "d8_temporal", "d10_counterfactual"),
    "D4": ("d1_conflict", "d2_anaphora"),
    "D5": ("d3_confabulation",),
}
ACCURACY_DIMS = ("D1", "D2", "D3", "D4", "D5")
METRICS = ("D1", "D2", "D3", "D4", "D5", "D6", "Rec", "Rea", "Trust", "Avg")
_UNSCORED_PREFIXES = ("LLM_ERROR", "TEXT_SESSIONS_ERROR")


def default_instances_path() -> Path:
    root = Path(os.getenv("MEMARENA_DATASET_DIR") or PROJECT_ROOT / "data" / "benchmark")
    return root / "eval_instances" / "d4_permission.jsonl"


def _truthy(value) -> bool:
    return value is True or str(value) == "True"


def subdim_of(record: Mapping) -> str:
    return str(record.get("dimension") or SUBDIM.get(str(record.get("question_id", "")).split("_")[0], ""))


def scored(details: Iterable[Mapping]) -> List[Mapping]:
    out = []
    for rec in details:
        if rec.get("answer_scored") is False or str(rec.get("answer_scored")) == "False":
            continue
        raw = rec.get("raw_response")
        if isinstance(raw, str) and raw.startswith(_UNSCORED_PREFIXES):
            continue
        out.append(rec)
    return out


def seed_scores(details: Iterable[Mapping], d6_ids: Optional[Set[str]], leak: str) -> Dict[str, float]:
    """All paper metrics (in %) for one judged cell file."""
    records = scored(details)
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for rec in records:
        c = counts[subdim_of(rec)]
        c[0] += _truthy(rec.get("correct"))
        c[1] += 1

    def pooled(subdims: Tuple[str, ...]) -> float:
        hit = sum(counts[s][0] for s in subdims)
        n = sum(counts[s][1] for s in subdims)
        return 100.0 * hit / n if n else float("nan")

    out = {dim: pooled(subs) for dim, subs in PAPER_DIMS.items()}
    out["D6"] = 100.0 * cell_scalars(records, d6_ids, leak=leak).f1_pu
    out["Rec"] = pooled(PAPER_DIMS["D1"] + PAPER_DIMS["D2"])
    out["Rea"] = pooled(PAPER_DIMS["D3"] + PAPER_DIMS["D4"])
    out["Trust"] = (out["D5"] + out["D6"]) / 2
    out["Avg"] = sum(out[d] for d in ("D1", "D2", "D3", "D4", "D5", "D6")) / 6
    return out


@dataclass
class Stat:
    mean: float
    sd: float
    n: int


def main_grid_stats(
    population: str = POPULATION_FACT,
    leak: str = LEAK_LABEL,
    instances_path: Optional[Path] = None,
) -> Dict[Tuple[str, str], Dict[str, Stat]]:
    """(backend, reader) -> metric -> mean ± SD over seeds. Fails if a cell is missing."""
    grid = load_all_cells(include_ablation=False)
    missing = [(s, b, r) for b in BACKENDS for r in READERS for s in SEEDS if (s, b, r) not in grid]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} main-grid cell files missing from {results_source_label()} "
            f"(check MEMARENA_RESULTS_DIR, or MEMARENA_RUNS_DIR / MEMARENA_JUDGE): {missing[:5]}"
        )
    d6_ids = population_ids(load_instances(instances_path or default_instances_path()), population)

    per_seed: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    for s in SEEDS:
        for b in BACKENDS:
            for r in READERS:
                details = json.loads(Path(grid[(s, b, r)].source_path).read_text()).get("details", [])
                per_seed[(b, r)].append(seed_scores(details, d6_ids, leak))

    stats: Dict[Tuple[str, str], Dict[str, Stat]] = {}
    for key, rows in per_seed.items():
        stats[key] = {
            m: Stat(st.mean(x[m] for x in rows), st.stdev([x[m] for x in rows]) if len(rows) > 1 else 0.0, len(rows))
            for m in METRICS
        }
    return stats
