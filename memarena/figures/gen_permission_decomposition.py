"""Generate the D6 leak/miss decomposition tables (app:dim-permission).

Three table bodies, all on the D6 population of ``eval.src.permission_metrics``
(default: the 144 fact-bearing items; leak = DENY item labelled DISCLOSE_CORRECT):

  d6_deny_composition_body.tex  DENY responses of the Oracle diagnostic vs. the four
                                deployable backends, pooled over readers x seeds:
                                leak / access-policy refusal / don't know / other
  d6_family_leak_body.tex       Oracle DENY leak by generator family, with the
                                per-reader range
  d6_bm25_audit_body.tex        can the paper's BM25 retriever find the DENY evidence?
                                (per-ego rank_bm25 index, one document per session,
                                B.15 tokenizer, hardened query; no model involved)

    python -m memarena.figures.gen_permission_decomposition
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from eval.src.permission_metrics import (
    POPULATION_ALL,
    POPULATION_FACT,
    d6_records,
    is_allow,
    is_leak,
    load_instances,
    population_ids,
    probe_family,
)
from memarena.figures.main_grid import READERS, default_instances_path, scored
from memarena.figures.paper_data import load_all_cells
from memarena.figures.paths import table_path

DEPLOYABLE = ("vanilla", "rag", "memobase", "memsearch")
FAMILY_LABEL = {
    "autonomous_privacy": "no explicit instruction (autonomous privacy)",
    "permission_compliance": "explicit sharing level stated in context",
    "anonymous_querier": "requester anonymous",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
K = 10


def _deny_records(keep) -> Dict[tuple, List[dict]]:
    grid = load_all_cells(include_ablation=False)
    out = {}
    for key, cell in grid.items():
        recs = d6_records(scored(json.loads(Path(cell.source_path).read_text()).get("details", [])), keep)
        out[key] = [r for r in recs if not is_allow(r)]
    return out


def composition(deny: Dict[tuple, List[dict]]) -> Dict[str, dict]:
    out = {}
    for name, backends in (("oracle", ("oracle",)), ("deployable", DEPLOYABLE)):
        recs = [r for (s, b, m), rs in deny.items() if b in backends for r in rs]
        cells = sum(1 for (s, b, m) in deny if b in backends)
        cat = [r.get("policy_category") for r in recs]
        n = len(recs)
        out[name] = {
            "records": n, "cells": cells,
            "leak": 100 * sum(is_leak(r) for r in recs) / n,
            "refuse": 100 * cat.count("REFUSE") / n,
            "dont_know": 100 * cat.count("DONT_KNOW") / n,
            "other": 100 * (cat.count("OTHER") + cat.count("DISCLOSE_WRONG")) / n,
        }
    return out


def family_leak(deny: Dict[tuple, List[dict]], instances: dict) -> Dict[str, dict]:
    fam = defaultdict(lambda: defaultdict(list))
    for (s, b, m), rs in deny.items():
        if b != "oracle":
            continue
        for r in rs:
            fam[probe_family(instances[r["question_id"]])][m].append(is_leak(r))
    out = {}
    for f, by_reader in fam.items():
        allv = [x for v in by_reader.values() for x in v]
        per = [100 * st.mean(by_reader[r]) for r in READERS if by_reader.get(r)]
        out[f] = {"n": len(allv), "leak": 100 * st.mean(allv), "lo": min(per), "hi": max(per)}
    return out


def bm25_audit(instances: dict, keep: set, dataset_dir: Path) -> dict:
    from rank_bm25 import BM25Okapi

    sessions = {}
    with gzip.open(dataset_dir / "corpus_sessions.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            sessions[r["session_id"]] = " ".join(t.get("text", "") for t in r.get("turns", []))
    ego_map = json.loads((dataset_dir / "ego_session_map.json").read_text())
    tok = lambda s: [w.lower() for w in TOKEN_RE.findall(s or "")]  # noqa: E731
    index, ranks, hits, outscored, n = {}, [], 0, 0, 0
    for qid, inst in instances.items():
        gt = inst.get("ground_truth") or {}
        if qid not in keep or gt.get("expected_disclosure") is not False:
            continue
        ego, gold = inst.get("ego_agent_id"), set(inst.get("evidence_session_ids") or [])
        if not (ego and gold):
            continue
        if ego not in index:
            mem = [s for s in ego_map.get(ego, []) if s in sessions]
            index[ego] = (mem, BM25Okapi([tok(sessions[s]) for s in mem]))
        mem, bm = index[ego]
        sc = bm.get_scores(tok(inst.get("query", "")))
        order = sorted(range(len(mem)), key=lambda i: -sc[i])
        pos = next((p for p, i in enumerate(order, 1) if mem[i] in gold), None)
        n += 1
        if pos:
            ranks.append(pos)
            hits += pos <= K
        gi = [i for i in range(len(mem)) if mem[i] in gold]
        if gi:
            best_gold = max(sc[i] for i in gi)
            best_other = max((sc[i] for i in range(len(mem)) if mem[i] not in gold), default=0.0)
            outscored += best_other > best_gold
    return {"n": n, "miss_at_k": 100 * (n - hits) / n, "median_rank": st.median(ranks),
            "pool_mean": st.mean(len(v[0]) for v in index.values()), "outscored": 100 * outscored / n}


def render(comp, fam, audit) -> Dict[str, str]:
    c_rows = []
    for name, label in (("oracle", "Oracle (evidence in context)"), ("deployable", "four deployable backends")):
        c = comp[name]
        c_rows.append(f"{label} & ${c['records']:,}$ ({c['cells']} cells) & ${c['leak']:.1f}$ & ${c['refuse']:.1f}$ & "
                      f"${c['dont_know']:.1f}$ & ${c['other']:.1f}$ \\\\".replace(",", "{,}", 1))
    f_rows = []
    for f in ("autonomous_privacy", "permission_compliance", "anonymous_querier"):
        if f in fam:
            v = fam[f]
            f_rows.append(f"{FAMILY_LABEL[f]} & ${v['n']}$ & ${v['leak']:.1f}$ & $[{v['lo']:.1f}, {v['hi']:.1f}]$ \\\\")
    a = audit
    a_rows = [f"DENY items & ${a['n']}$ \\\\",
              f"gold session outside the top {K} & ${a['miss_at_k']:.1f}\\%$ \\\\",
              f"median rank of the gold session & ${a['median_rank']:g}$ (pool: ${a['pool_mean']:.0f}$ sessions per ego) \\\\",
              f"a distractor outscores every gold session & ${a['outscored']:.1f}\\%$ \\\\"]
    return {"d6_deny_composition_body.tex": "\n".join(c_rows) + "\n",
            "d6_family_leak_body.tex": "\n".join(f_rows) + "\n",
            "d6_bm25_audit_body.tex": "\n".join(a_rows) + "\n"}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the D6 leak/miss decomposition table bodies.")
    ap.add_argument("--population", choices=(POPULATION_FACT, POPULATION_ALL), default=POPULATION_FACT)
    ap.add_argument("--instances", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    inst_path = args.instances or default_instances_path()
    instances = load_instances(inst_path)
    keep = population_ids(instances, args.population)
    deny = _deny_records(keep)
    comp, fam = composition(deny), family_leak(deny, instances)
    audit = bm25_audit(instances, keep, inst_path.parent.parent)
    for name, body in render(comp, fam, audit).items():
        out = (args.out_dir / name) if args.out_dir else table_path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
    print(f"[gen_permission_decomposition] population={args.population} "
          f"oracle={ {k: round(v, 1) if isinstance(v, float) else v for k, v in comp['oracle'].items()} } "
          f"deployable={ {k: round(v, 1) if isinstance(v, float) else v for k, v in comp['deployable'].items()} } "
          f"bm25={ {k: round(v, 1) if isinstance(v, float) else v for k, v in audit.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
