"""Recompute every main-grid D6 number from per-item judge records.

Reads the 5x5x3 main grid through ``memarena.figures.paper_data.load_all_cells``
and scores each cell with ``eval.src.permission_metrics`` on two populations:

  all   -- all 200 items (the pooled number of earlier drafts)
  fact  -- the 144 items that carry a protected fact (camera-ready headline)

Writes ``per_cell.json`` (every cell x population) and ``summary.md`` (the
aggregates the paper quotes) to ``--out-dir``. With ``--snapshot`` it also
checks the ``all`` population against a legacy ``per_cell_full.json``.

    python3 scripts/d6_recompute.py \
        --instances data/benchmark/eval_instances/d4_permission.jsonl \
        --snapshot out/d6_paired/per_cell_full.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.src.permission_metrics import (  # noqa: E402
    LABELS,
    LEAK_FLAG,
    POPULATIONS,
    cell_scalars,
    d6_records,
    is_allow,
    is_leak,
    load_instances,
    population_ids,
    probe_family,
)
from memarena.figures.paper_data import load_all_cells  # noqa: E402

BACKENDS = ("vanilla", "rag", "memobase", "memsearch", "oracle")
DEPLOYABLE = ("vanilla", "rag", "memobase", "memsearch")
READERS = ("0_6b", "llama3b", "7b", "8b", "32b")
READER_NAME = {"0_6b": "Qwen3-0.6B", "llama3b": "Llama-3.2-3B", "7b": "Mistral-7B", "8b": "Qwen3-8B", "32b": "Qwen3-32B"}


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instances", type=Path, default=REPO / "data/benchmark/eval_instances/d4_permission.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "out/d6_recompute")
    ap.add_argument("--snapshot", type=Path, default=None, help="legacy per_cell_full.json to cross-check the 'all' population")
    args = ap.parse_args()

    instances = load_instances(args.instances)
    pops = {p: population_ids(instances, p) for p in POPULATIONS}
    grid = load_all_cells(include_ablation=False)

    per_cell: dict[str, dict] = {}
    records_by_cell: dict[tuple[str, str, str], list] = {}
    for (seed, backend, reader), cell in sorted(grid.items()):
        details = json.loads(Path(cell.source_path).read_text()).get("details", [])
        records_by_cell[(seed, backend, reader)] = d6_records(details)
        per_cell[f"{seed}/{backend}/{reader}"] = {
            "source": str(Path(cell.source_path).relative_to(REPO)) if str(cell.source_path).startswith(str(REPO)) else str(cell.source_path),
            **{p: cell_scalars(details, pops[p]).as_dict() for p in POPULATIONS},
            "all_flag": cell_scalars(details, pops["all"], leak=LEAK_FLAG).as_dict(),
        }

    lines = ["# D6 recompute (main grid)", ""]
    lines.append(f"- instances: `{args.instances}` ({len(instances)} items; fact-defined {len(pops['fact'])})")
    lines.append(f"- cells: {len(per_cell)}")

    # Cross-check against the legacy snapshot on the pooled population.
    if args.snapshot and args.snapshot.exists():
        snap = json.loads(args.snapshot.read_text())
        match, mismatch = 0, []
        for key, row in per_cell.items():
            sv = snap.get(key, {}).get("scalars", {})
            ok = abs(row["all_flag"]["utility"] - sv.get("comply_allow_tp", -9)) < 1e-9 and abs(row["all_flag"]["leak_deny"] - sv.get("leak_deny_tp", -9)) < 1e-9
            match += ok
            if not ok:
                mismatch.append(key)
        lines.append(f"- snapshot `{args.snapshot.name}` (flag leak, all 200): {match}/{len(per_cell)} cells reproduce exactly; differing: {', '.join(mismatch) or 'none'}")

    for pop in POPULATIONS:
        lines += ["", f"## Population `{pop}`", "", "| backend | reader | seeds | F1_PU mean ± sd | P | U |", "|---|---|---|---|---|---|"]
        cell_means: dict[tuple[str, str], float] = {}
        for b in BACKENDS:
            for r in READERS:
                rows = [per_cell[k][pop] for k in per_cell if k.split("/")[1:] == [b, r]]
                f_m, f_sd = _mean_sd([100 * x["f1_pu"] for x in rows])
                p_m, _ = _mean_sd([x["privacy"] for x in rows])
                u_m, _ = _mean_sd([x["utility"] for x in rows])
                cell_means[(b, r)] = f_m
                lines.append(f"| {b} | {READER_NAME[r]} | {len(rows)} | {f_m:.1f} ± {f_sd:.1f} | {p_m:.3f} | {u_m:.3f} |")
        dep = {k: v for k, v in cell_means.items() if k[0] in DEPLOYABLE}
        best = max(dep, key=dep.get)
        orc = [cell_means[("oracle", r)] for r in READERS]
        lines += ["", f"- best deployable cell: {best[0]} x {READER_NAME[best[1]]} = {dep[best]:.1f}",
                  f"- Oracle per-reader F1_PU: min {min(orc):.1f}, max {max(orc):.1f}, mean {st.mean(orc):.1f}"]

        # DENY response composition, Oracle vs deployable, pooled over records.
        lines += ["", "| DENY composition | records | leak | " + " | ".join(LABELS) + " |", "|---|---|---|" + "---|" * len(LABELS)]
        for name, group in (("Oracle", ("oracle",)), ("deployable", DEPLOYABLE)):
            recs = [rec for (s, b, r), rs in records_by_cell.items() if b in group for rec in rs
                    if not is_allow(rec) and str(rec.get("question_id")) in pops[pop]]
            n = len(recs)
            shares = [100 * sum(x.get("policy_category") == lab for x in recs) / n for lab in LABELS]
            lines.append(f"| {name} | {n} | {100 * sum(is_leak(x) for x in recs) / n:.1f} | " + " | ".join(f"{v:.1f}" for v in shares) + " |")

        # Oracle DENY leak by generator family.
        fam = defaultdict(list)
        for (s, b, r), rs in records_by_cell.items():
            if b != "oracle":
                continue
            for rec in rs:
                qid = str(rec.get("question_id"))
                if is_allow(rec) or qid not in pops[pop]:
                    continue
                fam[probe_family(instances[qid])].append((r, is_leak(rec)))
        lines += ["", "| Oracle DENY family | items x cells | leak % | per-reader range |", "|---|---|---|---|"]
        for name, vals in sorted(fam.items()):
            per_r = [100 * st.mean([lk for rr, lk in vals if rr == r]) for r in READERS if any(rr == r for rr, _ in vals)]
            lines.append(f"| {name} | {len(vals)} | {100 * st.mean([lk for _, lk in vals]):.1f} | [{min(per_r):.1f}, {max(per_r):.1f}] |")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_cell.json").write_text(json.dumps(per_cell, indent=2), encoding="utf-8")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
