#!/usr/bin/env python3
"""Recompute ``metadata.query_timestamp`` of released eval instances.

The released instances were generated with the old rule: the start of the
FIRST evidence session, and no timestamp at all for items without evidence
(d3_confabulation abstain items; D6 predates its horizon rule). Time-filtered
backends (RAG cutoff, memory-cache ingest days) therefore could not see later
evidence. This script applies the current rule to existing files:

* every dimension except D6: ``MASim/ground_truth/query_time.py``
  (after the last evidence session; without evidence, after the ego's last
  session);
* D6 (d4_permission): ``d4_permission.horizon_timestamp`` (last ingested day
  + 0.99, after every session of the world).

Each rewritten instance also gets ``metadata.query_timestamp_rule``. Nothing
else changes: instances are rewritten line by line in the input order.

Usage::

    python scripts/fix_query_timestamps.py <run_dir> <out_dir>

``<run_dir>`` holds ``corpus_sessions.jsonl[.gz]``, ``ego_session_map.json``
and ``eval_instances/``; the fixed files go to ``<out_dir>/eval_instances/``.
``<out_dir>`` must differ from ``<run_dir>`` (never rewrite a release in
place). Prints, per task file, how many items changed, and how many moved
to another query day (the unit the time-filtered backends cut on).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.src.masim_loader import open_corpus_sessions  # noqa: E402
from MASim.ground_truth.d4_permission import horizon_timestamp  # noqa: E402
from MASim.ground_truth.query_time import query_timestamp, session_end_times  # noqa: E402

D6_FILE = "d4_permission.jsonl"


def _load_sessions(run_dir: Path) -> list:
    with open_corpus_sessions(run_dir) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return float(a) == float(b)


def fix_file(path: Path, out_path: Path, *, ego_map: dict, session_ends: dict, d6_horizon) -> dict:
    n = changed = was_missing = day_changed = 0
    rules: dict = {}
    lines_out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            lines_out.append(line)
            continue
        inst = json.loads(line)
        meta = inst.setdefault("metadata", {})
        old = meta.get("query_timestamp")
        if path.name == D6_FILE:
            new, rule = d6_horizon.value, d6_horizon.rule
        else:
            ego = str(inst.get("ego_agent_id") or "")
            new, rule = query_timestamp(meta.get("evidence_sessions") or [], ego_map.get(ego), session_ends)
        meta["query_timestamp"] = new
        meta["query_timestamp_rule"] = rule
        n += 1
        rule_key = "d6_horizon" if path.name == D6_FILE else rule
        rules[rule_key] = rules.get(rule_key, 0) + 1
        if not _same(old, new):
            changed += 1
            if old is None:
                was_missing += 1
            if old is None or int(float(old)) != int(float(new)):
                day_changed += 1
        lines_out.append(json.dumps(inst, ensure_ascii=False))
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return {"items": n, "changed": changed, "day_changed": day_changed, "was_missing": was_missing, "rules": rules}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == run_dir:
        sys.exit("out_dir must differ from run_dir: this script never rewrites a release in place")
    in_dir = run_dir / "eval_instances"
    files = sorted(in_dir.glob("d*.jsonl"))
    if not files:
        sys.exit(f"no eval_instances/d*.jsonl under {run_dir}")

    sessions = _load_sessions(run_dir)
    session_ends = session_end_times(sessions)
    ego_map = json.loads((run_dir / "ego_session_map.json").read_text(encoding="utf-8"))
    d6_horizon = horizon_timestamp(sessions)

    dst = out_dir / "eval_instances"
    dst.mkdir(parents=True, exist_ok=True)
    total = total_changed = total_day = 0
    print(f"{'task file':<26} {'items':>6} {'changed':>8} {'day_changed':>12} {'was_missing':>12}  rules")
    for f in files:
        rep = fix_file(f, dst / f.name, ego_map=ego_map, session_ends=session_ends, d6_horizon=d6_horizon)
        total += rep["items"]
        total_changed += rep["changed"]
        total_day += rep["day_changed"]
        rules = ", ".join(f"{k}={v}" for k, v in sorted(rep["rules"].items()))
        print(
            f"{f.name:<26} {rep['items']:>6} {rep['changed']:>8} {rep['day_changed']:>12} "
            f"{rep['was_missing']:>12}  {rules}"
        )
    print(f"{'TOTAL':<26} {total:>6} {total_changed:>8} {total_day:>12}")
    print("day_changed: the query day (int(query_timestamp), what the time-filtered backends use) moved or was missing")
    print(f"wrote {len(files)} files to {dst}")


if __name__ == "__main__":
    main()
