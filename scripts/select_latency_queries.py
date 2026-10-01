"""Select 90 query instances for latency replay (9 sub-dims × 10 each).

Selection rule (independent of cell — same 90 used for all 25 cells):
  * Group all eval_instances by SUB-DIM (the 9 d*_*.jsonl files in
    data/benchmark/eval_instances/).
  * Each instance gets a query_time:
      - prefer metadata.query_timestamp (simulator-emitted)
      - else fall back to max(end_time over evidence_session_ids)
        (questions must be asked after all referenced evidence)
  * Within each sub-dim, sort by query_time DESC, take top 10.
  * Total = 9 × 10 = 90.

Output:
  out/latency_selection/queries_s2.json  -- per-item entries with
    {instance_id, paper_dim, sub_dim, query_time, query_time_source,
     ego_agent_id, difficulty, query, ...}
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_INSTANCES = REPO / "data" / "benchmark" / "eval_instances"
SESSIONS_PATH = REPO / "data" / "benchmark" / "corpus_sessions.jsonl.gz"
OUT_DIR = REPO / "out" / "latency_selection"
OUT_FILE = OUT_DIR / "queries_s2.json"

PAPER_DIM_FROM_SUB = {
    "d5_cloze":          "D1",
    "d6_metadata":       "D2",
    "d7_qa":             "D3",
    "d8_temporal":       "D3",
    "d10_counterfactual":"D3",
    "d1_conflict":       "D4",
    "d2_anaphora":       "D4",
    "d3_confabulation":  "D5",
    "d4_permission":     "D6",
}
PAPER_DIM_LABEL = {
    "D1": "Recall: cloze",
    "D2": "Recall: metadata",
    "D3": "Reasoning: factual QA",
    "D4": "Reasoning: cross-session",
    "D5": "Trust: abstention",
    "D6": "Trust: permission",
}
# 9 sub-dims, ordered by the d-prefix shown in eval_instances filenames.
SUB_DIM_ORDER = [
    "d1_conflict", "d2_anaphora", "d3_confabulation", "d4_permission",
    "d5_cloze", "d6_metadata", "d7_qa", "d8_temporal", "d10_counterfactual",
]
PER_SUB_DIM = 10  # latest N per sub-dim
TOTAL_TARGET = PER_SUB_DIM * len(SUB_DIM_ORDER)  # 90


def load_session_end_times():
    end_t = {}
    with gzip.open(SESSIONS_PATH, "rt") as f:
        for line in f:
            s = json.loads(line)
            et = s.get("end_time")
            if et is not None:
                end_t[s["session_id"]] = et
    return end_t


def load_instances(sess_end):
    by_sub = defaultdict(list)
    sub_counts = Counter()
    src_counts = Counter()
    skipped = 0
    total = 0
    for sub_dim in SUB_DIM_ORDER:
        pd = PAPER_DIM_FROM_SUB[sub_dim]
        path = EVAL_INSTANCES / f"{sub_dim}.jsonl"
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            md = r.get("metadata") or {}
            qt = md.get("query_timestamp")
            qt_src = "metadata"
            if qt is None:
                ev_ids = r.get("evidence_session_ids") or []
                ev_ts = [sess_end.get(s) for s in ev_ids if sess_end.get(s) is not None]
                if ev_ts:
                    qt = max(ev_ts)
                    qt_src = "max_evidence_end"
            if qt is None:
                skipped += 1
                continue
            by_sub[sub_dim].append({
                "instance_id":       r["instance_id"],
                "paper_dim":         pd,
                "sub_dim":           sub_dim,
                "query_time":        qt,
                "query_time_source": qt_src,
                "ego_agent_id":      r.get("ego_agent_id"),
                "difficulty":        r.get("difficulty"),
                "query":             r.get("query"),
                "asker_agent_id":    r.get("asker_agent_id"),
                "answerer_agent_id": r.get("answerer_agent_id"),
                "evidence_session_ids": r.get("evidence_session_ids"),
            })
            sub_counts[sub_dim] += 1
            src_counts[qt_src] += 1
            total += 1
    return by_sub, sub_counts, src_counts, total, skipped


def select(by_sub):
    selected = []
    per_sub_stats = {}
    for sub_dim in SUB_DIM_ORDER:
        items = sorted(by_sub.get(sub_dim, []), key=lambda x: x["query_time"], reverse=True)
        chosen = items[:PER_SUB_DIM]
        selected.extend(chosen)
        if items:
            qts = [x["query_time"] for x in items]
            chosen_qts = [x["query_time"] for x in chosen]
            per_sub_stats[sub_dim] = {
                "paper_dim": PAPER_DIM_FROM_SUB[sub_dim],
                "pool": len(items),
                "taken": len(chosen),
                "pool_qt_min":   min(qts),
                "pool_qt_max":   max(qts),
                "cutoff_qt":     chosen_qts[-1] if chosen_qts else None,
                "difficulty_breakdown": Counter((x["difficulty"] or "?") for x in chosen),
                "qt_source_breakdown": Counter(x["query_time_source"] for x in chosen),
                "n_unique_egos": len({x["ego_agent_id"] for x in chosen}),
                "top_egos": Counter(x["ego_agent_id"] for x in chosen).most_common(3),
            }
    return selected, per_sub_stats


def fmt_qt(qt):
    """Render fractional-day timestamp as days+hours."""
    if qt is None:
        return "    -"
    days = int(qt)
    hours = (qt - days) * 24
    return f"d{days:>2d}+{hours:>5.2f}h"


def print_stats(total, sub_counts, src_counts, skipped, per_sub_stats, selected):
    print()
    print(f"# Pool: {total} eval instances across {len(sub_counts)} sub-dims  (skipped {skipped} with no qt + no evidence sessions)")
    print(f"# qt source: metadata={src_counts.get('metadata',0)}  derived(max_evidence_end)={src_counts.get('max_evidence_end',0)}")
    print(f"# Selected: {len(selected)} (target {TOTAL_TARGET} = {len(SUB_DIM_ORDER)} sub-dims × {PER_SUB_DIM})")
    print()
    print(f"{'Sub-dim':<22} {'Pdim':<5} {'Pool':>6} {'Take':>5}  {'Pool qt range':<32} {'Cutoff qt':<14} {'qt-src':<24} {'Difficulty':<24} {'Egos':>5}")
    print("-" * 165)
    for sub_dim in SUB_DIM_ORDER:
        s = per_sub_stats.get(sub_dim)
        if not s:
            print(f"{sub_dim:<22}  no data")
            continue
        diff_b = ", ".join(f"{k}={v}" for k, v in sorted(s["difficulty_breakdown"].items()))
        src_b  = ", ".join(f"{k}={v}" for k, v in sorted(s["qt_source_breakdown"].items()))
        ego_top = ", ".join(f"{e}={n}" for e, n in s["top_egos"]) if s["top_egos"] else "-"
        qt_range = f"[{fmt_qt(s['pool_qt_min'])}..{fmt_qt(s['pool_qt_max'])}]"
        print(f"{sub_dim:<22} {s['paper_dim']:<5} {s['pool']:>6} {s['taken']:>5}  {qt_range:<32} {fmt_qt(s['cutoff_qt']):<14} {src_b:<24} {diff_b:<24} {s['n_unique_egos']:>5}")
        print(f"      top egos in selection: {ego_top}")
    print()
    sel_qts = [x["query_time"] for x in selected]
    print(f"# Selection global qt range: [{fmt_qt(min(sel_qts))} .. {fmt_qt(max(sel_qts))}]  (sim spans {max(sel_qts)-min(sel_qts):.2f} days)")
    # Paper-dim roll-up
    pd_counts = Counter(x["paper_dim"] for x in selected)
    print(f"# Paper-dim roll-up:  " + "  ".join(f"{pd}={pd_counts.get(pd,0)}" for pd in ['D1','D2','D3','D4','D5','D6']))


def main():
    sess_end = load_session_end_times()
    print(f"loaded {len(sess_end)} sessions with end_time")
    by_sub, sub_counts, src_counts, total, skipped = load_instances(sess_end)
    selected, per_sub_stats = select(by_sub)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial":    "s2",
        "n_target": TOTAL_TARGET,
        "n_selected": len(selected),
        "per_sub_dim": PER_SUB_DIM,
        "sub_dim_order": SUB_DIM_ORDER,
        "items":    selected,
    }
    OUT_FILE.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_FILE}")
    print_stats(total, sub_counts, src_counts, skipped, per_sub_stats, selected)


if __name__ == "__main__":
    main()
