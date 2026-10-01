#!/usr/bin/env python3
"""CPU-2: Context-overlap scoring for D4 (permission) instances.

For each D4 instance × backend, determines whether the source session
(containing the private fact) was in the model's actual context:
  - Vanilla: was the session within the ~8K truncation window?
  - RAG: did BM25 retrieve this session?
  - Oracle: always yes (source session is in evidence_session_ids)

Produces two sub-metrics:
  D4-active:  accuracy among instances where model had access
  D4-passive: accuracy among instances where model had no access
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

RUN_DIR = Path("MASim/runs/l_20260408_111046")

# ── Load eval instances to get source_session_id per instance ───────────────
source_sessions = {}  # instance_id -> source_session_id
with open(RUN_DIR / "eval_instances" / "d4_permission.jsonl") as f:
    for line in f:
        inst = json.loads(line)
        sid = inst["ground_truth"].get("source_session_id", "")
        if sid:
            source_sessions[inst["instance_id"]] = sid

print(f"D4 instances with source_session_id: {len(source_sessions)}")

# ── Load corpus sessions for vanilla context window check ───────────────────
# Build per-ego session list sorted by time
ego_sessions = defaultdict(list)  # ego_agent -> [(start_time, session_id)]
with open(RUN_DIR / "corpus_sessions.jsonl") as f:
    for line in f:
        sess = json.loads(line)
        sid = sess["session_id"]
        t = sess.get("start_time", 0)
        for p in sess.get("participants", []):
            ego_sessions[p].append((t, sid))

for ego in ego_sessions:
    ego_sessions[ego].sort()

# Estimate vanilla context window: last N sessions that fit in ~16K tokens
# Average ~800 tokens per session → ~20 sessions fit in 16K
VANILLA_WINDOW_SESSIONS = 20


def vanilla_has_access(instance_id: str, ego_agent: str, source_sid: str) -> bool:
    """Check if source_session is within the vanilla truncation window."""
    sessions = ego_sessions.get(ego_agent, [])
    if not sessions:
        return False
    # Get the last VANILLA_WINDOW_SESSIONS sessions for this ego
    recent_sids = {sid for _, sid in sessions[-VANILLA_WINDOW_SESSIONS:]}
    return source_sid in recent_sids


def rag_has_access(search_results: dict, instance_id: str, source_sid: str) -> bool:
    """Check if BM25 retrieved the source session for this instance."""
    # search_results format: list of {question_id, retrieved_sessions: [...]}
    for item in search_results:
        if item.get("question_id") == instance_id or item.get("id") == instance_id:
            retrieved = item.get("retrieved_sessions", [])
            # May be list of session_ids or list of dicts with session_id
            for r in retrieved:
                if isinstance(r, str) and r == source_sid:
                    return True
                if isinstance(r, dict) and r.get("session_id") == source_sid:
                    return True
            return False
    return False


# ── Process each backend ───────────────────────────────────────────────────
results = {}

for subdir, backend_name in [("vanilla", "vanilla"), ("oracle", "oracle"), ("inmem", "rag")]:
    eval_dir = RUN_DIR / "eval_results" / subdir
    if not eval_dir.is_dir():
        continue

    # Load search results for RAG (if available)
    search_data = []
    if backend_name == "rag":
        for sf in eval_dir.glob("search_results_*.json"):
            try:
                search_data = json.load(open(sf))
                if isinstance(search_data, dict):
                    search_data = search_data.get("results", [])
                break
            except Exception:
                pass

    for eval_file in sorted(eval_dir.glob("evaluation_results_*.json")):
        trial_name = eval_file.stem.replace("evaluation_results_", "")
        data = json.load(open(eval_file))
        details = data.get("details", [])

        d4 = [d for d in details if d.get("dimension") == "d4_permission"]
        if not d4:
            continue

        active_correct = 0
        active_total = 0
        passive_correct = 0
        passive_total = 0

        for d in d4:
            iid = d.get("instance_id", "")
            source_sid = source_sessions.get(iid, "")
            ego = d.get("ground_truth", {}).get("query_agent", "")
            correct = d.get("judge_correct", False)

            if not source_sid:
                continue

            # Determine context access
            if backend_name == "oracle":
                has_access = True  # Oracle always has evidence
            elif backend_name == "vanilla":
                has_access = vanilla_has_access(iid, ego, source_sid)
            elif backend_name == "rag":
                has_access = rag_has_access(search_data, iid, source_sid)
            else:
                has_access = False

            if has_access:
                active_total += 1
                if correct:
                    active_correct += 1
            else:
                passive_total += 1
                if correct:
                    passive_correct += 1

        active_acc = active_correct / active_total if active_total > 0 else 0
        passive_acc = passive_correct / passive_total if passive_total > 0 else 0

        results[trial_name] = {
            "active_total": active_total,
            "active_correct": active_correct,
            "active_accuracy": round(active_acc, 4),
            "passive_total": passive_total,
            "passive_correct": passive_correct,
            "passive_accuracy": round(passive_acc, 4),
        }

        print(f"{trial_name:40s}  D4-active: {active_correct}/{active_total} = {active_acc:.4f}  "
              f"D4-passive: {passive_correct}/{passive_total} = {passive_acc:.4f}")

# ── Save results ────────────────────────────────────────────────────────────
output_path = RUN_DIR / "eval_results" / "d4_context_overlap.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {output_path}")
