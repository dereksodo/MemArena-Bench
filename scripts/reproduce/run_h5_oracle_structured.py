#!/usr/bin/env python3
"""H5 — Oracle-structured control.

Feeds Oracle ground-truth evidence sessions through Memobase's
structured-extraction pipeline, then uses the structured profiles
as reader evidence.

Two phases:
  Phase 1 (extraction): For each ego, ingest ONLY their Oracle
    evidence sessions into a fresh Memobase user, wait for profile
    extraction, pull the structured context.
  Phase 2 (evaluation): For each MemArena-L question, use the ego's
    Oracle-structured profile as evidence, call 5 readers + judge.

Usage:
    # Phase 1: extraction (takes ~2-3h with async processing)
    python3 scripts/run_h5_oracle_structured.py extract

    # Phase 2: evaluation (takes ~5h for 5 readers)
    python3 scripts/run_h5_oracle_structured.py eval \
        --model-tag 0_6b \
        --reader-endpoint http://127.0.0.1:8113/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

# Add project root + eval to sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "eval"))

from simple_eval import (  # noqa: E402
    load_instances,
    load_corpus_sessions,
    load_ego_session_map,
    extract_answer,
    score,
    call_llm,
    judge_answer,
    _build_source_context,
)
from MASim.prompts import SIMPLE_EVAL_SYSTEM  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
_RUN_DIR = Path(__file__).resolve().parent.parent / "MASim" / "runs" / "l_20260408_111046"
_OUT_BASE = _RUN_DIR / "eval_results" / "ablations" / "h5_oracle_structured"
_PROFILES_CACHE = _OUT_BASE / "oracle_structured_profiles.json"

MEMOBASE_URL = "http://127.0.0.1:8021"
MEMOBASE_TOKEN = "secret"

MODEL_NAMES = {
    "0_6b": "qwen3-0.6b",
    "llama3b": "llama-3.2-3b",
    "7b": "mistral-7b",
    "8b": "qwen3-8b",
    "32b": "qwen3-32b-awq",
}

_SKIP_JUDGE_DIMS = {"d11_exception", "d9_negation", "d4_permission", "d1_conflict"}


# ---------------------------------------------------------------------------
# Phase 1: Extraction
# ---------------------------------------------------------------------------

def _collect_oracle_sessions_per_ego(instances: List[dict]) -> Dict[str, Set[str]]:
    """Collect unique Oracle evidence session IDs per ego."""
    ego_sessions: Dict[str, Set[str]] = defaultdict(set)
    for inst in instances:
        ego = inst.get("ego_agent_id", "")
        ev_sessions = inst.get("metadata", {}).get("evidence_session_ids", [])
        if not ev_sessions:
            # Try from ground_truth
            gt = inst.get("ground_truth", {})
            ev_sessions = gt.get("evidence_session_ids", [])
        if not ev_sessions:
            continue
        for sid in ev_sessions:
            ego_sessions[ego].add(sid)
    return ego_sessions


def _collect_from_eval_results(oracle_eval_path: Path) -> Dict[str, Set[str]]:
    """Collect Oracle evidence sessions from existing eval results."""
    ego_sessions: Dict[str, Set[str]] = defaultdict(set)
    data = json.loads(oracle_eval_path.read_text())
    for r in data.get("details", []):
        ev = r.get("evidence_sessions", [])
        # Need to find which ego this belongs to
        # The instance has ego_agent_id in the eval_instances
        qid = r.get("instance_id", r.get("question_id", ""))
        ego_sessions[qid] = set(ev) if ev else set()
    return ego_sessions


async def _memobase_create_user(session: aiohttp.ClientSession) -> str:
    """Create a fresh Memobase user, return UUID."""
    async with session.post(
        f"{MEMOBASE_URL}/api/v1/users",
        headers={"Authorization": f"Bearer {MEMOBASE_TOKEN}"},
        json={},
    ) as resp:
        data = await resp.json()
        return data["data"]["id"]


async def _memobase_ingest_messages(
    session: aiohttp.ClientSession,
    uid: str,
    messages: List[dict],
) -> None:
    """Ingest messages into Memobase user's chat buffer."""
    # Format as chat messages
    chat_msgs = []
    for m in messages:
        chat_msgs.append({
            "role": "user",
            "content": m.get("text", ""),
        })

    # Send in batches of 10 messages
    batch_size = 10
    for i in range(0, len(chat_msgs), batch_size):
        batch = chat_msgs[i:i + batch_size]
        async with session.post(
            f"{MEMOBASE_URL}/api/v1/users/buffer/{uid}/chat",
            headers={"Authorization": f"Bearer {MEMOBASE_TOKEN}"},
            json={"messages": batch},
        ) as resp:
            await resp.json()


async def _memobase_get_context(
    session: aiohttp.ClientSession,
    uid: str,
    max_retries: int = 30,
    retry_delay: float = 10.0,
) -> str:
    """Pull structured context, retrying until profile extraction completes."""
    for attempt in range(max_retries):
        async with session.get(
            f"{MEMOBASE_URL}/api/v1/users/context/{uid}?max_token_size=4096",
            headers={"Authorization": f"Bearer {MEMOBASE_TOKEN}"},
        ) as resp:
            data = await resp.json()
            context = data.get("data", {}).get("context", "")
            # Check if profile has actual content (not just the template)
            if "## User Current Profile:" in context:
                # Check if there's content after the profile header
                after_profile = context.split("## User Current Profile:")[1]
                if after_profile.strip().startswith("-") and len(after_profile.strip()) > 5:
                    return context
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)

    # Return whatever we have after retries
    async with session.get(
        f"{MEMOBASE_URL}/api/v1/users/context/{uid}?max_token_size=4096",
        headers={"Authorization": f"Bearer {MEMOBASE_TOKEN}"},
    ) as resp:
        data = await resp.json()
        return data.get("data", {}).get("context", "")


async def _memobase_delete_user(session: aiohttp.ClientSession, uid: str) -> None:
    """Delete a Memobase user."""
    async with session.delete(
        f"{MEMOBASE_URL}/api/v1/users/{uid}",
        headers={"Authorization": f"Bearer {MEMOBASE_TOKEN}"},
    ) as resp:
        await resp.json()


async def run_extraction():
    """Phase 1: Extract Oracle evidence through Memobase for each ego."""
    # Load eval instances to get evidence session IDs
    instances = load_instances(_RUN_DIR)
    corpus_sessions = load_corpus_sessions(_RUN_DIR)
    ego_session_map = load_ego_session_map(_RUN_DIR)

    # Load Oracle eval results to get evidence_sessions per question
    oracle_path = _RUN_DIR / "eval_results" / "oracle" / "evaluation_results_oracle_0_6b_4omini.json"
    oracle_data = json.loads(oracle_path.read_text())

    # Map question_id -> evidence_sessions + ego_agent_id
    qid_to_info: Dict[str, dict] = {}
    for inst in instances:
        qid = inst.get("question_id", inst.get("instance_id", ""))
        qid_to_info[qid] = {
            "ego_agent_id": inst.get("ego_agent_id", ""),
            "evidence_sessions": [],
        }

    for r in oracle_data.get("details", []):
        qid = r.get("instance_id", r.get("question_id", ""))
        ev = r.get("evidence_sessions", [])
        if qid in qid_to_info:
            qid_to_info[qid]["evidence_sessions"] = ev

    # Collect unique sessions per ego
    ego_oracle_sessions: Dict[str, Set[str]] = defaultdict(set)
    for qid, info in qid_to_info.items():
        ego = info["ego_agent_id"]
        for sid in info["evidence_sessions"]:
            ego_oracle_sessions[ego].add(sid)

    print(f"Egos with Oracle sessions: {len(ego_oracle_sessions)}", file=sys.stderr)
    total_sessions = sum(len(v) for v in ego_oracle_sessions.values())
    print(f"Total unique Oracle sessions: {total_sessions}", file=sys.stderr)

    # For each ego, ingest their Oracle sessions into Memobase and extract profile
    profiles: Dict[str, str] = {}
    _OUT_BASE.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        for i, (ego, session_ids) in enumerate(sorted(ego_oracle_sessions.items())):
            if not session_ids:
                profiles[ego] = ""
                continue

            print(f"[{i+1}/{len(ego_oracle_sessions)}] Extracting {ego} "
                  f"({len(session_ids)} sessions)...", file=sys.stderr)

            # Collect messages from corpus sessions
            messages = []
            for sid in sorted(session_ids):
                sess = corpus_sessions.get(sid, {})
                turns = sess.get("turns", sess.get("messages", []))
                for t in turns:
                    text = t.get("text", t.get("content", ""))
                    if text:
                        messages.append({"text": text})

            if not messages:
                profiles[ego] = ""
                continue

            # Create user, ingest, wait for extraction, pull context
            try:
                uid = await _memobase_create_user(session)
                await _memobase_ingest_messages(session, uid, messages)
                # Wait for async extraction
                context = await _memobase_get_context(session, uid, max_retries=60, retry_delay=5.0)
                profiles[ego] = context
                # Cleanup
                await _memobase_delete_user(session, uid)
                print(f"  → profile len={len(context)} chars", file=sys.stderr)
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                profiles[ego] = ""

    # Save profiles cache
    _PROFILES_CACHE.write_text(json.dumps(profiles, indent=2))
    print(f"\nExtraction done. Saved {len(profiles)} profiles to {_PROFILES_CACHE}", file=sys.stderr)
    non_empty = sum(1 for v in profiles.values() if v.strip())
    print(f"  Non-empty profiles: {non_empty}/{len(profiles)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Phase 2: Evaluation
# ---------------------------------------------------------------------------

async def run_eval(
    model_tag: str,
    reader_endpoint: str,
    reader_api_key: str,
    judge_endpoint: str,
    judge_model: str,
    judge_api_key: str,
    no_think: bool,
    concurrency: int,
    limit: Optional[int],
) -> dict:
    t0 = time.time()
    model_name = MODEL_NAMES.get(model_tag, model_tag)

    # Load cached profiles
    if not _PROFILES_CACHE.exists():
        print("ERROR: profiles cache not found. Run 'extract' phase first.", file=sys.stderr)
        sys.exit(1)
    profiles = json.loads(_PROFILES_CACHE.read_text())

    # Load instances
    instances = load_instances(_RUN_DIR, limit=limit)
    corpus_sessions = load_corpus_sessions(_RUN_DIR)
    print(f"Loaded {len(instances)} instances, {len(profiles)} profiles", file=sys.stderr)

    # Build prompts using structured profiles as evidence
    prompt_data: List[Tuple[str, str, dict]] = []
    for inst in instances:
        ego = inst.get("ego_agent_id", "")
        profile = profiles.get(ego, "")
        query = inst.get("query", "").strip()

        if profile:
            user_prompt = (
                f"You are answering on behalf of {ego.replace('_', ' ')}.\n\n"
                f"=== Structured memory profile ===\n\n"
                f"{profile}\n\n"
                f"=== End of profile ===\n\n"
                f"Question: {query}"
            )
        else:
            user_prompt = (
                f"You are answering on behalf of {ego.replace('_', ' ')}.\n\n"
                f"(No memory profile available)\n\n"
                f"Question: {query}"
            )

        system = SIMPLE_EVAL_SYSTEM
        if no_think:
            user_prompt = "/no_think\n" + user_prompt
        prompt_data.append((system, user_prompt, inst))

    # Run reader
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def _process(idx: int, system: str, user: str, inst: dict) -> dict:
        nonlocal done
        async with sem:
            raw, meta = await asyncio.to_thread(
                call_llm,
                system, user,
                endpoint=reader_endpoint, model=model_name,
                api_key=reader_api_key,
                temperature=0.0, max_tokens=512,
            )
            prediction = extract_answer(raw)
            result = score(inst, prediction, corpus_sessions=corpus_sessions)
            result["raw_response"] = raw
            result["question_id"] = inst.get("question_id", f"h5_{idx}")
            result["dimension"] = inst.get("dimension", "")

            # Judge
            dim = inst.get("dimension", "")
            if result.get("expected_mode") == "answer" and dim not in _SKIP_JUDGE_DIMS:
                gold = result["gold"]
                source_context = _build_source_context(inst, corpus_sessions)
                verdict = judge_answer(
                    inst, gold, prediction,
                    endpoint=judge_endpoint, model=judge_model, api_key=judge_api_key,
                    source_context=source_context,
                )
                result["judge_correct"] = verdict["correct"]
                result["correct"] = verdict["correct"]
                result["score"] = 1.0 if verdict["correct"] else 0.0
                result["scoring_method"] = "llm_judge"

            done += 1
            if done % 50 == 0:
                print(f"  [{model_tag}] {done}/{len(instances)} done", file=sys.stderr)
            return result

    tasks = [_process(i, s, u, inst) for i, (s, u, inst) in enumerate(prompt_data)]
    results = await asyncio.gather(*tasks)

    # Summarize
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    accuracy = correct / total if total else 0.0

    by_dim = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_dim[r.get("dimension", "unknown")]["total"] += 1
        if r.get("correct"):
            by_dim[r.get("dimension", "unknown")]["correct"] += 1

    summary = {
        "total": total, "correct": correct, "accuracy": accuracy,
        "reader_model": model_name, "model_tag": model_tag,
        "judge_model": judge_model,
        "by_dimension": {
            dim: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for dim, v in by_dim.items()
        },
        "latency_s": time.time() - t0,
    }
    print(f"\n[{model_tag}] DONE: {correct}/{total} = {accuracy:.4f}", file=sys.stderr)
    for dim, v in sorted(by_dim.items()):
        acc = v["correct"] / v["total"] if v["total"] else 0
        print(f"  {dim}: {v['correct']}/{v['total']} = {acc:.4f}", file=sys.stderr)

    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="§H5: Oracle-structured control")
    sub = parser.add_subparsers(dest="phase")

    sub.add_parser("extract", help="Phase 1: extract profiles from Oracle sessions")

    eval_p = sub.add_parser("eval", help="Phase 2: evaluate with structured profiles")
    eval_p.add_argument("--model-tag", required=True, choices=list(MODEL_NAMES.keys()))
    eval_p.add_argument("--reader-endpoint", default="http://127.0.0.1:8113/v1")
    eval_p.add_argument("--reader-api-key", default="EMPTY")
    eval_p.add_argument("--judge-endpoint", default="https://openrouter.ai/api/v1")
    eval_p.add_argument("--judge-model", default="openai/gpt-4o-mini-2024-07-18")
    eval_p.add_argument("--judge-api-key", default=os.getenv("OPENROUTER_API_KEY", ""))
    eval_p.add_argument("--no-think", action="store_true")
    eval_p.add_argument("--concurrency", type=int, default=32)
    eval_p.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.phase == "extract":
        asyncio.run(run_extraction())
    elif args.phase == "eval":
        if not args.judge_api_key:
            print("ERROR: --judge-api-key or OPENROUTER_API_KEY required", file=sys.stderr)
            sys.exit(1)
        _OUT_BASE.mkdir(parents=True, exist_ok=True)
        out_path = _OUT_BASE / f"evaluation_results_h5_oracle_structured_{args.model_tag}_s1.json"
        result = asyncio.run(run_eval(
            model_tag=args.model_tag,
            reader_endpoint=args.reader_endpoint,
            reader_api_key=args.reader_api_key,
            judge_endpoint=args.judge_endpoint,
            judge_model=args.judge_model,
            judge_api_key=args.judge_api_key,
            no_think=args.no_think,
            concurrency=args.concurrency,
            limit=args.limit,
        ))
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"Wrote {out_path}", file=sys.stderr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
