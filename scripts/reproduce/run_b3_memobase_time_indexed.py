#!/usr/bin/env python3
"""B3 — Time-indexed Memobase.

Reuses existing Qwen3-8B Memobase memcache files (no re-ingest) and
applies a retrieve-time filter: parse each fact's "[mention YYYY/MM/DD]"
span, derive (valid_from, valid_to), keep only facts whose span is
<= query_time anchor.

query_time fallback (SOP §B3): if metadata.query_time is missing,
use max(valid_from over retrieved facts) + 1 day.

Usage:
    python3 scripts/run_b3_memobase_time_indexed.py \\
        --seed 1 --dims d3_confabulation d4_permission \\
        --reader-endpoint http://127.0.0.1:8110/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "eval"))

from simple_eval import (  # noqa: E402
    load_instances,
    load_corpus_sessions,
    extract_answer,
    score,
    judge_answer,
    _build_source_context,
)
from MASim.prompts import SIMPLE_EVAL_SYSTEM  # noqa: E402

import requests  # noqa: E402

_RUN_DIR = Path(__file__).resolve().parent.parent / "MASim" / "runs" / "l_20260408_111046"
_OUT_BASE = _RUN_DIR / "eval_results" / "ablations" / "b3_memobase_time_indexed"

# Map seed -> source memobase memcache (Qwen3-8B extractor, A_paired config).
# Note: s1 memcache for Qwen3-8B doesn't exist (only s2/s3/s4 were computed).
# Per SOP B3 "do not re-ingest" cost-saving rule, remap seeds 1/2/3 → s2/s3/s4.
_MEMCACHE_FOR_SEED = {
    1: _RUN_DIR / "eval_results_s2" / "memory_cache" / "memory_cache" / "memcache_memobase_A_paired_qwen3_8b.jsonl",
    2: _RUN_DIR / "eval_results_s3" / "memory_cache" / "memory_cache" / "memcache_memobase_A_paired_qwen3_8b.jsonl",
    3: _RUN_DIR / "eval_results_s4" / "memory_cache" / "memory_cache" / "memcache_memobase_A_paired_qwen3_8b.jsonl",
}

_SKIP_JUDGE_DIMS = {"d11_exception", "d9_negation", "d4_permission", "d1_conflict"}

# Regex for [mention YYYY/MM/DD] or [mention YYYY-MM-DD] or [mentioned on YYYY/MM/DD]
_DATE_RE = re.compile(
    r"\[(?:mention(?:ed)?(?:\s+on)?)\s+(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\]",
    flags=re.IGNORECASE,
)
# Regex for inline dates like "YYYY/MM/DD" (fallback)
_DATE_INLINE_RE = re.compile(r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})")

# Regex to split memobase profile text into facts (per line starting with "- ")
_FACT_LINE_RE = re.compile(r"^-\s+(.+)$", flags=re.MULTILINE)


def _parse_dates(text: str) -> List[datetime]:
    """Extract all mention dates from fact text."""
    out = []
    for m in _DATE_RE.finditer(text):
        try:
            out.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    if not out:
        for m in _DATE_INLINE_RE.finditer(text):
            try:
                out.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass
    return out


def _split_facts(profile_text: str) -> List[str]:
    """Split Memobase profile text into per-fact lines."""
    return [m.group(1).strip() for m in _FACT_LINE_RE.finditer(profile_text)]


def _fact_valid_from(fact: str) -> Optional[datetime]:
    """Earliest mention date in a fact, or None."""
    dates = _parse_dates(fact)
    return min(dates) if dates else None


def _derive_query_time(
    instance: dict,
    retrieved_facts: List[str],
) -> Optional[datetime]:
    """Return query_time anchor; fall back to max(valid_from) + 1 day."""
    meta = instance.get("metadata", {}) or {}
    # Try explicit metadata. Require absolute-epoch semantics (seconds since 1970,
    # so value > Jan 1 2020 = 1577836800). Relative sim-time fields are skipped
    # and we fall through to the max-valid_from heuristic below.
    for key in ("query_time", "query_timestamp", "occur_ts"):
        val = meta.get(key) or instance.get(key)
        if val:
            try:
                if isinstance(val, (int, float)):
                    if float(val) < 1577836800:  # pre-2020, clearly relative
                        continue
                    return datetime.fromtimestamp(float(val))
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                if dt.year < 2020:
                    continue
                return dt
            except Exception:
                pass

    # Fallback: max(valid_from) across retrieved facts + 1 day
    dates = []
    for f in retrieved_facts:
        vf = _fact_valid_from(f)
        if vf:
            dates.append(vf)
    if dates:
        return max(dates) + timedelta(days=1)
    return None


def _filter_facts_by_time(
    facts: List[str],
    query_time: Optional[datetime],
    time_window_days: int = 0,
) -> Tuple[List[str], int]:
    """Keep facts whose earliest mention <= query_time + window. Return (filtered, n_dropped)."""
    if query_time is None or not facts:
        return facts, 0
    threshold = query_time + timedelta(days=time_window_days)
    kept = []
    dropped = 0
    for f in facts:
        vf = _fact_valid_from(f)
        if vf is None:
            # Facts without dates: keep (undated general profile)
            kept.append(f)
        elif vf <= threshold:
            kept.append(f)
        else:
            dropped += 1
    return kept, dropped


def _load_memcache(seed: int) -> Dict[str, List[str]]:
    """Load memobase memcache for the seed. Returns {question_id: [profile_text per hit]}."""
    path = _MEMCACHE_FOR_SEED[seed]
    out: Dict[str, List[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("instance_id") or row.get("question_id", "")
            mems = row.get("memories") or row.get("search_results") or []
            texts = []
            for m in mems:
                if isinstance(m, dict):
                    t = m.get("text") or m.get("content") or ""
                elif isinstance(m, str):
                    t = m
                else:
                    t = ""
                if t:
                    texts.append(t)
            out[qid] = texts
    return out


def build_prompt_from_profile(
    instance: dict,
    profile_text: str,
) -> Tuple[str, str]:
    query = instance.get("query", "").strip()
    ego = instance.get("ego_agent_id", "")
    user_prompt = (
        f"You are answering on behalf of {ego.replace('_', ' ')}.\n\n"
        f"=== Structured memory profile (time-filtered) ===\n\n"
        f"{profile_text}\n\n"
        f"=== End of profile ===\n\n"
        f"Question: {query}"
    )
    return SIMPLE_EVAL_SYSTEM, user_prompt


def call_llm_with_seed(
    system: str, user: str, *, endpoint: str, model: str, api_key: str,
    temperature: float, max_tokens: int, seed: int,
) -> Tuple[str, dict]:
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    payload = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
        "seed": seed,
    }
    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return text, {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except Exception as e:
            if attempt == 4:
                return f"LLM_ERROR: {type(e).__name__}: {str(e)[:200]}", {}
            time.sleep(1 + attempt)
    return "LLM_ERROR: unreachable", {}


async def run_b3(
    seed: int,
    model_tag: str,
    model_name: str,
    reader_endpoint: str,
    reader_api_key: str,
    judge_endpoint: str,
    judge_model: str,
    judge_api_key: str,
    dims: List[str],
    concurrency: int,
    time_window_days: int,
    fallback_on_empty: str,
    no_think: bool,
) -> dict:
    t0 = time.time()

    instances = load_instances(_RUN_DIR, dimensions=dims)
    print(f"[s{seed}] Loaded {len(instances)} instances", file=sys.stderr)
    corpus_sessions = load_corpus_sessions(_RUN_DIR)
    memcache = _load_memcache(seed)
    print(f"[s{seed}] Loaded memcache with {len(memcache)} entries", file=sys.stderr)

    # Build time-filtered prompts
    prompt_data: List[Tuple[str, str, dict, dict]] = []  # (sys, user, inst, filter_meta)
    fallback_count = 0
    total_facts_before = 0
    total_facts_after = 0

    for inst in instances:
        qid = inst.get("question_id", inst.get("instance_id", ""))
        hits = memcache.get(qid, [])
        combined_profile = "\n\n---\n\n".join(hits) if hits else ""

        # Split into facts, derive query_time, filter
        facts = _split_facts(combined_profile)
        total_facts_before += len(facts)

        # For query_time, use facts as the retrieved-evidence pool
        query_time = _derive_query_time(inst, facts)
        filtered_facts, dropped = _filter_facts_by_time(facts, query_time, time_window_days)
        total_facts_after += len(filtered_facts)

        # If filter drops everything, optionally fall back
        used_fallback = False
        if not filtered_facts and facts:
            if fallback_on_empty == "memobase":
                filtered_facts = facts
                used_fallback = True
                fallback_count += 1

        # Rebuild profile text from filtered facts (preserve header)
        if hits and combined_profile:
            # Keep the memobase header, replace the facts block
            lines = combined_profile.split("\n")
            header_lines = []
            for ln in lines:
                if ln.startswith("- "):
                    break
                header_lines.append(ln)
            header = "\n".join(header_lines)
            rebuilt = header + "\n" + "\n".join(f"- {f}" for f in filtered_facts)
        else:
            rebuilt = "(no profile available)"

        system, user = build_prompt_from_profile(inst, rebuilt)
        if no_think:
            user = "/no_think\n" + user
        prompt_data.append((system, user, inst, {
            "n_facts_before": len(facts),
            "n_facts_after": len(filtered_facts),
            "query_time": query_time.isoformat() if query_time else None,
            "used_fallback": used_fallback,
        }))

    print(f"[s{seed}] Facts: {total_facts_before} → {total_facts_after} "
          f"(-{total_facts_before - total_facts_after}); fallbacks: {fallback_count}/{len(instances)}",
          file=sys.stderr)

    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def _process(idx: int, system: str, user: str, inst: dict, fmeta: dict) -> dict:
        nonlocal done
        async with sem:
            raw, lm = await asyncio.to_thread(
                call_llm_with_seed,
                system, user,
                endpoint=reader_endpoint, model=model_name,
                api_key=reader_api_key,
                temperature=0.3, max_tokens=512, seed=seed,
            )
            prediction = extract_answer(raw)
            result = score(inst, prediction, corpus_sessions=corpus_sessions)
            result["raw_response"] = raw
            result["question_id"] = inst.get("question_id", inst.get("instance_id", f"b3_{idx}"))
            result["dimension"] = inst.get("dimension", "")
            result["filter_meta"] = fmeta
            dim = inst.get("dimension", "")
            if result.get("expected_mode") == "answer" and dim not in _SKIP_JUDGE_DIMS:
                gold = result["gold"]
                src = _build_source_context(inst, corpus_sessions)
                verdict = judge_answer(
                    inst, gold, prediction,
                    endpoint=judge_endpoint, model=judge_model, api_key=judge_api_key,
                    source_context=src,
                )
                result["judge_correct"] = verdict["correct"]
                result["correct"] = verdict["correct"]
                result["score"] = 1.0 if verdict["correct"] else 0.0
                result["scoring_method"] = "llm_judge"
            done += 1
            if done % 50 == 0:
                print(f"  [s{seed}] {done}/{len(instances)} done", file=sys.stderr)
            return result

    tasks = [_process(i, s, u, inst, fm) for i, (s, u, inst, fm) in enumerate(prompt_data)]
    results = await asyncio.gather(*tasks)

    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    by_dim = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        d = r.get("dimension", "unknown")
        by_dim[d]["total"] += 1
        if r.get("correct"):
            by_dim[d]["correct"] += 1

    summary = {
        "total": total, "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "reader_model": model_name, "model_tag": model_tag,
        "seed": seed, "dims": dims,
        "judge_model": judge_model,
        "time_window_days": time_window_days,
        "fallback_on_empty": fallback_on_empty,
        "fallback_count": fallback_count,
        "n_facts_before": total_facts_before,
        "n_facts_after": total_facts_after,
        "by_dimension": {
            dim: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for dim, v in by_dim.items()
        },
        "latency_s": time.time() - t0,
    }
    print(f"\n[s{seed}] DONE: {correct}/{total} = {correct/total if total else 0:.4f}",
          file=sys.stderr)
    for dim, v in sorted(by_dim.items()):
        acc = v["correct"] / v["total"] if v["total"] else 0
        print(f"  {dim}: {v['correct']}/{v['total']} = {acc:.4f}", file=sys.stderr)
    print(f"  fallback_count: {fallback_count}/{total} = "
          f"{fallback_count/total*100:.1f}% of queries fell back",
          file=sys.stderr)
    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="B3: time-indexed Memobase")
    parser.add_argument("--model-tag", default="8b")
    parser.add_argument("--model-name", default="qwen3-8b")
    parser.add_argument("--reader-endpoint", default="http://127.0.0.1:8110/v1")
    parser.add_argument("--reader-api-key", default="EMPTY")
    parser.add_argument("--judge-endpoint", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini-2024-07-18")
    parser.add_argument("--openrouter-api-key", default=os.getenv("OPENROUTER_API_KEY", ""))
    parser.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--dims", nargs="+", default=["d3_confabulation", "d4_permission"])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--time-window-days", type=int, default=0,
                        help="widen filter if recall too low; SOP default 0 (exact anchor)")
    parser.add_argument("--fallback-on-empty", choices=["memobase", "none"], default="memobase")
    parser.add_argument("--no-think", action="store_true", default=True)
    args = parser.parse_args()

    if not args.openrouter_api_key:
        print("ERROR: --openrouter-api-key or OPENROUTER_API_KEY required", file=sys.stderr)
        sys.exit(1)

    _OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_BASE / f"evaluation_results_b3_{args.model_tag}_s{args.seed}_4omini.json"

    result = asyncio.run(run_b3(
        seed=args.seed,
        model_tag=args.model_tag,
        model_name=args.model_name,
        reader_endpoint=args.reader_endpoint,
        reader_api_key=args.reader_api_key,
        judge_endpoint=args.judge_endpoint,
        judge_model=args.judge_model,
        judge_api_key=args.openrouter_api_key,
        dims=args.dims,
        concurrency=args.concurrency,
        time_window_days=args.time_window_days,
        fallback_on_empty=args.fallback_on_empty,
        no_think=args.no_think,
    ))
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
