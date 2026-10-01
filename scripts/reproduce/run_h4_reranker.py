#!/usr/bin/env python3
"""H4 — Reranker baseline on BM25 retrieval.

For each eval question:
  1. Load cached BM25 top-10 search results.
  2. Rerank with cross-encoder/ms-marco-MiniLM-L-12-v2 → top-5.
  3. Build TEXT_SESSIONS prompt with reranked hits + question.
  4. Call reader LLM for answering.
  5. Judge via gpt-4o-mini on OpenRouter.

Usage:
    python3 scripts/run_h4_reranker.py \
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
from typing import Any, Dict, List, Optional, Tuple

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
# Paths
# ---------------------------------------------------------------------------
_RUN_DIR = Path(__file__).resolve().parent.parent / "MASim" / "runs" / "l_20260408_111046"
_SEARCH_RESULTS = _RUN_DIR / "eval_results" / "inmem" / "search_results_rag_0_6b.json"
_OUT_BASE = _RUN_DIR / "eval_results" / "ablations" / "h4_reranker"

# Model tag -> served model name mapping
MODEL_NAMES = {
    "0_6b": "qwen3-0.6b",
    "llama3b": "llama-3.2-3b",
    "7b": "mistral-7b",
    "8b": "qwen3-8b",
    "32b": "qwen3-32b-awq",
}

# Dimension-aware hints (same as simple_eval)
_DIM_HINTS = {
    "d1_conflict": "The person may have said contradictory things at different times.",
    "d2_anaphora": "Pay attention to who 'they/them/it' refers to.",
    "d3_confabulation": "Only answer with information actually present.",
    "d4_permission": "Consider whether you should share this information.",
    "d5_cloze": "",
    "d6_metadata": "Recall specific details about when/where/how.",
    "d7_qa": "",
    "d8_temporal": "Pay attention to the timing and order of events.",
    "d9_negation": "Pay attention to what was explicitly denied or corrected.",
    "d10_counterfactual": "Consider hypothetical scenarios based on the history.",
}

# Skip-judge dims (rule-based scoring)
_SKIP_JUDGE_DIMS = {"d11_exception", "d9_negation", "d4_permission", "d1_conflict"}


def load_search_results() -> Dict[str, List[dict]]:
    """Load cached BM25 search results, keyed by question_id."""
    data = json.loads(_SEARCH_RESULTS.read_text())
    return {item["question_id"]: item["hits"] for item in data}


def rerank_hits(
    reranker,
    query: str,
    hits: List[dict],
    top_k: int = 5,
) -> List[dict]:
    """Rerank hits with cross-encoder, return top-k."""
    if not hits or len(hits) <= top_k:
        return hits

    pairs = [[query, h["text"]] for h in hits]
    scores = reranker.predict(pairs)

    scored = list(zip(hits, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [h for h, s in scored[:top_k]]


def build_reranked_prompt(
    instance: dict,
    reranked_hits: List[dict],
) -> Tuple[str, str]:
    """Build TEXT_SESSIONS style prompt from reranked hits."""
    query = instance.get("query", "").strip()
    dim = instance.get("dimension", "")
    meta = instance.get("metadata", {})
    query_agent = meta.get("query_agent", "")

    # Build context from reranked hits
    context_parts = []
    for i, hit in enumerate(reranked_hits):
        text = hit.get("text", "")
        ts = hit.get("occur_ts", "")
        if ts:
            context_parts.append(f"[{ts}]\n{text}")
        else:
            context_parts.append(text)

    context_block = "\n\n---\n\n".join(context_parts) if context_parts else "(no relevant history found)"

    # Dimension hint
    hint = _DIM_HINTS.get(dim, "")
    hint_line = f"\nHint: {hint}" if hint else ""

    user_prompt = (
        f"You are answering on behalf of {query_agent.replace('_', ' ') if query_agent else 'the person'}.\n\n"
        f"=== Relevant conversation history ===\n\n"
        f"{context_block}\n\n"
        f"=== End of history ==={hint_line}\n\n"
        f"Question: {query}"
    )

    return SIMPLE_EVAL_SYSTEM, user_prompt


async def run_h4(
    model_tag: str,
    reader_endpoint: str,
    reader_api_key: str,
    judge_endpoint: str,
    judge_model: str,
    judge_api_key: str,
    no_think: bool,
    concurrency: int,
    rerank_top_k: int,
    limit: Optional[int],
) -> dict:
    t0 = time.time()
    model_name = MODEL_NAMES.get(model_tag, model_tag)

    # Load instances and search results
    instances = load_instances(_RUN_DIR, limit=limit)
    search_results = load_search_results()
    corpus_sessions = load_corpus_sessions(_RUN_DIR)
    print(f"Loaded {len(instances)} instances, {len(search_results)} search results", file=sys.stderr)

    # Load reranker
    print("Loading ms-marco cross-encoder...", file=sys.stderr)
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    print("Reranker loaded.", file=sys.stderr)

    # Pre-rerank all questions (single-threaded, fast enough)
    print("Reranking all questions...", file=sys.stderr)
    reranked_cache: Dict[str, List[dict]] = {}
    for i, inst in enumerate(instances):
        qid = inst.get("question_id", "")
        hits = search_results.get(qid, [])
        query = inst.get("query", "")
        reranked_cache[qid] = rerank_hits(reranker, query, hits, top_k=rerank_top_k)
        if (i + 1) % 200 == 0:
            print(f"  Reranked {i+1}/{len(instances)}", file=sys.stderr)
    print(f"Reranking done ({len(reranked_cache)} questions)", file=sys.stderr)
    del reranker  # free GPU memory

    # Build prompts
    prompt_data: List[Tuple[str, str, dict]] = []
    for inst in instances:
        qid = inst.get("question_id", "")
        hits = reranked_cache.get(qid, [])
        system, user = build_reranked_prompt(inst, hits)
        if no_think:
            user = "/no_think\n" + user
        prompt_data.append((system, user, inst))

    # Run reader with semaphore
    sem = asyncio.Semaphore(concurrency)
    results: List[dict] = []
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
            result["answer_time_ms"] = meta.get("elapsed_ms")
            result["prompt_tokens"] = meta.get("prompt_tokens")
            result["completion_tokens"] = meta.get("completion_tokens")
            result["question_id"] = inst.get("question_id", f"h4_{idx}")
            result["dimension"] = inst.get("dimension", "")

            # Judge for open-ended answer dims
            dim = inst.get("dimension", "")
            if result.get("expected_mode") == "answer" and dim not in _SKIP_JUDGE_DIMS:
                gold = result["gold"]
                source_context = _build_source_context(inst, corpus_sessions)
                verdict = judge_answer(
                    inst, gold, prediction,
                    endpoint=judge_endpoint, model=judge_model, api_key=judge_api_key,
                    source_context=source_context,
                )
                result["judge_verdict"] = verdict["verdict"]
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
        dim = r.get("dimension", "unknown")
        by_dim[dim]["total"] += 1
        if r.get("correct"):
            by_dim[dim]["correct"] += 1

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "reader_model": model_name,
        "model_tag": model_tag,
        "reranker": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "bm25_candidates": 10,
        "rerank_top_k": rerank_top_k,
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
    parser = argparse.ArgumentParser(description="§H4: BM25 + ms-marco reranker baseline")
    parser.add_argument("--model-tag", required=True, choices=list(MODEL_NAMES.keys()))
    parser.add_argument("--reader-endpoint", default="http://127.0.0.1:8113/v1")
    parser.add_argument("--reader-api-key", default="EMPTY")
    parser.add_argument("--judge-endpoint", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini-2024-07-18")
    parser.add_argument("--judge-api-key", default=os.getenv("OPENROUTER_API_KEY", ""))
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.judge_api_key:
        print("ERROR: --judge-api-key or OPENROUTER_API_KEY required", file=sys.stderr)
        sys.exit(1)

    _OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_BASE / f"evaluation_results_h4_reranker_{args.model_tag}_s1.json"

    result = asyncio.run(run_h4(
        model_tag=args.model_tag,
        reader_endpoint=args.reader_endpoint,
        reader_api_key=args.reader_api_key,
        judge_endpoint=args.judge_endpoint,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
        no_think=args.no_think,
        concurrency=args.concurrency,
        rerank_top_k=args.rerank_top_k,
        limit=args.limit,
    ))

    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
