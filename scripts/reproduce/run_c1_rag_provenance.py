#!/usr/bin/env python3
"""C1 — RAG with provenance-chunk metadata (§H7 / Wave-2 W1).

Retriever: BGE-M3 top-5 (frozen, no λ or fusion).
Chunk header: [session=<sid> day=<day_idx> speaker=<agent> time=<ts>]
Reader: Qwen3-8B (bf16) on :8110, GPU 1.
Judge: openai/gpt-4o-mini-2024-07-18 via OpenRouter.

Usage:
    python3 scripts/run_c1_rag_provenance.py --seed 1
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
    load_ego_session_map,
    extract_answer,
    score,
    judge_answer,
    _build_source_context,
)
from MASim.prompts import SIMPLE_EVAL_SYSTEM  # noqa: E402

import requests  # noqa: E402
import numpy as np  # noqa: E402

_RUN_DIR = Path(__file__).resolve().parent.parent / "MASim" / "runs" / "l_20260408_111046"
_OUT_BASE = _RUN_DIR / "eval_results" / "ablations" / "c1_rag_provenance"

_SKIP_JUDGE_DIMS = {"d11_exception", "d9_negation", "d4_permission", "d1_conflict"}


# ---------------------------------------------------------------------------
# BGE-M3 per-ego index (reused from B2)
# ---------------------------------------------------------------------------

class BgeM3EgoIndex:
    def __init__(self):
        self._model = None
        self._indexes: Dict[str, Tuple[np.ndarray, List[dict]]] = {}

    def _ensure_model(self):
        if self._model is not None:
            return
        from FlagEmbedding import BGEM3FlagModel
        print("Loading BAAI/bge-m3...", file=sys.stderr)
        self._model = BGEM3FlagModel(os.environ.get("MEMARENA_BGE_M3", "BAAI/bge-m3"), use_fp16=True)

    def build_ego(self, ego_id: str, turns: List[dict]):
        if ego_id in self._indexes:
            return
        self._ensure_model()
        texts = [t.get("text", "") for t in turns]
        if not texts:
            self._indexes[ego_id] = (np.zeros((0, 1024), dtype=np.float32), [])
            return
        out = self._model.encode(texts, batch_size=128, max_length=2048,
                                 return_dense=True, return_sparse=False,
                                 return_colbert_vecs=False)
        vecs = out["dense_vecs"]
        if not isinstance(vecs, np.ndarray):
            vecs = np.array(vecs, dtype=np.float32)
        self._indexes[ego_id] = (vecs.astype(np.float32), turns)

    def search(self, ego_id: str, query: str, top_k: int) -> List[dict]:
        if ego_id not in self._indexes:
            return []
        self._ensure_model()
        embs, turns = self._indexes[ego_id]
        if embs.shape[0] == 0:
            return []
        qvec = self._model.encode([query], batch_size=1, max_length=2048,
                                  return_dense=True, return_sparse=False,
                                  return_colbert_vecs=False)["dense_vecs"]
        if not isinstance(qvec, np.ndarray):
            qvec = np.array(qvec, dtype=np.float32)
        qvec = qvec.astype(np.float32).reshape(-1)
        scores = embs @ qvec
        k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        hits = []
        for i in top_idx:
            t = turns[int(i)]
            hits.append({
                "msg_id": t.get("turn_id"),
                "score": float(scores[int(i)]),
                "text": t.get("text", ""),
                "session_id": t.get("session_id", ""),
                "speaker_id": t.get("speaker_id", ""),
                "timestamp": t.get("timestamp", 0),
                "session_start": t.get("session_start", ""),
            })
        return hits


def build_ego_turn_map(
    corpus_sessions: Dict[str, dict],
    ego_session_map: Dict[str, List[str]],
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for ego, sids in ego_session_map.items():
        turns: List[dict] = []
        for sid in sids:
            sess = corpus_sessions.get(sid)
            if not sess:
                continue
            start_iso = sess.get("start_time") or ""
            for t in sess.get("turns", []):
                t = {**t, "session_start": start_iso, "session_id": sid}
                turns.append(t)
        out[ego] = turns
    return out


def _day_idx_from_timestamp(ts) -> str:
    """MASim timestamp is fractional day-of-run. Return int day index."""
    try:
        return str(int(float(ts)))
    except (ValueError, TypeError):
        return "?"


def format_chunk_with_provenance(hit: dict) -> str:
    """Prepend session/day/speaker/time metadata header to the text."""
    sid = hit.get("session_id", "?")
    day = _day_idx_from_timestamp(hit.get("timestamp", 0))
    speaker = hit.get("speaker_id", "?")
    ts = hit.get("timestamp", 0)
    header = f"[session={sid} day={day} speaker={speaker} time={ts}]"
    text = hit.get("text", "")
    return f"{header}\n{text}"


def build_prompt_from_hits(instance: dict, hits: List[dict]) -> Tuple[str, str]:
    query = instance.get("query", "").strip()
    ego = instance.get("ego_agent_id", "")
    blocks = [format_chunk_with_provenance(h) for h in hits]
    ctx = "\n\n---\n\n".join(blocks) if blocks else "(no relevant history found)"
    user = (
        f"You are answering on behalf of {ego.replace('_', ' ')}.\n\n"
        f"=== Relevant conversation history (with session/day/speaker/time metadata) ===\n\n"
        f"{ctx}\n\n"
        f"=== End of history ===\n\n"
        f"Question: {query}"
    )
    return SIMPLE_EVAL_SYSTEM, user


def call_llm_with_seed(system, user, *, endpoint, model, api_key, temperature, max_tokens, seed):
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
    }
    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"], {
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": data.get("usage", {}).get("completion_tokens"),
            }
        except Exception as e:
            if attempt == 4:
                return f"LLM_ERROR: {type(e).__name__}: {str(e)[:200]}", {}
            time.sleep(1 + attempt)
    return "LLM_ERROR: unreachable", {}


async def run_c1(
    seed: int,
    model_name: str,
    reader_endpoint: str,
    reader_api_key: str,
    judge_endpoint: str,
    judge_model: str,
    judge_api_key: str,
    dims: List[str],
    concurrency: int,
    top_k: int,
    no_think: bool,
) -> dict:
    t0 = time.time()
    print(f"[s{seed}] starting", file=sys.stderr)

    instances = load_instances(_RUN_DIR, dimensions=dims)
    corpus_sessions = load_corpus_sessions(_RUN_DIR)
    ego_session_map = load_ego_session_map(_RUN_DIR)
    ego_turns = build_ego_turn_map(corpus_sessions, ego_session_map)
    print(f"[s{seed}] {len(instances)} instances, {len(ego_turns)} egos", file=sys.stderr)

    # Build per-ego BGE-M3 indexes
    dense = BgeM3EgoIndex()
    for i, (ego, turns) in enumerate(ego_turns.items()):
        dense.build_ego(ego, turns)
        if (i + 1) % 10 == 0:
            print(f"  built BGE-M3 index for {i+1}/{len(ego_turns)} egos", file=sys.stderr)

    # Build prompts
    prompt_data: List[Tuple[str, str, dict]] = []
    for inst in instances:
        ego = inst.get("ego_agent_id", "")
        query = inst.get("query", "").strip()
        hits = dense.search(ego, query, top_k=top_k)
        system, user = build_prompt_from_hits(inst, hits)
        if no_think:
            user = "/no_think\n" + user
        prompt_data.append((system, user, inst))

    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def _process(idx, system, user, inst):
        nonlocal done
        async with sem:
            raw, lm = await asyncio.to_thread(
                call_llm_with_seed, system, user,
                endpoint=reader_endpoint, model=model_name, api_key=reader_api_key,
                temperature=0.3, max_tokens=512, seed=seed,
            )
            prediction = extract_answer(raw)
            result = score(inst, prediction, corpus_sessions=corpus_sessions)
            result["raw_response"] = raw
            result["question_id"] = inst.get("question_id", inst.get("instance_id", f"c1_{idx}"))
            result["dimension"] = inst.get("dimension", "")
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
            if done % 100 == 0:
                print(f"  [s{seed}] {done}/{len(instances)} done", file=sys.stderr)
            return result

    tasks = [_process(i, s, u, inst) for i, (s, u, inst) in enumerate(prompt_data)]
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
        "seed": seed, "top_k": top_k,
        "reader_model": model_name, "judge_model": judge_model,
        "dims": dims,
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
    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="C1: RAG with provenance-chunking")
    parser.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--model-name", default="qwen3-8b")
    parser.add_argument("--reader-endpoint", default="http://127.0.0.1:8110/v1")
    parser.add_argument("--reader-api-key", default="EMPTY")
    parser.add_argument("--judge-endpoint", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini-2024-07-18")
    parser.add_argument("--openrouter-api-key", default=os.getenv("OPENROUTER_API_KEY", ""))
    parser.add_argument("--dims", nargs="+",
                        default=["d1_conflict", "d2_anaphora", "d3_confabulation",
                                 "d4_permission", "d5_cloze", "d6_metadata",
                                 "d7_qa", "d8_temporal", "d10_counterfactual"])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-think", action="store_true", default=True)
    args = parser.parse_args()

    if not args.openrouter_api_key:
        print("ERROR: --openrouter-api-key or OPENROUTER_API_KEY required", file=sys.stderr)
        sys.exit(1)

    _OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_BASE / f"evaluation_results_c1_8b_s{args.seed}_4omini.json"

    result = asyncio.run(run_c1(
        seed=args.seed,
        model_name=args.model_name,
        reader_endpoint=args.reader_endpoint, reader_api_key=args.reader_api_key,
        judge_endpoint=args.judge_endpoint, judge_model=args.judge_model,
        judge_api_key=args.openrouter_api_key,
        dims=args.dims, concurrency=args.concurrency,
        top_k=args.top_k, no_think=args.no_think,
    ))
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
