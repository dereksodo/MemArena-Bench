#!/usr/bin/env python3
"""B2 — Stronger RAG variants (hybrid_rrf, entity_filter, temporal_prior).

Reuses:
  - Cached BM25 top-10 hits (eval_results/inmem/search_results_rag_0_6b.json)
  - BGE-M3 encoder (FlagEmbedding) built lazily over ego sessions
  - Qwen3-8B reader on :8110
  - 4o-mini judge via OpenRouter

Usage:
    python3 scripts/run_b2_rag_strong.py \\
        --variant rag_hybrid_rrf --seed 1 \\
        --dims d1_conflict d3_confabulation d4_permission
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
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
_OUT_BASE = _RUN_DIR / "eval_results" / "ablations" / "b2_rag_strong"
_BM25_CACHE = _RUN_DIR / "eval_results" / "inmem" / "search_results_rag_0_6b.json"
_PERSONAS_PATH = _RUN_DIR / "agents_personas.jsonl"

_SKIP_JUDGE_DIMS = {"d11_exception", "d9_negation", "d4_permission", "d1_conflict"}

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})T")


def _load_agent_registry() -> List[str]:
    """Return deterministic list of agent names + slug tokens for entity matching."""
    names = set()
    for line in _PERSONAS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        aid = d.get("agent_id", "")
        if aid:
            names.add(aid)
            # human-readable form
            names.add(aid.replace("_", " ").title())
            # individual name parts (first, last)
            for part in aid.replace("_", " ").title().split():
                if len(part) > 2:
                    names.add(part)
    return sorted(names, key=lambda x: -len(x))  # longer matches first


def _extract_entities(text: str, agent_names: List[str]) -> set:
    """Match any agent name + capitalized multi-word tokens in text."""
    out = set()
    tl = text.lower()
    for name in agent_names:
        if name.lower() in tl:
            out.add(name.lower())
    # Also capture CapitalizedWords as entities (cheap NER alternative)
    for m in re.finditer(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", text):
        out.add(m.group(0).lower())
    return out


def _parse_occur_ts(ts) -> Optional[datetime]:
    """Accept ISO strings (BM25 cache) or numeric floats (dense index)."""
    if ts is None or ts == "":
        return None
    if isinstance(ts, (int, float)):
        # Fractional simulation day relative to run start (not Unix epoch);
        # Use unix-reference for _relative_ ordering within the run.
        try:
            # Day-granularity is sufficient for the 30-day temporal kernel.
            return datetime(2025, 1, 1) + __import__("datetime").timedelta(days=float(ts))
        except (ValueError, OverflowError):
            return None
    m = _DATE_RE.match(str(ts))
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _load_bm25_cache() -> Dict[str, List[dict]]:
    data = json.loads(_BM25_CACHE.read_text())
    return {item["question_id"]: item["hits"] for item in data}


# ---------------------------------------------------------------------------
# BGE-M3 retrieval (per ego, built lazily)
# ---------------------------------------------------------------------------

class BgeM3EgoIndex:
    """Minimal ego-scoped BGE-M3 retriever. Builds index from corpus sessions."""

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
        """Build (embeddings, turns) for one ego."""
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
                "occur_ts": t.get("timestamp", ""),
                "thread_id": t.get("session_id", ""),
                "user_id": t.get("speaker_id", ""),
            })
        return hits


def build_ego_turn_map(
    corpus_sessions: Dict[str, dict],
    ego_session_map: Dict[str, List[str]],
) -> Dict[str, List[dict]]:
    """For each ego, return the flat list of turn dicts across their sessions."""
    out: Dict[str, List[dict]] = {}
    for ego, sids in ego_session_map.items():
        turns: List[dict] = []
        for sid in sids:
            sess = corpus_sessions.get(sid)
            if not sess:
                continue
            start_iso = sess.get("start_time") or ""
            for t in sess.get("turns", []):
                # Augment with session absolute start date for temporal prior
                t = {**t, "session_start": start_iso, "session_id": sid}
                turns.append(t)
        out[ego] = turns
    return out


# ---------------------------------------------------------------------------
# Variant fusion functions
# ---------------------------------------------------------------------------

def rrf_merge(bm25_hits: List[dict], dense_hits: List[dict],
              rrf_k: int = 60, top_k: int = 5) -> List[dict]:
    scores: Dict[str, float] = {}
    by_id: Dict[str, dict] = {}
    for rank, h in enumerate(bm25_hits):
        mid = h["msg_id"]
        scores[mid] = scores.get(mid, 0) + 1.0 / (rrf_k + rank + 1)
        by_id[mid] = h
    for rank, h in enumerate(dense_hits):
        mid = h["msg_id"]
        scores[mid] = scores.get(mid, 0) + 1.0 / (rrf_k + rank + 1)
        if mid not in by_id:
            by_id[mid] = h
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    return [{**by_id[mid], "score": s} for mid, s in ordered[:top_k]]


def entity_filter(dense_hits: List[dict], query: str,
                  agent_names: List[str], top_k: int = 5) -> Tuple[List[dict], bool]:
    query_entities = _extract_entities(query, agent_names)
    if not query_entities:
        return dense_hits[:top_k], True  # fallback
    kept = []
    for h in dense_hits:
        hit_entities = _extract_entities(h.get("text", ""), agent_names)
        if hit_entities & query_entities:
            kept.append(h)
    if len(kept) < 2:
        return dense_hits[:top_k], True  # fallback to unfiltered
    return kept[:top_k], False


def temporal_prior_rerank(dense_hits: List[dict], query_anchor: Optional[datetime],
                          lam: float, top_k: int = 5) -> List[dict]:
    if query_anchor is None or lam == 0:
        return dense_hits[:top_k]
    rescored = []
    for h in dense_hits:
        dt = _parse_occur_ts(h.get("occur_ts", ""))
        if dt is None:
            rescored.append((h["score"], h))
            continue
        delta_days = abs((dt - query_anchor).days)
        boost = lam * math.exp(-delta_days / 30.0)  # 30-day kernel
        rescored.append((h["score"] + boost, h))
    rescored.sort(key=lambda x: -x[0])
    return [{**h, "score": s} for s, h in rescored[:top_k]]


# ---------------------------------------------------------------------------
# Prompt + LLM
# ---------------------------------------------------------------------------

def build_prompt_from_hits(instance: dict, hits: List[dict]) -> Tuple[str, str]:
    query = instance.get("query", "").strip()
    ego = instance.get("ego_agent_id", "")
    blocks = []
    for h in hits:
        ts = h.get("occur_ts", "")
        blocks.append(f"[{ts}]\n{h.get('text','')}" if ts else h.get("text", ""))
    ctx = "\n\n---\n\n".join(blocks) if blocks else "(no relevant history found)"
    user = (
        f"You are answering on behalf of {ego.replace('_', ' ')}.\n\n"
        f"=== Relevant conversation history ===\n\n"
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


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

async def run_variant(
    variant: str,
    seed: int,
    lam: float,
    model_name: str,
    reader_endpoint: str,
    reader_api_key: str,
    judge_endpoint: str,
    judge_model: str,
    judge_api_key: str,
    dims: List[str],
    concurrency: int,
    no_think: bool,
) -> dict:
    t0 = time.time()
    print(f"[{variant} s{seed} λ={lam}] starting", file=sys.stderr)

    instances = load_instances(_RUN_DIR, dimensions=dims)
    corpus_sessions = load_corpus_sessions(_RUN_DIR)
    ego_session_map = load_ego_session_map(_RUN_DIR)
    ego_turns = build_ego_turn_map(corpus_sessions, ego_session_map)
    print(f"[{variant} s{seed}] {len(instances)} instances, "
          f"{len(ego_turns)} egos, avg {sum(len(v) for v in ego_turns.values())/max(len(ego_turns),1):.0f} turns/ego",
          file=sys.stderr)

    bm25_cache = _load_bm25_cache()
    agent_names = _load_agent_registry()

    # Build BGE-M3 indexes for variants that need them
    need_dense = variant in ("rag_hybrid_rrf", "rag_entity_filter", "rag_temporal_prior")
    dense = BgeM3EgoIndex() if need_dense else None
    if dense is not None:
        for i, (ego, turns) in enumerate(ego_turns.items()):
            dense.build_ego(ego, turns)
            if (i + 1) % 10 == 0:
                print(f"  [{variant}] built BGE-M3 index for {i+1}/{len(ego_turns)} egos",
                      file=sys.stderr)

    # Build prompts
    prompt_data: List[Tuple[str, str, dict, dict]] = []
    fallback_count = 0
    for inst in instances:
        qid = inst.get("question_id", inst.get("instance_id", ""))
        ego = inst.get("ego_agent_id", "")
        query = inst.get("query", "").strip()
        bm25_hits = bm25_cache.get(qid, [])

        meta = {"variant": variant, "n_bm25": len(bm25_hits), "n_dense_before_fuse": 0,
                "n_final": 0, "used_fallback": False}

        if variant == "rag_hybrid_rrf":
            dense_hits = dense.search(ego, query, top_k=10) if dense else []
            meta["n_dense_before_fuse"] = len(dense_hits)
            final = rrf_merge(bm25_hits, dense_hits, rrf_k=60, top_k=5)

        elif variant == "rag_entity_filter":
            dense_hits = dense.search(ego, query, top_k=50) if dense else []
            meta["n_dense_before_fuse"] = len(dense_hits)
            final, used_fb = entity_filter(dense_hits, query, agent_names, top_k=5)
            meta["used_fallback"] = used_fb
            if used_fb:
                fallback_count += 1

        elif variant == "rag_temporal_prior":
            dense_hits = dense.search(ego, query, top_k=10) if dense else []
            meta["n_dense_before_fuse"] = len(dense_hits)
            # Query anchor = max occur_ts across bm25 hits + 1 day (proxy for "now")
            anchors = [_parse_occur_ts(h.get("occur_ts", "")) for h in bm25_hits]
            anchors = [a for a in anchors if a is not None]
            query_anchor = max(anchors) if anchors else None
            final = temporal_prior_rerank(dense_hits, query_anchor, lam, top_k=5)

        else:
            raise ValueError(f"unknown variant: {variant}")

        meta["n_final"] = len(final)
        system, user = build_prompt_from_hits(inst, final)
        if no_think:
            user = "/no_think\n" + user
        prompt_data.append((system, user, inst, meta))

    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def _process(idx, system, user, inst, meta):
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
            result["question_id"] = inst.get("question_id", inst.get("instance_id", f"b2_{idx}"))
            result["dimension"] = inst.get("dimension", "")
            result["filter_meta"] = meta
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
                print(f"  [{variant} s{seed}] {done}/{len(instances)} done", file=sys.stderr)
            return result

    tasks = [_process(i, s, u, inst, m) for i, (s, u, inst, m) in enumerate(prompt_data)]
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
        "variant": variant, "seed": seed, "lambda": lam,
        "reader_model": model_name, "judge_model": judge_model,
        "dims": dims, "fallback_count": fallback_count,
        "by_dimension": {
            dim: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for dim, v in by_dim.items()
        },
        "latency_s": time.time() - t0,
    }
    print(f"\n[{variant} s{seed} λ={lam}] DONE: {correct}/{total} = {correct/total if total else 0:.4f}",
          file=sys.stderr)
    for dim, v in sorted(by_dim.items()):
        acc = v["correct"] / v["total"] if v["total"] else 0
        print(f"  {dim}: {v['correct']}/{v['total']} = {acc:.4f}", file=sys.stderr)
    if fallback_count:
        print(f"  fallback_count: {fallback_count}/{total}", file=sys.stderr)
    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="B2: stronger RAG variants")
    parser.add_argument("--variant", required=True,
                        choices=["rag_hybrid_rrf", "rag_entity_filter", "rag_temporal_prior"])
    parser.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5,
                        help="temporal_prior λ; ignored by other variants")
    parser.add_argument("--model-name", default="qwen3-8b")
    parser.add_argument("--reader-endpoint", default="http://127.0.0.1:8110/v1")
    parser.add_argument("--reader-api-key", default="EMPTY")
    parser.add_argument("--judge-endpoint", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini-2024-07-18")
    parser.add_argument("--openrouter-api-key", default=os.getenv("OPENROUTER_API_KEY", ""))
    parser.add_argument("--dims", nargs="+",
                        default=["d1_conflict", "d3_confabulation", "d4_permission"])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--no-think", action="store_true", default=True)
    args = parser.parse_args()

    if not args.openrouter_api_key:
        print("ERROR: --openrouter-api-key or OPENROUTER_API_KEY required", file=sys.stderr)
        sys.exit(1)

    _OUT_BASE.mkdir(parents=True, exist_ok=True)
    suffix = f"_lam{args.lam}" if args.variant == "rag_temporal_prior" else ""
    out_path = _OUT_BASE / f"evaluation_results_b2_{args.variant}_8b_s{args.seed}{suffix}_4omini.json"

    result = asyncio.run(run_variant(
        variant=args.variant, seed=args.seed, lam=args.lam,
        model_name=args.model_name,
        reader_endpoint=args.reader_endpoint, reader_api_key=args.reader_api_key,
        judge_endpoint=args.judge_endpoint, judge_model=args.judge_model,
        judge_api_key=args.openrouter_api_key,
        dims=args.dims, concurrency=args.concurrency, no_think=args.no_think,
    ))
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
