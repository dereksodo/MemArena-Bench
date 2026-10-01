#!/usr/bin/env python3
"""GraphRAG v2 — Step C: HippoRAG retrieval, emit memcache JSONL.

Reads:
  - hipporag_index/      built by build_hipporag_index.py
  - masim_qa_*.json      the 1579 queries to answer per seed

Per query, calls HippoRAG.retrieve(query, num_to_retrieve=5),
takes the returned (docs, doc_scores), and emits one JSONL line
in the same schema as memcache_memos_A_paired_*.jsonl so the
existing eval.cli --system memory_cache pipeline consumes it.

  {
    "instance_id": ...,
    "namespace": ...,
    "asker_id": ...,
    "query": ...,
    "system": "hipporag",
    "config": "A_paired",
    "memories": [
      {"id": "doc_<idx>", "text": "<passage>", "score": float, ...},
      ...
    ]
  }

Run from the venv:
    ~/venvs/hipporag/bin/python scripts/run_hipporag_retrieval.py \
      --index-dir MASim/runs/.../hipporag_index \
      --qa        MASim/runs/.../masim_qa_memcache_memos_A_paired_qwen3_8b.json \
      --out       MASim/runs/.../memcache_hipporag_A_paired_qwen3_8b.jsonl \
      --top-k 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index-dir", required=True, dest="index_dir")
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=5, dest="top_k")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1",
                    dest="llm_base_url")
    ap.add_argument("--embedding-model", default="nvidia/NV-Embed-v2",
                    dest="embedding_model")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="how many queries to retrieve at a time")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set — export OpenRouter key first")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[hpret] loading qa from {args.qa}", flush=True)
    qa = json.load(Path(args.qa).open())
    qars = qa.get("qars", qa) if isinstance(qa, dict) else qa
    print(f"[hpret]   {len(qars)} queries", flush=True)

    from hipporag import HippoRAG
    print(f"[hpret] HippoRAG load: index_dir={args.index_dir}", flush=True)
    hr = HippoRAG(
        save_dir=args.index_dir,
        llm_model_name=args.llm_model,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_model,
    )
    if not getattr(hr, "ready_to_retrieve", False):
        print("[hpret] preparing retrieval objects ...", flush=True)
        hr.prepare_retrieval_objects()
    print(f"[hpret] ready_to_retrieve={hr.ready_to_retrieve}", flush=True)

    queries: list[str] = []
    meta_per_q: list[dict] = []
    for q in qars:
        queries.append(q.get("Q") or q.get("query") or "")
        m = q.get("meta") or {}
        meta_per_q.append({
            "instance_id": q.get("id"),
            "namespace": m.get("ego_agent_id") or m.get("answerer_agent_id") or "",
            "asker_id": m.get("asker_agent_id") or "",
        })

    print(f"[hpret] retrieving in batches of {args.batch_size}", flush=True)
    t0 = time.time()
    fout = out_path.open("w")
    n_done = 0
    for i in range(0, len(queries), args.batch_size):
        chunk = queries[i : i + args.batch_size]
        chunk_meta = meta_per_q[i : i + args.batch_size]
        results = hr.retrieve(queries=chunk, num_to_retrieve=args.top_k)
        # results is List[QuerySolution] (or tuple if extra return values)
        if isinstance(results, tuple):
            results = results[0]
        for sol, m in zip(results, chunk_meta):
            memories = []
            docs = list(sol.docs) if sol.docs is not None else []
            scores = list(sol.doc_scores) if sol.doc_scores is not None else []
            for j, doc in enumerate(docs):
                memories.append({
                    "id": f"hp_doc_{j}",
                    "text": doc,
                    "score": float(scores[j]) if j < len(scores) else 0.0,
                    "occur_ts": "",
                    "thread_id": "",
                })
            rec = {
                "instance_id": m["instance_id"],
                "namespace": str(m["namespace"]),
                "asker_id": str(m["asker_id"]),
                "query": sol.question,
                "query_timestamp": None,
                "ingest_day_cutoff": 0,
                "system": "hipporag",
                "config": "A_paired",
                "extractor_model": "default",
                "extractor_endpoint": "local_sglang",
                "memories": memories,
            }
            fout.write(json.dumps(rec) + "\n")
            n_done += 1
        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-3)
        eta = (len(queries) - n_done) / max(rate, 1e-3)
        print(f"[hpret] {n_done}/{len(queries)} ({rate:.2f}/s, eta {eta/60:.1f}min)",
              flush=True)
        fout.flush()
    fout.close()
    print(f"[hpret] DONE → {out_path} ({n_done} queries in {(time.time()-t0)/60:.1f}min)",
          flush=True)


if __name__ == "__main__":
    main()
