#!/usr/bin/env python3
"""GraphRAG v2 — Step B: build HippoRAG index over the corpus.

Loads ALL sessions from masim_transcript_*.jsonl, groups messages
by meta.session_id, renders a session text per group, and indexes
the whole corpus with HippoRAG (NeurIPS 2024).

Per the GraphRAG v2 plan in d779b02:
  - llm: gpt-4o-mini via OpenRouter (set OPENAI_BASE_URL +
         OPENAI_API_KEY in env)
  - embedding: nvidia/NV-Embed-v2 (HippoRAG default; runs on
               whatever GPU CUDA_VISIBLE_DEVICES points to)

The save_dir lands the persistent KG + parquet embeddings.

Run from the venv (NOT the main env):
    ~/venvs/hipporag/bin/python scripts/build_hipporag_index.py \
      --transcript MASim/runs/.../masim_transcript_memcache_memos_A_paired_qwen3_8b.jsonl \
      --save-dir   MASim/runs/.../hipporag_index/

Note: the corpus (and therefore this index) is byte-identical
across s2/s3/s4. Build ONCE under e.g. eval_results_s2/.../, the
seeds reuse it via symlink or env override.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


def render_session_text(messages: list[tuple[str, str]], max_chars: int = 4000) -> str:
    parts: list[str] = []
    total = 0
    for speaker, text in messages:
        chunk = f"{speaker}: {text}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk) + 1
    return "\n".join(parts)


def load_corpus(transcript_path: Path, max_chars: int) -> list[str]:
    by_sess: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with transcript_path.open() as f:
        for line in f:
            r = json.loads(line)
            meta = r.get("meta", {}) or {}
            sid = meta.get("session_id")
            text = (r.get("text") or "").strip()
            if not sid or not text:
                continue
            speaker = str(meta.get("speaker_slug") or meta.get("speaker_name") or
                          r.get("user_id", ""))
            by_sess[sid].append((speaker, text))
    docs: list[str] = []
    for sid in by_sess:
        docs.append(render_session_text(by_sess[sid], max_chars=max_chars))
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--save-dir", required=True, dest="save_dir")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1",
                    dest="llm_base_url")
    ap.add_argument("--embedding-model", default="nvidia/NV-Embed-v2",
                    dest="embedding_model")
    ap.add_argument("--max-session-chars", type=int, default=4000,
                    dest="max_session_chars",
                    help="cap per-session text (NV-Embed-v2 has a 32K context)")
    ap.add_argument("--limit", type=int, default=0,
                    help="take only first N sessions (debug); 0 = all")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        # Reuse the OpenRouter key
        sys.exit("OPENAI_API_KEY not set — export the OpenRouter key first")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hpidx] loading corpus from {args.transcript}", flush=True)
    docs = load_corpus(Path(args.transcript), max_chars=args.max_session_chars)
    print(f"[hpidx] {len(docs)} sessions loaded", flush=True)
    if args.limit > 0:
        docs = docs[: args.limit]
        print(f"[hpidx] limited to first {len(docs)} sessions", flush=True)

    from hipporag import HippoRAG
    print(f"[hpidx] HippoRAG init: save_dir={save_dir}, llm={args.llm_model} "
          f"@ {args.llm_base_url}, embed={args.embedding_model}", flush=True)
    hr = HippoRAG(
        save_dir=str(save_dir),
        llm_model_name=args.llm_model,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_model,
    )

    print(f"[hpidx] indexing {len(docs)} docs ...", flush=True)
    t0 = time.time()
    hr.index(docs=docs)
    print(f"[hpidx] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
