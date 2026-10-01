from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .base import normalize_score
from .http_base import HttpRetrievalAdapter
from ..types import MessageEntry, SearchHit


class MemosAdapter(HttpRetrievalAdapter):
    def __init__(self, *, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        base_url = str(cfg.get("base_url") or os.getenv("MEMOS_BASE_URL") or "").strip()
        api_key = str(cfg.get("api_key") or os.getenv("MEMOS_API_KEY") or "").strip() or None
        if not base_url:
            raise ValueError("MemosAdapter requires base_url (cfg.memos.base_url or MEMOS_BASE_URL)")
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_s=int(cfg.get("timeout_s") or os.getenv("MEMOS_TIMEOUT_S") or 600),
            max_retries=int(cfg.get("max_retries", 3)),
        )
        self.add_path = str(cfg.get("add_path", "/product/add"))
        self.search_path = str(cfg.get("search_path", "/product/search"))
        # Process-wide cap on in-flight /product/add chunks. Without
        # this, 32 namespaces × 30 chunks each gather'd at once = 960
        # simul-connections per cell, which on 8b+ readers (per-add
        # hold time 5-10s) immediately punches through uvicorn
        # limit_concurrency=100 → 100% RST. Sharing one Semaphore
        # across all add() calls bounds total in-flight to MEMOS_ADD_SEM
        # (default 32) which sits below the 100 limit even with 3
        # parallel build_memory_cache cells (each has its own adapter
        # instance, so 3 × 32 = 96 connections to 3 servers = 32 per
        # server). Override via env if needed.
        self._add_sem: Optional[asyncio.Semaphore] = None
        self._add_sem_size = int(os.getenv("MEMOS_ADD_SEM", "32"))

    def _build_batch_payload(self, namespace: str, batch: List[MessageEntry]) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"[Group: {m.thread_id}][Speaker: {m.user_id}] {m.text}",
                    "chat_time": m.occur_ts,
                }
                for m in batch
            ],
            "user_id": namespace,
            "async_mode": "sync",
        }

    def _build_search_payload(self, namespace: str, query: str, top_k: int) -> dict:
        return {
            "query": query,
            "user_id": namespace,
        }

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        """Batch messages by thread_id, fire all chunks concurrently.

        Mirrors the production config that built memos memcache for
        0_6b/llama3b/7b in the original H200 sweep. Per-namespace
        chunks fire via asyncio.gather, letting MemOS server-side
        ingest queue them in parallel.

        Note on the 8b debug session: an earlier branch serialised
        chunks here to mitigate `Can not write request body` errors,
        but the real cause was stale neo4j volumes accumulating across
        runs (causing O(N) MATCH scans and uvicorn hangs). That root
        cause is fixed in run_l_all_models.sh teardown, so the
        original gather-based approach is restored. Loud error logging
        + count-error-as-n preserved for visibility.
        """
        t0 = time.time()
        max_batch = int(
            os.getenv("MEMOS_MAX_BATCH")
            or 15  # H200 + SGLang default; override via env if needed
        )

        if not messages:
            return {
                "namespace": namespace,
                "indexed_messages": 0,
                "failed_messages": 0,
                "batches": 0,
                "n_errors": 0,
                "first_errors": [],
                "latency_ms": 0,
                "backend_meta": {"adapter": "memos"},
            }

        # Group messages by thread, preserving order
        by_thread: "OrderedDict[str, List[MessageEntry]]" = OrderedDict()
        for m in messages:
            by_thread.setdefault(m.thread_id, []).append(m)

        # Build every chunk up front so we can fire them all via gather
        chunks: List[Tuple[str, List[MessageEntry]]] = []
        for tid, thread_msgs in by_thread.items():
            for i in range(0, len(thread_msgs), max_batch):
                chunks.append((tid, thread_msgs[i : i + max_batch]))

        # Lazily create the process-wide chunk semaphore (must be
        # bound to the running event loop, so we can't construct it in
        # __init__ which may run before the loop exists).
        if self._add_sem is None:
            self._add_sem = asyncio.Semaphore(self._add_sem_size)

        async def _insert_chunk(tid: str, chunk: List[MessageEntry]) -> Tuple[int, Optional[str]]:
            payload = self._build_batch_payload(namespace, chunk)
            async with self._add_sem:
                try:
                    await self._request_with_retry("POST", self.add_path, json_body=payload)
                    return (len(chunk), None)
                except Exception as e:
                    return (
                        -len(chunk),
                        f"{type(e).__name__}: {str(e)[:200]} "
                        f"(tid={tid} n={len(chunk)})",
                    )

        results = await asyncio.gather(*(_insert_chunk(tid, ch) for tid, ch in chunks))

        ok = 0
        failed = 0
        first_errors: List[str] = []
        for n_or_neg, err in results:
            if n_or_neg >= 0:
                ok += n_or_neg
            else:
                failed += -n_or_neg
                if err is not None and len(first_errors) < 3:
                    first_errors.append(err)

        if failed > 0:
            print(
                f"[memos_adapter.add] ns={namespace}: {ok} ok, {failed} errors "
                f"across {len(chunks)} chunks; first: {first_errors}",
                file=sys.stderr,
            )

        return {
            "namespace": namespace,
            "indexed_messages": ok,
            "failed_messages": failed,
            "n_errors": failed,
            "first_errors": first_errors,
            "batches": len(chunks),
            "latency_ms": int((time.time() - t0) * 1000),
            "backend_meta": {"adapter": "memos"},
        }

    async def search(self, *, namespace: str, query: str, top_k: int, **_: Any) -> List[SearchHit]:
        # Scope kwargs (ego_id, query_timestamp, ...) are ignored: the store
        # is namespaced per ego at ingest and holds what was ingested.
        payload = self._build_search_payload(namespace, query, top_k)
        data = await self._request_with_retry("POST", self.search_path, json_body=payload)

        # MemOS returns structured response: data.text_mem, data.act_mem, etc.
        # Each category contains [{cube_id, memories: [...], total_nodes}]
        # Collect memories from all categories
        rows: list = []
        inner = data.get("data") or data
        if isinstance(inner, dict):
            for key in ("text_mem", "act_mem", "para_mem", "pref_mem",
                        "memory_detail_list", "hits", "results"):
                entries = inner.get(key)
                if not entries:
                    continue
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and "memories" in entry:
                            rows.extend(entry["memories"])
                        elif isinstance(entry, dict):
                            rows.append(entry)
        elif isinstance(inner, list):
            rows = inner

        rows = rows[: max(1, int(top_k))]

        # MemOS v1.0 uses "memory" field; older versions use "memory_value"/"text"
        # Score may be absent — assign descending rank scores if missing
        has_scores = any(r.get("score") is not None or r.get("relevance_score") is not None for r in rows)

        out: List[SearchHit] = []
        for i, r in enumerate(rows):
            text = str(
                r.get("memory") or r.get("memory_value") or r.get("text")
                or r.get("content") or r.get("memory_content") or r.get("value") or ""
            )
            if has_scores:
                raw_score = float(r.get("score") or r.get("relevance_score") or 0.0)
            else:
                raw_score = 1.0 - (i / max(len(rows), 1))  # rank-based score

            out.append(
                SearchHit(
                    msg_id=str(r.get("msg_id") or r.get("id") or r.get("memory_id") or f"memos_{i}"),
                    score=raw_score,
                    text=text,
                    occur_ts=str(r.get("occur_ts") or r.get("created_at") or r.get("timestamp") or ""),
                    thread_id=str(r.get("thread_id") or ""),
                    user_id=(int(r["user_id"]) if r.get("user_id") is not None else None),
                )
            )
        return out
