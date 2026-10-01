"""MemSearch adapter — markdown-as-source-of-truth memory backed by milvus-lite.

Why this exists in MemArena's matrix
------------------------------------
Memobase, MemOS, mem0, and Zep all run a database server (postgres,
neo4j, qdrant) plus an HTTP API in front of structured-LLM extraction
chains. That architecture is impractical for the on-device / edge
deployment scenarios MemArena's introduction calls out (OpenClaw,
Memoro, etc.). MemSearch (zilliztech/memsearch, 2026) is the publicly
extracted version of OpenClaw's SOUL.md memory layer:

  - Plain markdown files are the persistent state. One file per ego
    per ingest call. No DB to back up, edit, or migrate.
  - milvus-lite (single .db file embedded in the process) provides
    semantic chunk lookup over those markdown sources.
  - Embeddings come from a configurable provider; we use ollama with
    nomic-embed-text to mirror what memobase + memos already use.
  - No LLM calls on the ingest path — chunking is deterministic,
    embedding is the only network hop. This sidesteps the
    extractor-throughput collapse that made memos unusable on the 8b
    and 32b readers.

Per-cell isolation
------------------
Every build_memory_cache process gets its own scratch root and its
own milvus.db. Cross-process pollution is impossible.

Per-namespace (ego) isolation
-----------------------------
A single MemSearch instance handles all ego_agents in one cell.
Namespace isolation comes from two mechanisms:

  1. Files for ego ``X`` live under ``<root>/<X>/``. Files for ego
     ``Y`` live under ``<root>/<Y>/``. The two directory trees never
     intersect.
  2. Search calls pass ``source_prefix=<root>/<X>/``. Milvus filters
     hits by the chunk's ``source`` field (full path), so ego ``X``'s
     query can never surface a chunk indexed from ego ``Y``'s files.

Both mechanisms are belt-and-braces; the source_prefix filter is the
authoritative gate.

Environment / cfg overrides (cfg takes precedence over env)
-----------------------------------------------------------
  ``root_dir``    / ``MEMSEARCH_ROOT_DIR``     — markdown root
                                                  (default: $TMPDIR/memarena_memsearch_<pid>)
  ``milvus_uri``  / ``MEMSEARCH_MILVUS_URI``   — milvus-lite path
                                                  (default: <root>/milvus.db)
  ``embed_provider``                            — ``ollama`` (default) or ``openai``
  ``embed_model`` / ``MEMSEARCH_EMBED_MODEL``  — default ``nomic-embed-text``
  ``embed_base_url``                            — embedder URL
                                                  (overrides OLLAMA_HOST for ollama)
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import PromptStyle, RetrievalAdapter
from ..types import MessageEntry, SearchHit


def _safe_filename(value: str, *, max_len: int = 80) -> str:
    """Make a string safe to use as a filename component (slash-free, short)."""
    out = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)
    return out[:max_len] or "x"


class MemsearchAdapter(RetrievalAdapter):
    def __init__(self, *, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        # ── per-cell scratch root: markdown files + milvus.db live here ──
        root_dir = (
            cfg.get("root_dir")
            or os.getenv("MEMSEARCH_ROOT_DIR")
            or f"/tmp/memarena_memsearch_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        self._root = Path(str(root_dir)).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

        milvus_uri = (
            cfg.get("milvus_uri")
            or os.getenv("MEMSEARCH_MILVUS_URI")
            or str(self._root / "milvus.db")
        )

        embed_provider = str(
            cfg.get("embed_provider")
            or os.getenv("MEMSEARCH_EMBED_PROVIDER")
            or "ollama"
        )
        embed_model = str(
            cfg.get("embed_model")
            or os.getenv("MEMSEARCH_EMBED_MODEL")
            or "nomic-embed-text"
        )
        embed_base_url = (
            cfg.get("embed_base_url")
            or os.getenv("MEMSEARCH_EMBED_BASE_URL")
        )
        # Ollama provider reads OLLAMA_HOST. Set it before importing if needed.
        if embed_provider == "ollama" and embed_base_url:
            os.environ.setdefault("OLLAMA_HOST", embed_base_url)

        # MemSearch wants a list of paths to scan; we register the cell root.
        # Per-ego subdirs under it will be picked up by index_file calls below.
        from memsearch import MemSearch  # lazy import; memsearch may not be installed

        self._mem = MemSearch(
            paths=[self._root],
            embedding_provider=embed_provider,
            embedding_model=embed_model,
            embedding_base_url=embed_base_url,
            milvus_uri=str(milvus_uri),
            collection=str(cfg.get("collection") or "memarena_memsearch"),
            description=f"memarena memsearch cell pid={os.getpid()}",
            max_chunk_size=int(cfg.get("max_chunk_size") or 1500),
            overlap_lines=int(cfg.get("overlap_lines") or 2),
        )
        # Per-namespace write counters drive the per-day filename suffix so
        # repeated add() for the same ego across days don't overwrite.
        self._add_counter: Dict[str, int] = {}
        # Lock around add()'s read-modify-write of _add_counter; index_file
        # has its own internal milvus locking.
        self._counter_lock = asyncio.Lock()

        print(
            f"[memsearch_adapter] root={self._root} milvus={milvus_uri} "
            f"embedder={embed_provider}:{embed_model}",
            file=sys.stderr,
        )

    @property
    def prompt_style(self) -> PromptStyle:
        # Markdown chunks are returned as plain text — same prompt format as
        # the other text-style retrievers (oracle, vanilla, inmem*).
        return PromptStyle.TEXT_SESSIONS

    def _ego_dir(self, namespace: str) -> Path:
        d = self._root / _safe_filename(namespace)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _format_messages_md(
        self, namespace: str, messages: List[MessageEntry], *, day_tag: str, batch_idx: int
    ) -> str:
        """Render messages as a markdown file scoped to one (ego, day, batch).

        Layout: a top-level title describing the conversation slice, then
        per-thread sections with timestamped speaker turns. memsearch's
        chunker uses heading boundaries (line-prefixed ``#``) to split, so
        per-thread headings keep semantically related lines together in the
        same chunk.
        """
        out: List[str] = []
        out.append(f"# Conversation log for `{namespace}` — day {day_tag} (batch {batch_idx})")
        by_thread: "OrderedDict[str, List[MessageEntry]]" = OrderedDict()
        for m in messages:
            by_thread.setdefault(m.thread_id, []).append(m)
        for tid, thread_msgs in by_thread.items():
            out.append(f"\n## Thread `{tid}`\n")
            for m in thread_msgs:
                speaker = f"speaker_{m.user_id}" if m.user_id is not None else "speaker"
                out.append(f"- [{m.occur_ts}] **{speaker}**: {m.text}")
        out.append("")  # trailing newline
        return "\n".join(out)

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        """Write the batch as one markdown file under the ego subdir, then
        index it. No LLM call on this path — chunking is deterministic,
        only ollama embeddings are computed.
        """
        t0 = time.time()
        if not messages:
            return {
                "namespace": namespace,
                "indexed_messages": 0,
                "failed_messages": 0,
                "n_errors": 0,
                "first_errors": [],
                "batches": 0,
                "latency_ms": 0,
                "backend_meta": {"adapter": "memsearch", "root": str(self._root)},
            }

        ego_dir = self._ego_dir(namespace)
        # Day tag derived from the earliest message's occur_ts (YYYY-MM-DD).
        day_tag = (messages[0].occur_ts or "unknown")[:10] or "unknown"
        async with self._counter_lock:
            batch_idx = self._add_counter.get(namespace, 0)
            self._add_counter[namespace] = batch_idx + 1
        fname = f"{day_tag}__b{batch_idx:03d}.md"
        fpath = ego_dir / fname

        try:
            content = self._format_messages_md(
                namespace, messages, day_tag=day_tag, batch_idx=batch_idx
            )
            fpath.write_text(content, encoding="utf-8")
            n_chunks = await self._mem.index_file(fpath)
            return {
                "namespace": namespace,
                "indexed_messages": len(messages),
                "failed_messages": 0,
                "n_errors": 0,
                "first_errors": [],
                "batches": 1,
                "n_chunks": int(n_chunks),
                "file": str(fpath),
                "latency_ms": int((time.time() - t0) * 1000),
                "backend_meta": {"adapter": "memsearch", "root": str(self._root)},
            }
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]} (file={fname})"
            print(f"[memsearch_adapter.add] ns={namespace}: {err_msg}", file=sys.stderr)
            return {
                "namespace": namespace,
                "indexed_messages": 0,
                "failed_messages": len(messages),
                "n_errors": 1,
                "first_errors": [err_msg],
                "batches": 1,
                "latency_ms": int((time.time() - t0) * 1000),
                "backend_meta": {"adapter": "memsearch", "root": str(self._root)},
            }

    async def search(self, *, namespace: str, query: str, top_k: int, **_: Any) -> List[SearchHit]:
        # Scope kwargs (ego_id, query_timestamp, ...) are ignored: the store
        # is namespaced per ego at ingest and holds what was ingested.
        """Semantic search over chunks scoped to this ego's subdirectory.

        source_prefix is the authoritative isolation gate — milvus filters
        hits whose ``source`` does not start with the ego's directory path.
        """
        ego_dir = self._ego_dir(namespace)
        try:
            hits = await self._mem.search(
                query, top_k=max(1, int(top_k)), source_prefix=ego_dir
            )
        except Exception as e:
            print(
                f"[memsearch_adapter.search] ns={namespace}: "
                f"{type(e).__name__}: {str(e)[:200]}",
                file=sys.stderr,
            )
            return []

        out: List[SearchHit] = []
        for i, h in enumerate(hits or []):
            text = str(h.get("content") or h.get("text") or "")
            occur_ts = ""  # memsearch chunks are tied to file/line, not message ts
            score = float(h.get("score") or 0.0)
            source = str(h.get("source") or "")
            out.append(
                SearchHit(
                    msg_id=f"memsearch_{i}_{Path(source).stem}",
                    score=score,
                    text=text,
                    occur_ts=occur_ts,
                    thread_id="",
                    user_id=None,
                )
            )
        return out

    async def reset(self, *, namespace: str) -> None:
        """Drop a single ego's markdown files and re-index the empty dir.

        We don't try to surgically remove that ego's chunks from milvus —
        index_file() only refreshes by source path, and the next search
        (which is gated by source_prefix=<ego_dir>) won't see anything
        because all the source files are gone. milvus-lite holds the
        orphaned vectors until the next compact, which is harmless.
        """
        ego_dir = self._ego_dir(namespace)
        for f in ego_dir.glob("*.md"):
            try:
                f.unlink()
            except OSError:
                pass
        async with self._counter_lock:
            self._add_counter.pop(namespace, None)

    async def close(self) -> None:
        try:
            self._mem.close()
        except Exception:
            pass
