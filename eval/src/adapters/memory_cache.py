"""MemoryCacheAdapter — answering-side replay of a pre-built memory-system cache.

The companion script `scripts/reproduce/build_memory_cache.py` runs day-by-day ingest +
end-of-day search against a real memory system (Mem0 / Memobase / MemOS) and
writes the retrieved memories to a JSONL file. This adapter replays that file
during the answering phase: search() looks up the row by question_id (the
namespace argument is only used for sanity-checking) and returns the stored
hits as SearchHits.

This decouples the expensive ingest+extraction phase (run once per
(system, extractor) pair on H200) from the cheap answering phase (run once per
answerer model). It also makes the eval fully reproducible from a frozen cache
artifact.

Cache row schema (one JSON object per line):
    {
      "instance_id":         "d7_346bef9dac2b",
      "namespace":           "leila_karimi",        # ego_agent_id
      "asker_id":            "rania_haddad",
      "query":               "What does Leila ...",
      "query_timestamp":     7.5375,
      "ingest_day_cutoff":   7,
      "system":              "mem0",                # mem0 | memobase | memos
      "config":              "A_paired",            # A_paired | B_remote
      "extractor_model":     "Qwen3-8B-AWQ",
      "extractor_endpoint":  "local_sglang",
      "memories": [
        {"id": "...", "text": "Leila said ...", "score": 0.87,
         "occur_ts": "...", "thread_id": "..."},
        ...
      ],
      "memory_count":        5,
      "retrieve_latency_ms": 42,
      "error":               null                   # or "MEMORY_ERROR: ..." string
    }

If `memories` is null OR `error` is non-null, search() returns an empty list and
the upstream answering pipeline is expected to mark the row as errored
(prediction=None, pre_scored.reason='MEMORY_ERROR'). This is the same
contract used for LLM_ERROR rows (see eval/src/answering.py and
scripts/patch_llm_errors.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import PromptStyle, RetrievalAdapter
from ..types import MessageEntry, SearchHit


class MemoryCacheAdapter(RetrievalAdapter):
    """Replay a pre-built memory-system retrieval cache.

    Construct with the path to a JSONL cache file. add() is a no-op (the
    cache is frozen). search() returns the cached hits for the question_id
    encoded in the search query (or by namespace+query lookup as fallback).
    """

    def __init__(
        self,
        *,
        cache_path: Path | str,
        expected_extractor: Optional[str] = None,
        expected_system: Optional[str] = None,
        expected_config: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        """Args:
            cache_path: JSONL file produced by scripts/reproduce/build_memory_cache.py
            expected_extractor: if set, every row must match
                cfg.extractor_model (e.g. "Qwen3-8B-AWQ" or
                "anthropic/claude-3.5-haiku"). Asserted on load.
            expected_system: if set, every row must match (e.g. "mem0")
            expected_config: if set, every row must match ("A_paired" or "B_remote")
            strict: when True, mismatched-extractor / missing-row lookups raise.
                When False, missing rows return [] (search is best-effort).
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"memory cache not found: {cache_path}")
        self.cache_path = cache_path
        self.strict = strict

        # Index rows by instance_id (preferred lookup) and by (namespace, query) hash
        # in case the answering side only knows the query string.
        self._by_iid: Dict[str, dict] = {}
        self._by_ns_q: Dict[tuple, dict] = {}
        self._n_errored = 0
        self._n_loaded = 0

        for line in cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            self._n_loaded += 1

            if expected_extractor is not None and row.get("extractor_model") != expected_extractor:
                raise ValueError(
                    f"cache extractor mismatch in {cache_path}: "
                    f"expected {expected_extractor}, got {row.get('extractor_model')!r} "
                    f"on instance_id={row.get('instance_id')!r}"
                )
            if expected_system is not None and row.get("system") != expected_system:
                raise ValueError(
                    f"cache system mismatch: expected {expected_system}, got {row.get('system')!r}"
                )
            if expected_config is not None and row.get("config") != expected_config:
                raise ValueError(
                    f"cache config mismatch: expected {expected_config}, got {row.get('config')!r}"
                )

            iid = str(row.get("instance_id") or "")
            if iid:
                self._by_iid[iid] = row
            ns = str(row.get("namespace") or "")
            q = str(row.get("query") or "")
            if ns and q:
                self._by_ns_q[(ns, q)] = row

            if row.get("error") or row.get("memories") is None:
                self._n_errored += 1

    @property
    def prompt_style(self) -> PromptStyle:
        # Memory systems return short distilled fact snippets. They should be
        # passed through the retrieval-context path, not rebuilt from raw
        # corpus sessions.
        return PromptStyle.JSON_CONTEXT

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        # Cache is pre-built; ingest is a no-op at answering time.
        return {
            "namespace": namespace,
            "indexed_messages": 0,
            "mode": "memory_cache_replay",
            "cache_path": str(self.cache_path),
            "note": "ingest skipped — cache is frozen, see scripts/reproduce/build_memory_cache.py",
        }

    async def search(
        self,
        *,
        namespace: str,
        query: str,
        top_k: int,
        instance_id: Optional[str] = None,
        ego_id: Optional[str] = None,  # accepted for adapter API compat
        **_: Any,
    ) -> List[SearchHit]:
        """Replay cached memories for one query.

        Lookup priority:
          1) instance_id (most reliable; passed by patched pipeline)
          2) (namespace, query) tuple (fallback)
        """
        row: Optional[dict] = None
        if instance_id and instance_id in self._by_iid:
            row = self._by_iid[instance_id]
        elif (namespace, query) in self._by_ns_q:
            row = self._by_ns_q[(namespace, query)]

        if row is None:
            if self.strict:
                raise KeyError(
                    f"memory cache miss: namespace={namespace!r} query={query[:60]!r} "
                    f"instance_id={instance_id!r} (cache_path={self.cache_path})"
                )
            return []

        # Errored row → empty hits, downstream marks as MEMORY_ERROR
        if row.get("error") or row.get("memories") is None:
            return []

        memories = row.get("memories") or []
        hits: List[SearchHit] = []
        for i, m in enumerate(memories[: max(1, int(top_k))]):
            hits.append(
                SearchHit(
                    msg_id=str(m.get("id") or m.get("msg_id") or f"cache_{i}"),
                    score=float(m.get("score", 1.0 - i / max(len(memories), 1))),
                    text=str(m.get("text") or m.get("memory") or ""),
                    occur_ts=str(m.get("occur_ts") or ""),
                    thread_id=str(m.get("thread_id") or namespace),
                    user_id=None,
                )
            )
        return hits

    async def health(self) -> dict:
        return {
            "ok": True,
            "detail": "memory_cache_replay",
            "cache_path": str(self.cache_path),
            "rows_loaded": self._n_loaded,
            "rows_errored": self._n_errored,
            "latency_ms": 0,
        }
