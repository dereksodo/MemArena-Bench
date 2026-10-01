"""Mem0 SDK adapter — uses mem0ai package directly, no HTTP server needed.

LLM extraction can be routed two ways:
  Config A (paired, on-device): point at the shared SGLang endpoint serving
    the same model used for answering. Default behavior.
  Config B (remote constant): point at OpenRouter (or any OpenAI-compatible
    endpoint) and use a hosted model like Claude 3.5 Haiku for extraction.

The embedder always runs locally via sentence-transformers.
The vector store always runs locally (Qdrant on /tmp by default).

Environment / cfg overrides (cfg takes precedence over env):
  llm_endpoint  / MEM0_LLM_ENDPOINT       — base URL for LLM API
  llm_model     / MEM0_LLM_MODEL          — model id (default: "default" for SGLang)
  llm_api_key   / MEM0_LLM_API_KEY        — bearer token (default: "EMPTY" for SGLang)
  embedder_provider / MEM0_EMBEDDER       — default "huggingface"
  embedder_model    / MEM0_EMBEDDER_MODEL — default all-MiniLM-L6-v2
  collection_name   / MEM0_COLLECTION     — Qdrant collection (default memarena_mem0)
  qdrant_path       / MEM0_QDRANT_PATH    — Qdrant on-disk path
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from .base import PromptStyle, RetrievalAdapter
from ..types import MessageEntry, SearchHit


# ─── mem0 log noise suppression ──────────────────────────────────────────────
# Mem0 1.0.x has two known noisy warning/error paths that don't affect
# correctness but flood stderr and make it hard to grep real errors:
#
#   1. sqlite history.db: "cannot commit - no transaction is active"
#      — mem0 calls COMMIT on a connection where the transaction was
#      already auto-closed by an earlier statement. Benign.
#
#   2. "Error processing memory action: {'id': 'N', ..., 'event': 'UPDATE',
#      'old_memory': '[]'}, Error: 'N'" — LLM hallucinated an UPDATE
#      action on a memory id that doesn't exist yet. Happens on very
#      small memory stores when the extractor LLM over-reaches. Benign.
#
# We install a logging Filter on the `mem0` logger that drops only these
# two patterns — any other mem0 error (LLM failure, Qdrant connection,
# etc.) passes through unchanged.

class _Mem0NoiseFilter(logging.Filter):
    _patterns = [
        re.compile(r"cannot commit.*no transaction is active", re.IGNORECASE),
        re.compile(r"no transaction is active", re.IGNORECASE),
        re.compile(r"Error processing memory action:.*Error:\s*'", re.IGNORECASE),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        for pat in self._patterns:
            if pat.search(msg):
                return False
        return True


# Install the noise filter on everywhere records might emerge. Python
# logging filters attached to a LOGGER only run at the originating
# call site (they're not consulted during propagation up the parent
# chain), so we have to attach to the HANDLERS that ultimately write
# the records to stderr. We cover:
#   1. `logging.lastResort` — the default handler used when no
#      explicit handlers are attached anywhere in the hierarchy
#      (this is mem0's default state as of 1.0.x — it doesn't
#      configure its own handlers, so records propagate up to root
#      and then to lastResort).
#   2. Every handler attached to the root logger (handles the
#      `logging.basicConfig()` case).
#   3. Every handler attached to any existing `mem0.*` logger found
#      in the manager.loggerDict (belt-and-suspenders for versions
#      that DO install handlers on submodule loggers).
# Idempotent via a marker attribute on the filter instance.

def _install_mem0_noise_filter() -> None:
    marker = "_memarena_mem0_noise_filter"
    flt = _Mem0NoiseFilter()

    def _attach_once(handler: logging.Handler) -> None:
        if getattr(handler, marker, False):
            return
        handler.addFilter(flt)
        setattr(handler, marker, True)

    # (1) lastResort handles the "no configured handlers" case that
    # triggers when mem0 emits records with default logging setup.
    if logging.lastResort is not None:
        _attach_once(logging.lastResort)

    # (2) root logger handlers (basicConfig case)
    for h in logging.getLogger().handlers:
        _attach_once(h)

    # (3) any mem0.* loggers that already have their own handlers
    for name in list(logging.Logger.manager.loggerDict):
        if not (name == "mem0" or name.startswith("mem0.")):
            continue
        lg = logging.getLogger(name)
        for h in lg.handlers:
            _attach_once(h)


_install_mem0_noise_filter()


# ─── qdrant local_collection write serialization ────────────────────────────
# mem0 1.0.4's AsyncMemory.add() fires concurrent qdrant calls via
# `asyncio.gather(process_fact_for_search(...) for fact in new_retrieved_facts)`,
# and each `process_fact_for_search` calls `asyncio.to_thread(vector_store.search)`.
# qdrant's LOCAL mode (which mem0 uses when we pass it an on-disk path)
# is **not** thread-safe: concurrent read/write threads race on
# `LocalCollection.deleted` and the internal vectors array, producing
#
#   IndexError: index N is out of bounds for axis 0 with size N
#   ValueError: operands could not be broadcast together with shapes (N,) (N-2,)
#
# during `scores & ~self.deleted` mask operations. The race is inside
# mem0's own code, so an asyncio.Lock at our adapter layer cannot help
# (mem0 holds no lock across its internal gather).
#
# Monkey-patch the LocalCollection mutating methods and `search` with a
# shared threading.Lock applied once per process. This serializes every
# qdrant operation from ANY mem0 call site. Idempotent via a class-level
# sentinel, so importing this module multiple times is safe.

def _install_qdrant_local_lock() -> None:
    import threading
    try:
        from qdrant_client.local.local_collection import LocalCollection
    except ImportError:
        return
    if getattr(LocalCollection, "_memarena_locked", False):
        return
    _GLOBAL_LC_LOCK = threading.Lock()
    for method_name in (
        "search",
        "search_groups",
        "upsert",
        "delete",
        "delete_vectors",
        "delete_payload",
        "retrieve",
        "scroll",
        "count",
    ):
        original = getattr(LocalCollection, method_name, None)
        if original is None:
            continue

        def _make_wrapped(fn):
            def _wrapped(self, *args, **kwargs):
                with _GLOBAL_LC_LOCK:
                    return fn(self, *args, **kwargs)
            _wrapped.__name__ = fn.__name__
            _wrapped.__doc__ = fn.__doc__
            return _wrapped

        setattr(LocalCollection, method_name, _make_wrapped(original))
    LocalCollection._memarena_locked = True


# Phase 7.0 follow-up (2026-04-15): the threading.Lock above is
# NOT a cosmetic monkey-patch. It is a load-bearing bug fix for a
# real mem0 1.0.4 × qdrant LocalCollection race in the async add
# path. Phase 7.0.2's vanilla smoke (H200 commit 9136caf) showed
# that without this lock, mem0's concurrent asyncio.gather over
# per-batch facts collapses the qdrant write success rate to
# 5-15% and corrupts the collection with shape-broadcast errors.
# Phase 6's 2-agent control run never exposed this because it ran
# serially (one add per day, no gather), and its 44% exception
# rate came from a different, synchronous-JSON-parse failure mode.
# We therefore keep this patch installed unconditionally, even in
# "vanilla" mode, and explicitly document it as a bug fix rather
# than a driver-layer intervention in the paper's Discussion.
_install_qdrant_local_lock()


# ─── OpenRouter provider routing (force Anthropic, skip Bedrock) ────────────
# When mem0's LLM points at OpenRouter and the served model is a Claude
# family, the route defaults to whichever provider OpenRouter picks. We
# observed Amazon Bedrock's Claude 3.5 Haiku silently returning empty
# `content: ""` after day 4 of a 50-agent ingest, which mem0 parses as
# `Expecting value: line 1 column 1 (char 0)` and eventually stalls in
# retry loops. See AWS re:Post "Empty responses from Claude 3 Haiku on
# Bedrock" and mem0ai/mem0#4054 for matching reports.
#
# Fix: inject `extra_body={"provider": {"order": ["Anthropic"],
# "allow_fallbacks": false}}` on every chat.completions.create call
# whose base_url is OpenRouter. Pins upstream to Anthropic's
# first-party endpoint, bypassing Bedrock entirely. Idempotent.

def _install_openrouter_provider_pin() -> None:
    try:
        from mem0.llms import openai as _mem0_openai_mod
    except ImportError:
        return
    cls = getattr(_mem0_openai_mod, "OpenAILLM", None)
    if cls is None or getattr(cls, "_memarena_openrouter_pinned", False):
        return
    original_generate = cls.generate_response

    def _patched_generate(self, messages, response_format=None, tools=None,
                          tool_choice="auto", **kwargs):
        base_url = ""
        try:
            base_url = str(getattr(self.client, "base_url", "") or "")
        except Exception:
            pass
        if "openrouter" in base_url.lower():
            existing = kwargs.get("extra_body") or {}
            if isinstance(existing, dict) and "provider" not in existing:
                existing = dict(existing)
                # Exclude Amazon Bedrock to avoid its silent empty-
                # response bug on Claude 3.5 Haiku. Some OpenRouter
                # model slugs (e.g. anthropic/claude-3.5-haiku) don't
                # have an Anthropic first-party endpoint at all, so we
                # can't pin to Anthropic exclusively — we have to
                # block Bedrock specifically and let whatever remains
                # (Anthropic first-party or Google Vertex) serve it.
                existing["provider"] = {
                    "ignore": ["amazon-bedrock"],
                    "allow_fallbacks": True,
                }
                kwargs["extra_body"] = existing
        return original_generate(
            self,
            messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    cls.generate_response = _patched_generate
    cls._memarena_openrouter_pinned = True


import os as _phase5_os
if (
    _phase5_os.environ.get("PHASE5_DISABLE_OPENROUTER_PIN", "0") != "1"
    and _phase5_os.environ.get("MEMARENA_MEM0_VANILLA", "0") != "1"
):
    _install_openrouter_provider_pin()


class Mem0SdkAdapter(RetrievalAdapter):
    """Adapter using Mem0's Python SDK with configurable LLM + local embedder."""

    def __init__(self, *, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}

        # Phase 7.0 vanilla mode — when True, skip server-mode auto-
        # detection (stay in LOCAL qdrant) and skip the Qwen3 /no_think
        # prompt override. Paired with the module-level
        # MEMARENA_MEM0_VANILLA env-var gate that disables the
        # qdrant LocalCollection lock and the OpenRouter provider pin.
        vanilla = bool(cfg.get("vanilla", False))
        self._vanilla = vanilla

        llm_endpoint = cfg.get("llm_endpoint", os.getenv("MEM0_LLM_ENDPOINT", "http://127.0.0.1:8100/v1"))
        llm_model = cfg.get("llm_model", os.getenv("MEM0_LLM_MODEL", "default"))
        llm_api_key = cfg.get("llm_api_key", os.getenv("MEM0_LLM_API_KEY", "EMPTY"))
        # Mem0's fact-extraction prompt asks for a large structured JSON
        # response (facts array + actions). Default 1024 is too tight for
        # smaller models — Qwen3-0.6B truncates mid-string and mem0 then
        # fails with "Unterminated string starting at ...". Bump to 4096
        # so weak extractors at least get a fighting chance; strong
        # extractors are unaffected (they produce short outputs anyway).
        llm_max_tokens = int(cfg.get("llm_max_tokens", os.getenv("MEM0_LLM_MAX_TOKENS", "4096")))

        embedder_provider = cfg.get("embedder_provider", os.getenv("MEM0_EMBEDDER", "huggingface"))
        embedder_model = cfg.get("embedder_model", os.getenv("MEM0_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

        collection_name = cfg.get("collection_name", os.getenv("MEM0_COLLECTION", "memarena_mem0"))

        # Vector store mode: HTTP server (preferred) or local on-disk.
        # Server mode: set `qdrant_host` + `qdrant_port` (or
        # MEM0_QDRANT_HOST / MEM0_QDRANT_PORT). Defaults to localhost:6333
        # if a qdrant server is reachable at startup; otherwise falls back
        # to local on-disk mode via `qdrant_path`.
        #
        # Why server mode: mem0 1.0.4 calls qdrant from multiple threads
        # via asyncio.to_thread within a single .add() (it gathers
        # process_fact_for_search concurrently). qdrant LOCAL mode is not
        # thread-safe — concurrent reads/writes race on .deleted/.vectors
        # producing `IndexError: index N is out of bounds` and broadcast
        # shape mismatches at 50-agent scale. Server mode runs qdrant as
        # a proper multi-threaded service and eliminates the race.
        explicit_host = cfg.get("qdrant_host") or os.getenv("MEM0_QDRANT_HOST")
        explicit_port = cfg.get("qdrant_port") or os.getenv("MEM0_QDRANT_PORT")
        use_server_mode = bool(explicit_host) or bool(explicit_port)
        if not use_server_mode and not vanilla:
            # Auto-detect a running qdrant server on localhost:6333.
            # In vanilla mode we stay in LOCAL mode unconditionally so
            # the measurement surface matches Phase 5.2's control script.
            try:
                import urllib.request
                with urllib.request.urlopen("http://127.0.0.1:6333/collections", timeout=0.5) as r:
                    if r.status == 200:
                        use_server_mode = True
                        explicit_host = "127.0.0.1"
                        explicit_port = 6333
            except Exception:
                pass

        if use_server_mode:
            self._qdrant_server_mode = True
            self._qdrant_host = str(explicit_host or "127.0.0.1")
            self._qdrant_port = int(explicit_port or 6333)
            self._qdrant_path = None
            self._owns_qdrant_path = False
        else:
            self._qdrant_server_mode = False
            # Local on-disk fallback.
            explicit_path = cfg.get("qdrant_path") or os.getenv("MEM0_QDRANT_PATH")
            if explicit_path:
                qdrant_path = explicit_path
                self._owns_qdrant_path = False
            else:
                qdrant_path = tempfile.mkdtemp(prefix="mem0_qdrant_")
                self._owns_qdrant_path = True
            self._qdrant_path = qdrant_path
            self._qdrant_host = None
            self._qdrant_port = None

        # Per-run sqlite history DB path — concurrent mem0 runs used to
        # share ~/.mem0/history.db and race each other into readonly
        # errors. Caller can pin a per-run path; otherwise we use a
        # unique tempfile. close() cleans it up if we own it.
        explicit_history_db = cfg.get("history_db_path") or os.getenv("MEM0_HISTORY_DB_PATH")
        if explicit_history_db:
            history_db_path = explicit_history_db
            self._owns_history_db = False
        else:
            _fd, history_db_path = tempfile.mkstemp(prefix="mem0_history_", suffix=".db")
            os.close(_fd)
            self._owns_history_db = True
        self._history_db_path = history_db_path

        # Embedding dimensions lookup
        dim_map = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "BAAI/bge-m3": 1024,
        }
        embed_dims = dim_map.get(embedder_model, 384)

        # Use AsyncMemory (mem0 1.0+) instead of sync Memory. AsyncMemory
        # exposes await-able add/search/delete_all backed by AsyncOpenAI
        # and AsyncQdrantClient, which lets concurrent callers actually
        # hit the SGLang endpoint in parallel. The sync Memory class
        # holds a single httpx client whose internal lock forced every
        # asyncio.to_thread wrapper to serialize at the transport layer
        # (observed on H200 as #running-req: 1 in SGLang even with
        # a concurrent driver).
        try:
            from mem0 import AsyncMemory
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Mem0SdkAdapter requires mem0ai >= 1.0 with AsyncMemory. "
                "Upgrade with `pip install -U 'mem0ai>=1.0'`."
            ) from e

        # mem0 1.0.4's AsyncMemory.from_config is itself async, so we
        # can't instantiate it from a synchronous __init__. Stash the
        # config + class reference and lazy-init on first async call
        # via _ensure_mem(), gated by an asyncio.Lock so concurrent
        # first-callers don't double-initialize.
        self._AsyncMemory_cls = AsyncMemory
        self._mem: Any = None  # populated on first _ensure_mem()
        self._init_lock = asyncio.Lock()
        # Global write-lock around mem0 mutating ops. Qdrant's local
        # mode (what mem0 uses when we pass it an on-disk path) has a
        # read-after-write race in `_payload_and_non_deleted_mask`: if
        # two coroutines mutate the shared collection concurrently,
        # the second one's argsort indices reference a mask whose
        # length is one step behind, producing
        # `IndexError: index N is out of bounds for axis 0 with size N`.
        # We force add/search/delete through a single lock instance.
        # Cross-agent parallelism still happens at the SGLang layer
        # (extractor LLM call is outside the lock) via asyncio.gather
        # across multiple adapter instances, but each adapter only
        # serves one run, so within-run concurrency serializes here.
        self._mem0_write_lock = asyncio.Lock()

        vector_store_config = {
            "collection_name": collection_name,
            "embedding_model_dims": embed_dims,
        }
        if self._qdrant_server_mode:
            vector_store_config["host"] = self._qdrant_host
            vector_store_config["port"] = self._qdrant_port
            print(
                f"[mem0sdk] qdrant SERVER mode @ {self._qdrant_host}:{self._qdrant_port}"
                f" collection={collection_name}",
                file=sys.stderr,
            )
        else:
            vector_store_config["path"] = self._qdrant_path
            print(
                f"[mem0sdk] qdrant LOCAL mode @ {self._qdrant_path}"
                f" (threading.Lock serialization active)",
                file=sys.stderr,
            )

        mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "api_key": llm_api_key,
                    "openai_base_url": llm_endpoint,
                    "temperature": 0.1,
                    "max_tokens": llm_max_tokens,
                },
            },
            "embedder": {
                "provider": embedder_provider,
                "config": {
                    "model": embedder_model,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": vector_store_config,
            },
            "history_db_path": history_db_path,
        }

        # Qwen3 extractor models default to "thinking" mode, which emits
        # <think>…</think> chain-of-thought tokens before the actual JSON
        # answer. mem0's fact-extraction prompt expects a clean JSON
        # object; thinking-mode output truncates at max_tokens and leaves
        # the parser with `Unterminated string` / `Expecting value` /
        # `'facts' KeyError`. Qwen3's official switch-off is to include
        # the literal string `/no_think` in the system prompt.
        #
        # The served model name is usually an alias (`default` on our
        # SGLang setup), so `llm_model` alone is unreliable for family
        # detection. The driver passes `extractor_family` explicitly
        # (sniffed from the output filename); fall back to regex on the
        # llm_model id otherwise.
        family = (cfg.get("extractor_family") or "").lower()
        model_lc = (llm_model or "").lower()
        is_qwen3 = (
            family == "qwen3"
            or "qwen3" in model_lc
            or "qwen-3" in model_lc
        )
        if is_qwen3 and not vanilla:
            try:
                from mem0.configs.prompts import (
                    FACT_RETRIEVAL_PROMPT,
                    DEFAULT_UPDATE_MEMORY_PROMPT,
                )
                mem0_config["custom_fact_extraction_prompt"] = (
                    FACT_RETRIEVAL_PROMPT.rstrip() + "\n\n/no_think"
                )
                mem0_config["custom_update_memory_prompt"] = (
                    DEFAULT_UPDATE_MEMORY_PROMPT.rstrip() + "\n\n/no_think"
                )
                print(
                    "[mem0sdk] applied /no_think prompt tail to "
                    "Qwen3 fact_extraction + update_memory prompts",
                    file=sys.stderr,
                )
            except ImportError:
                pass

        # Stash the config for lazy initialization in _ensure_mem()
        self._mem0_config = mem0_config
        self.extractor_tag = f"{llm_model}@{llm_endpoint}"

    async def _ensure_mem(self):
        """Lazy-init AsyncMemory on first use. mem0 1.0.4's
        `AsyncMemory.from_config` is itself async, so it can't be called
        from __init__. We guard with an asyncio.Lock and a double-checked
        nil to make concurrent first-callers safe.

        OpenRouter env-key is popped around the from_config call to
        prevent mem0's OpenAI LLM backend from silently hijacking our
        explicit endpoint — see the openai.py code in mem0 1.0.x:

            if os.environ.get("OPENROUTER_API_KEY"):
                self.client = OpenAI(api_key=OPENROUTER_API_KEY,
                                     base_url="https://openrouter.ai/api/v1")
            else:
                ...use explicit config...

        For Config A (local SGLang) the env var must be invisible. For
        Config B (remote OpenRouter extractor) we pass the key via
        cfg["llm_api_key"] instead, so popping is correct for both.
        """
        if self._mem is not None:
            return self._mem
        async with self._init_lock:
            if self._mem is not None:  # double-checked under lock
                return self._mem
            _saved_or_key = os.environ.pop("OPENROUTER_API_KEY", None)
            try:
                self._mem = await self._AsyncMemory_cls.from_config(self._mem0_config)
            finally:
                if _saved_or_key is not None:
                    os.environ["OPENROUTER_API_KEY"] = _saved_or_key
            return self._mem

    @property
    def prompt_style(self) -> PromptStyle:
        return PromptStyle.TEXT_SESSIONS

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        """Batch-ingest messages via mem0's conversation API.

        Mem0's `.add()` accepts either a single string or a list of chat
        messages (`[{"role": ..., "content": ...}, ...]`). The list form
        lets mem0 extract facts from the whole conversation in ONE LLM
        call instead of one call per turn.

        This method groups consecutive same-thread_id messages into one
        mem0 call per session, and issues those calls concurrently via
        `asyncio.gather`. Because the underlying `AsyncMemory` is truly
        async (AsyncOpenAI + AsyncQdrantClient), gather actually lands
        multiple requests at SGLang at the same time — driver
        concurrency becomes real SGLang #running-req concurrency.
        """
        if not messages:
            return {
                "namespace": namespace,
                "indexed_messages": 0,
                "n_errors": 0,
                "first_errors": [],
                "mode": "mem0_sdk",
            }

        mem = await self._ensure_mem()

        # Group consecutive messages by thread_id so each batch is one
        # conversation. Crossing thread_ids in one .add() would confuse
        # mem0's conversational extraction.
        batches: List[List[MessageEntry]] = []
        current: List[MessageEntry] = []
        current_tid: Optional[str] = None
        for m in messages:
            if current_tid is None or m.thread_id == current_tid:
                current.append(m)
                current_tid = m.thread_id
            else:
                batches.append(current)
                current = [m]
                current_tid = m.thread_id
        if current:
            batches.append(current)

        async def _add_one_batch(batch: List[MessageEntry]) -> tuple:
            conv = [
                {
                    "role": "user",
                    "content": f"[{m.meta.get('speaker') or (m.user_id if m.user_id is not None else 'unknown')}] {m.text}",
                }
                for m in batch
            ]
            meta = {
                "thread_id": batch[0].thread_id,
                "occur_ts": batch[0].occur_ts,
                "session_day": batch[0].meta.get("session_day"),
            }
            try:
                if getattr(self, "_qdrant_server_mode", False):
                    await mem.add(conv, user_id=namespace, metadata=meta)
                else:
                    async with self._mem0_write_lock:
                        await mem.add(conv, user_id=namespace, metadata=meta)
                return (len(batch), None)
            except Exception as e:
                return (
                    -len(batch),
                    f"{type(e).__name__}: {str(e)[:200]} "
                    f"(batch tid={batch[0].thread_id} n={len(batch)})",
                )

        # In qdrant SERVER mode the vector store is thread-safe, so we
        # can fire session-batches concurrently via asyncio.gather — big
        # throughput win for the LLM extractor. In LOCAL mode we keep
        # serial per-namespace to dodge the LocalCollection
        # read-after-write race described in _install_qdrant_local_lock.
        if getattr(self, "_qdrant_server_mode", False):
            results = await asyncio.gather(*(_add_one_batch(_b) for _b in batches))
        else:
            results = []
            for _b in batches:
                results.append(await _add_one_batch(_b))

        added = 0
        n_errors = 0
        first_errors: List[str] = []
        for n_or_neg, err in results:
            if n_or_neg >= 0:
                added += n_or_neg
            else:
                n_errors += -n_or_neg
                if err is not None and len(first_errors) < 3:
                    first_errors.append(err)

        if n_errors > 0:
            print(
                f"[mem0sdk.add] ns={namespace}: {added} ok, {n_errors} errors "
                f"across {len(batches)} batches; first: {first_errors}",
                file=sys.stderr,
            )

        return {
            "namespace": namespace,
            "indexed_messages": added,
            "n_batches": len(batches),
            "n_errors": n_errors,
            "first_errors": first_errors,
            "mode": "mem0_sdk",
        }

    async def reset(self, *, namespace: str) -> None:
        """Drop all memories for a single namespace. Used by sanity gates and
        per-trial isolation. AsyncMemory.delete_all is awaitable."""
        try:
            mem = await self._ensure_mem()
            if getattr(self, "_qdrant_server_mode", False):
                await mem.delete_all(user_id=namespace)
            else:
                async with self._mem0_write_lock:
                    await mem.delete_all(user_id=namespace)
        except Exception as e:
            print(
                f"[mem0sdk.reset] ns={namespace} failed: "
                f"{type(e).__name__}: {str(e)[:200]}",
                file=sys.stderr,
            )

    async def close(self) -> None:
        """Clean up the auto-created tempdir + async clients, if any.
        Safe to call multiple times."""
        # Server mode: drop the collection so subsequent runs are hermetic.
        if getattr(self, "_qdrant_server_mode", False):
            try:
                import urllib.request
                cn = self._mem0_config.get("vector_store", {}).get("config", {}).get("collection_name")
                if cn:
                    req = urllib.request.Request(
                        f"http://{self._qdrant_host}:{self._qdrant_port}/collections/{cn}",
                        method="DELETE",
                    )
                    urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass
        # Drop Qdrant scratch dir if we own it
        if getattr(self, "_owns_qdrant_path", False):
            p = getattr(self, "_qdrant_path", None)
            if p:
                shutil.rmtree(p, ignore_errors=True)
                self._owns_qdrant_path = False
        # Drop sqlite history DB if we auto-created it
        if getattr(self, "_owns_history_db", False):
            hp = getattr(self, "_history_db_path", None)
            if hp and os.path.exists(hp):
                try:
                    os.remove(hp)
                except OSError:
                    pass
                self._owns_history_db = False
        # Try to close AsyncMemory's internal async HTTP client if it
        # exposes a close method. Not all mem0 versions do, so best-effort.
        if self._mem is not None:
            try:
                close_fn = getattr(self._mem, "aclose", None) or getattr(self._mem, "close", None)
                if close_fn is not None:
                    result = close_fn()
                    if hasattr(result, "__await__"):
                        await result
            except Exception:
                pass

    async def search(self, *, namespace: str, query: str, top_k: int, **_: Any) -> List[SearchHit]:
        # Scope kwargs (ego_id, query_timestamp, ...) are ignored: the store
        # is namespaced per ego at ingest and holds what was ingested.
        try:
            mem = await self._ensure_mem()
            if getattr(self, "_qdrant_server_mode", False):
                results = await mem.search(query, user_id=namespace, limit=top_k)
            else:
                async with self._mem0_write_lock:
                    results = await mem.search(query, user_id=namespace, limit=top_k)
        except Exception as e:
            print(
                f"[mem0sdk.search] ns={namespace} query={query[:40]!r} failed: "
                f"{type(e).__name__}: {str(e)[:200]}",
                file=sys.stderr,
            )
            return []

        hits = []
        result_list = results.get("results", results) if isinstance(results, dict) else results
        for r in result_list:
            mem = r if isinstance(r, dict) else getattr(r, "__dict__", {})
            hits.append(SearchHit(
                msg_id=str(mem.get("id", "")),
                score=float(mem.get("score", 1.0)),
                text=str(mem.get("memory", mem.get("text", ""))),
                occur_ts=str(mem.get("metadata", {}).get("occur_ts", "") if mem.get("metadata") else ""),
                thread_id=str(mem.get("metadata", {}).get("thread_id", "") if mem.get("metadata") else ""),
                user_id=None,
            ))
        return hits
