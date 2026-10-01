#!/usr/bin/env python3
"""Build a frozen memory-system retrieval cache for one (system × extractor) trial.

This is the ingest driver for the Mem0 / Memobase / MemOS memory backends. It
performs day-by-day monotonic ingest of an L-scale corpus and queries the
memory system at the end of each simulated day for the eval instances whose
query_timestamp falls on that day. The retrieved memories are written to a
JSONL cache that the answering pipeline (via MemoryCacheAdapter) replays
without re-running the expensive extraction phase.

Why this exists:
  - Memory systems do an LLM-based extraction on ingest. Running ingest once
    per (answerer model) is wasteful — the retrieved memories don't depend on
    which LLM ultimately answers. Running ingest once per (system, extractor)
    and replaying via cache turns N answerer trials into ~free.
  - Day-batched isolation is a paper-level invariant: a query at timestamp
    t=6.67 must only see sessions through day floor(t)=6. Doing ingest +
    query in chronological order makes this trivially correct without any
    snapshot/restore plumbing inside the memory systems.

Usage on H200:
    # Config A paired — extractor matches the answerer for that tier
    python -m scripts.reproduce.build_memory_cache \
        --system mem0 \
        --run-dir MASim/runs/l_20260408_111046 \
        --extractor-model Qwen3-8B-AWQ \
        --extractor-endpoint http://127.0.0.1:8100/v1 \
        --extractor-api-key EMPTY \
        --extractor-config A_paired \
        --output memory_cache/mem0/A_paired_qwen3_8b.jsonl

    # Config B remote — Claude 3.5 Haiku via OpenRouter, one cache per system
    python -m scripts.reproduce.build_memory_cache \
        --system mem0 \
        --run-dir MASim/runs/l_20260408_111046 \
        --extractor-model anthropic/claude-3.5-haiku \
        --extractor-endpoint https://openrouter.ai/api/v1 \
        --extractor-api-key "$OPENROUTER_API_KEY" \
        --extractor-config B_remote \
        --output memory_cache/mem0/B_remote_claude35haiku.jsonl

Inputs (must exist under --run-dir):
    corpus_sessions.jsonl     — one session per line, fields described below
    ego_session_map.json      — {ego_agent_id: [session_id, ...]}
    eval_instances/d*.jsonl   — one query per line, with instance_id, query,
                                query_timestamp, ego_agent_id, asker_agent_id

Output:
    JSONL cache, one row per eval instance, schema documented in
    eval/src/adapters/memory_cache.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo root importable when invoked as `python scripts/build_memory_cache.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.src.types import MessageEntry  # noqa: E402
from memarena.text_cleaning import clean_llm_text  # noqa: E402


# ─── tiny .env loader so the script works without python-dotenv ──────────────

def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


load_env()


def clean_mem0_scratch(paths: Optional[List[Path]] = None) -> None:
    """Wipe the **specific** scratch paths passed in.

    Concurrent mem0 runs used to share state via ``~/.mem0/history.db``
    and glob-matched ``/tmp/mem0_qdrant_*`` directories. A later run's
    startup wiped paths still in use by earlier siblings, producing
    sqlite ``readonly database`` and numpy ``IndexError`` cascades.

    This version only wipes the *exact* paths passed by the caller.
    Callers are responsible for passing per-run-unique paths.
    """
    if not paths:
        return
    for p in paths:
        if p and p.exists():
            shutil.rmtree(p, ignore_errors=True)


# ─── corpus loaders ──────────────────────────────────────────────────────────

def load_corpus_sessions(run_dir: Path) -> Dict[str, dict]:
    """Index corpus_sessions.jsonl by session_id."""
    from eval.src.masim_loader import open_corpus_sessions
    out: Dict[str, dict] = {}
    with open_corpus_sessions(run_dir) as fh:
        lines = fh.read().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        sess = json.loads(line)
        sid = sess.get("session_id") or sess.get("id")
        if sid:
            out[sid] = sess
    return out


def load_ego_session_map(run_dir: Path) -> Dict[str, List[str]]:
    """Load {ego_agent_id: [session_id, ...]}."""
    path = run_dir / "ego_session_map.json"
    if not path.exists():
        raise FileNotFoundError(f"missing ego_session_map.json at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_instances(run_dir: Path) -> List[dict]:
    """Concatenate all eval_instances/d*.jsonl files."""
    inst_dir = run_dir / "eval_instances"
    if not inst_dir.exists():
        raise FileNotFoundError(f"missing eval_instances/ at {inst_dir}")
    out: List[dict] = []
    for f in sorted(inst_dir.glob("d*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                inst = json.loads(line)
                if isinstance(inst, dict) and "query" in inst:
                    inst["query"] = clean_llm_text(inst.get("query", ""))
                out.append(inst)
            except json.JSONDecodeError:
                pass
    return out


# ─── day extraction ──────────────────────────────────────────────────────────

def session_day(sess: dict) -> int:
    """Get the simulated day index for a session.

    Field conventions (tried in order):
      1. top-level "day" (int)
      2. top-level "start_time" (float fractional day) — MemArena corpus
      3. top-level "timestamp" (float)
      4. min(turn.timestamp) across the session's turns
    Returns floor() as int day index.
    """
    if "day" in sess:
        return int(sess["day"])
    if "start_time" in sess:
        return int(float(sess["start_time"]))
    if "timestamp" in sess:
        return int(float(sess["timestamp"]))
    turns = sess.get("turns") or []
    if turns:
        ts_values = [float(t.get("timestamp", 0.0)) for t in turns if t.get("timestamp") is not None]
        if ts_values:
            return int(min(ts_values))
    return 0


def session_to_messages(sess: dict, sid: str) -> List[MessageEntry]:
    """Flatten a session into MessageEntry list (one per turn)."""
    out: List[MessageEntry] = []
    turns = sess.get("turns") or []
    for i, t in enumerate(turns):
        speaker = (
            t.get("speaker_id")
            or t.get("speaker")
            or t.get("user_id")
            or ""
        )
        out.append(
            MessageEntry(
                msg_id=str(t.get("msg_id") or t.get("turn_id") or f"{sid}_t{i}"),
                occur_ts=str(t.get("timestamp", "")),
                deliver_ts=str(t.get("timestamp", "")),
                user_id=None,  # MemArena agents are string IDs, MessageEntry expects int
                thread_id=sid,
                text=str(t.get("text") or t.get("content") or ""),
                meta={
                    "speaker": speaker,
                    "session_id": sid,
                    "modality": sess.get("modality", ""),
                    "session_day": session_day(sess),
                },
            )
        )
    return out


def instance_day(inst: dict) -> int:
    meta = inst.get("metadata") or {}
    ts = meta.get("query_timestamp")
    if ts is None:
        ts = inst.get("query_timestamp")
    if ts is None:
        return 0
    return int(float(ts))


def instance_ego(inst: dict) -> str:
    """Resolve the ego agent whose memory should answer this instance.

    New MASim outputs usually carry top-level ``ego_agent_id``. Some older or
    hardened dimensions only carry the same identity in metadata or in the
    answerer/target fields, so use the same broad fallback policy as the eval
    loader instead of silently dropping those rows from the frozen cache.
    """
    meta = inst.get("metadata") or {}
    gt = inst.get("ground_truth") or {}
    candidates = (
        inst.get("ego_agent_id"),
        meta.get("ego_agent_id"),
        meta.get("query_agent"),
        meta.get("instance_query_agent"),
        inst.get("answerer_agent_id"),
        meta.get("answerer_agent_id"),
        gt.get("query_agent"),
        gt.get("target_agent"),
        inst.get("target_agent"),
        meta.get("target_agent"),
    )
    for value in candidates:
        s = str(value or "").strip()
        if s and s not in {"__assistant__", "assistant"}:
            return s
    return ""


# ─── adapter construction ────────────────────────────────────────────────────

def build_adapter_for_extraction(
    *,
    system: str,
    extractor_model: str,
    extractor_endpoint: str,
    extractor_api_key: str,
    qdrant_path: str,
    llm_max_tokens: int = 4096,
    history_db_path: Optional[str] = None,
    run_tag: Optional[str] = None,
    extractor_family: Optional[str] = None,
    vanilla: bool = False,
) -> Any:
    """Build the right adapter with extractor LLM cfg overrides."""
    from eval.src.adapters import build_adapter

    if system == "mem0":
        # Per-run-unique collection name so concurrent mem0 runs against
        # a shared Qdrant on-disk directory don't clobber each other's
        # vectors. run_tag falls back to a short time-based token.
        # In vanilla mode we use a fixed collection name to mirror the
        # Phase 5.2 control script (no per-run suffix plumbing).
        coll_suffix = run_tag or f"{int(time.time())}_{os.getpid()}"
        collection_name = (
            "memarena_mem0_vanilla"
            if vanilla
            else f"memarena_{system}_{coll_suffix}"
        )
        cfg = {
            "mem0sdk": {
                "llm_endpoint": extractor_endpoint,
                "llm_model": extractor_model,
                "llm_api_key": extractor_api_key,
                "llm_max_tokens": llm_max_tokens,
                "qdrant_path": qdrant_path,
                "collection_name": collection_name,
                "history_db_path": history_db_path,
                "extractor_family": extractor_family,
                "vanilla": vanilla,
            }
        }
        return build_adapter("mem0sdk", cfg=cfg, output_dir=Path("."))
    if system == "memobase":
        cfg = {"memobase": {}}
        return build_adapter("memobase", cfg=cfg, output_dir=Path("."))
    if system == "memos":
        cfg = {"memos": {}}
        return build_adapter("memos", cfg=cfg, output_dir=Path("."))
    if system == "memsearch":
        # Per-cell scratch root keyed off run_tag; falls back to PID so
        # parallel cells never collide. Markdown files + milvus.db both
        # live under this root.
        suffix = run_tag or f"{int(time.time())}_{os.getpid()}"
        root_dir = (
            os.getenv("MEMSEARCH_ROOT_DIR_BASE", "/tmp/memarena_memsearch")
            + f"/{suffix}"
        )
        cfg = {
            "memsearch": {
                "root_dir": root_dir,
                # Embedder defaults to ollama via OLLAMA_HOST. The wrapper
                # script (run_l_all_models.sh / setup_memory_backends.sh)
                # already starts a per-cell ollama and exports OLLAMA_HOST
                # before launching this process.
            }
        }
        return build_adapter("memsearch", cfg=cfg, output_dir=Path("."))
    raise ValueError(f"unknown system: {system}")


# ─── main ingest loop ────────────────────────────────────────────────────────

async def build_cache(args) -> None:
    run_dir = Path(args.run_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] corpus_sessions from {run_dir}", file=sys.stderr)
    corpus = load_corpus_sessions(run_dir)
    print(f"[load]   {len(corpus)} sessions", file=sys.stderr)

    print(f"[load] ego_session_map", file=sys.stderr)
    ego_map = load_ego_session_map(run_dir)
    print(f"[load]   {len(ego_map)} ego agents", file=sys.stderr)

    print(f"[load] eval_instances/", file=sys.stderr)
    instances = load_all_instances(run_dir)
    print(f"[load]   {len(instances)} instances", file=sys.stderr)
    if args.qa_limit and args.qa_limit > 0:
        instances = instances[: args.qa_limit]
        print(f"[limit] using first {len(instances)} eval instances", file=sys.stderr)

    # Group instances by (ego_agent, day). Keep a side list for instances that
    # cannot be mapped into a memory namespace; replay still needs one cache row
    # per QA instance, otherwise strict memory_cache lookup fails late.
    instances_by_ego_day: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    unscheduled_instances: List[Tuple[dict, str, str]] = []
    for inst in instances:
        ego = instance_ego(inst)
        if not ego:
            unscheduled_instances.append((inst, "", "missing ego/query-agent identity"))
            continue
        if ego not in ego_map:
            unscheduled_instances.append((inst, ego, f"ego not present in ego_session_map: {ego}"))
            continue
        day = instance_day(inst)
        instances_by_ego_day[(ego, day)].append(inst)

    # Optional ego-agent limit for smoke runs
    ego_agents = sorted(ego_map.keys())
    if args.user_limit and args.user_limit > 0:
        ego_agents = ego_agents[: args.user_limit]
        print(f"[limit] using first {len(ego_agents)} ego agents", file=sys.stderr)

    # Hermetic start — per-run unique paths only, never glob siblings.
    # Build a short run tag from output filename + pid so two concurrent
    # runs on the same host never share a scratch path.
    run_tag = f"{Path(args.output).stem}_{os.getpid()}"
    history_db_path = str(Path("/tmp") / f"mem0_history_{run_tag}.db")
    own_qdrant = Path(args.qdrant_path)
    if getattr(args, "vanilla", False):
        # Phase 7.0 vanilla: do not pre-wipe scratch paths. Caller is
        # responsible for starting with a clean /tmp/mem0_qdrant_* dir
        # (Phase 5.2 control script's convention).
        print(
            f"[init] vanilla mode — skipping clean_mem0_scratch "
            f"(qdrant={own_qdrant}, history_db={history_db_path})",
            file=sys.stderr,
        )
    else:
        clean_mem0_scratch([own_qdrant, Path(history_db_path)])
        print(
            f"[init] cleaned mem0 scratch (run_tag={run_tag}, "
            f"qdrant={own_qdrant}, history_db={history_db_path})",
            file=sys.stderr,
        )

    # Extractor family detection — the `--extractor-model` flag usually
    # resolves to `default` (SGLang's served-model-name alias) and so
    # can't be used to detect the real model family. Sniff the output
    # filename stem instead — our convention is
    # `A_paired_<family>_<variant>.jsonl` (e.g. A_paired_qwen3_0_6b,
    # A_paired_llama3_2_3b, A_paired_qwen3_8b_awq).
    out_stem_lc = Path(args.output).stem.lower()
    extractor_family = None
    for fam in ("qwen3", "llama", "mistral", "claude", "haiku"):
        if fam in out_stem_lc:
            extractor_family = fam
            break
    # Also check the explicit --extractor-model for OpenRouter-style ids
    em_lc = (args.extractor_model or "").lower()
    if extractor_family is None:
        for fam in ("qwen3", "llama", "mistral", "claude", "haiku"):
            if fam in em_lc:
                extractor_family = fam
                break
    if extractor_family:
        print(f"[init] extractor_family={extractor_family}", file=sys.stderr)

    print(
        f"[init] adapter system={args.system} extractor={args.extractor_model} "
        f"namespace_mode=ego_agent_id",
        file=sys.stderr,
    )
    adapter = build_adapter_for_extraction(
        system=args.system,
        extractor_model=args.extractor_model,
        extractor_endpoint=args.extractor_endpoint,
        extractor_api_key=args.extractor_api_key,
        qdrant_path=args.qdrant_path,
        llm_max_tokens=int(args.max_tokens),
        history_db_path=history_db_path,
        run_tag=run_tag,
        extractor_family=extractor_family,
        vanilla=bool(getattr(args, "vanilla", False)),
    )

    # Pre-compute: for each ego, a dict day → [(sid, sess), ...]
    # AND the full ordered list of days we need to walk globally.
    ego_sessions_by_day: Dict[str, Dict[int, List[Tuple[str, dict]]]] = {}
    global_days: set = set()
    for ego in ego_agents:
        session_ids = ego_map.get(ego, [])
        per_day: Dict[int, List[Tuple[str, dict]]] = defaultdict(list)
        for sid in session_ids:
            sess = corpus.get(sid)
            if sess is None:
                continue
            d = session_day(sess)
            per_day[d].append((sid, sess))
        ego_sessions_by_day[ego] = per_day
        global_days.update(per_day.keys())
    # Also include days that have queries but no new sessions
    for (ego, d) in instances_by_ego_day:
        if ego in ego_sessions_by_day:
            global_days.add(d)
    day_order = sorted(global_days)

    total_queries = sum(
        len(instances_by_ego_day.get((ego, d), []))
        for ego in ego_agents for d in day_order
    )
    print(
        f"[plan] {len(ego_agents)} agents × {len(day_order)} days, "
        f"{total_queries} scheduled queries, "
        f"{len(unscheduled_instances)} unscheduled, concurrency={args.concurrency}",
        file=sys.stderr,
    )

    # ── Reset all namespaces up-front (one per agent) ───────────────────
    def _cache_error_row(inst: dict, ego: str, reason: str) -> Dict[str, Any]:
        meta = inst.get("metadata") or {}
        qts = meta.get("query_timestamp")
        if qts is None:
            qts = inst.get("query_timestamp")
        asker = inst.get("asker_agent_id") or meta.get("asker_agent_id") or ""
        return {
            "instance_id": inst.get("instance_id") or "",
            "namespace": ego,
            "asker_id": asker,
            "query": inst.get("query") or "",
            "query_timestamp": qts,
            "ingest_day_cutoff": instance_day(inst),
            "system": args.system,
            "config": args.extractor_config,
            "extractor_model": args.extractor_model,
            "extractor_endpoint": _endpoint_label(args.extractor_endpoint),
            "memories": None,
            "memory_count": 0,
            "retrieve_latency_ms": None,
            "error": f"MEMORY_ERROR: {reason}",
        }

    async def _reset_ego(ego: str) -> None:
        try:
            await adapter.reset(namespace=ego)
        except (TypeError, AttributeError):
            pass

    sem = asyncio.Semaphore(max(1, int(args.concurrency)))

    async def _gated(task):
        async with sem:
            return await task

    await asyncio.gather(*(_gated(_reset_ego(e)) for e in ego_agents))

    # ── Output JSONL streaming ──────────────────────────────────────────
    n_rows_written = 0
    n_errored = 0
    n_ingest_batches = 0
    # What actually landed, not only what was submitted. Adapters report
    # indexed_messages / n_errors / flush_error from add(); the Memobase
    # adapter counts a body errno (HTTP 200 + {"errno": 500}) as an error.
    ingest = {
        "sessions_submitted": 0,
        "messages_submitted": 0,
        "messages_succeeded": 0,
        "messages_failed": 0,
        "flush_errors": 0,
        "add_exceptions": 0,
        "ego_days_unreported": 0,  # add() gave no indexed_messages
    }
    t_start = time.time()

    # Lock protects the fout handle since we have concurrent writers.
    write_lock = asyncio.Lock()
    fout = out_path.open("w", encoding="utf-8")
    # Run summary is written in the finally block below so a crashed or
    # SIGTERM'd run still leaves behind a summary.json for triage.
    summary_path = out_path.with_suffix(".summary.json")
    run_status = "incomplete"
    error_detail: Optional[str] = None
    _reraise: Optional[BaseException] = None

    async def _ingest_ego_day(ego: str, day: int) -> int:
        """Ingest all of `ego`'s sessions whose day == `day`.

        Flattens every session's MessageEntry list into one combined
        list and hands it to the adapter in a SINGLE call. The adapter
        groups by thread_id and fires all session-batches concurrently
        via `asyncio.gather` backed by AsyncMemory, so the whole day's
        ingest for one agent actually parallelizes at the transport
        layer (real SGLang #running-req > 1).
        """
        sessions = ego_sessions_by_day.get(ego, {}).get(day, [])
        if not sessions:
            return 0
        all_msgs: List[MessageEntry] = []
        for sid, sess in sessions:
            all_msgs.extend(session_to_messages(sess, sid))
        if not all_msgs:
            return 0
        ingest["sessions_submitted"] += len(sessions)
        ingest["messages_submitted"] += len(all_msgs)
        try:
            result = await adapter.add(namespace=ego, messages=all_msgs)
        except Exception as e:
            import traceback as _tb
            ingest["add_exceptions"] += 1
            ingest["messages_failed"] += len(all_msgs)
            print(
                f"[ingest_err] ego={ego} day={day}: "
                f"{type(e).__name__}: {str(e)[:200]}\n"
                f"{_tb.format_exc()}",
                file=sys.stderr,
            )
            return 0
        result = result if isinstance(result, dict) else {}
        n_err = int(result.get("n_errors") or 0)
        if result.get("indexed_messages") is None:
            ingest["ego_days_unreported"] += 1
        else:
            ingest["messages_succeeded"] += int(result["indexed_messages"])
        ingest["messages_failed"] += n_err
        if result.get("flush_error"):
            ingest["flush_errors"] += 1
        if n_err or result.get("flush_error"):
            print(
                f"[ingest_err] ego={ego} day={day}: {n_err}/{len(all_msgs)} messages failed, "
                f"flush_error={result.get('flush_error')!r} first={result.get('first_errors')}",
                file=sys.stderr,
            )
        return int(result.get("n_batches", len(sessions)))

    async def _query_one(ego: str, day: int, inst: dict) -> Dict[str, Any]:
        iid = inst.get("instance_id") or ""
        query = inst.get("query") or ""
        asker = inst.get("asker_agent_id") or (inst.get("metadata") or {}).get("asker_agent_id") or ""
        qts = (inst.get("metadata") or {}).get("query_timestamp")
        row: Dict[str, Any] = {
            "instance_id": iid,
            "namespace": ego,
            "asker_id": asker,
            "query": query,
            "query_timestamp": qts,
            "ingest_day_cutoff": day,
            "system": args.system,
            "config": args.extractor_config,
            "extractor_model": args.extractor_model,
            "extractor_endpoint": _endpoint_label(args.extractor_endpoint),
            "memories": None,
            "memory_count": 0,
            "retrieve_latency_ms": None,
            "error": None,
        }
        try:
            t_q = time.time()
            hits = await adapter.search(namespace=ego, query=query, top_k=int(args.top_k))
            latency_ms = int((time.time() - t_q) * 1000)
            row["memories"] = [
                {
                    "id": h.msg_id,
                    "text": h.text,
                    "score": float(h.score),
                    "occur_ts": h.occur_ts,
                    "thread_id": h.thread_id,
                }
                for h in (hits or [])
            ]
            row["memory_count"] = len(row["memories"])
            row["retrieve_latency_ms"] = latency_ms
        except Exception as e:
            row["error"] = f"MEMORY_ERROR: {type(e).__name__}: {str(e)[:300]}"
            if args.verbose:
                traceback.print_exc(file=sys.stderr)
        return row

    async def _ingest_ego_day_gated(ego: str, day: int) -> int:
        async with sem:
            return await _ingest_ego_day(ego, day)

    async def _query_ego_day_gated(ego: str, day: int) -> List[Dict[str, Any]]:
        """Run all queries for (ego, day) serially inside one gate slot."""
        queries = instances_by_ego_day.get((ego, day), [])
        rows: List[Dict[str, Any]] = []
        async with sem:
            for inst in queries:
                row = await _query_one(ego, day, inst)
                rows.append(row)
        return rows

    _hw_marker_path = getattr(args, "hw_marker_file", None)

    def _emit_cache_marker(instance_id: str, phase: str, ts_start_ms: int, ts_end_ms: int) -> None:
        if not _hw_marker_path:
            return
        import threading
        row = json.dumps({
            "ts_start_ms": ts_start_ms, "ts_end_ms": ts_end_ms,
            "instance_id": instance_id, "phase": phase,
            "llm_model": args.extractor_model,
            "prompt_tokens": None, "completion_tokens": None, "call_idx": 0,
        }, ensure_ascii=False) + "\n"
        with open(_hw_marker_path, "a") as f:
            f.write(row)

    try:
        # ── Main per-day loop with inside-day parallel fan-out over agents ──
        for day_idx, day in enumerate(day_order):
            t_day = time.time()
            _day_ts_start = int(time.time() * 1000)

            # Step A: ingest all agents' sessions for this day in parallel
            ingest_tasks = [_ingest_ego_day_gated(e, day) for e in ego_agents]
            batch_counts = await asyncio.gather(*ingest_tasks)
            n_ingest_batches += sum(batch_counts)

            _ingest_ts_end = int(time.time() * 1000)
            _emit_cache_marker(f"day_{day_idx}_ingest", "ingest", _day_ts_start, _ingest_ts_end)

            # Step B: barrier. memory state is now "sessions ≤ day" for all agents
            # Run all agents' queries for this day in parallel
            query_tasks = [_query_ego_day_gated(e, day) for e in ego_agents]
            rows_per_agent = await asyncio.gather(*query_tasks)

            _query_ts_end = int(time.time() * 1000)
            _emit_cache_marker(f"day_{day_idx}_query", "query", _ingest_ts_end, _query_ts_end)

            # Step C: serialize rows to JSONL under lock
            async with write_lock:
                for rows in rows_per_agent:
                    for row in rows:
                        if row.get("error"):
                            n_errored += 1
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_rows_written += 1
                fout.flush()

            elapsed_day = time.time() - t_day
            day_queries = sum(len(r) for r in rows_per_agent)
            day_batches = sum(batch_counts)
            print(
                f"[day {day_idx+1}/{len(day_order)}] day={day} "
                f"ingest_batches={day_batches} queries={day_queries} "
                f"elapsed={elapsed_day:.1f}s total_rows={n_rows_written} errored={n_errored} "
                f"msgs_ok={ingest['messages_succeeded']}/{ingest['messages_submitted']} "
                f"msgs_failed={ingest['messages_failed']} flush_errors={ingest['flush_errors']}",
                file=sys.stderr,
            )

        if unscheduled_instances:
            async with write_lock:
                for inst, ego, reason in unscheduled_instances:
                    fout.write(json.dumps(_cache_error_row(inst, ego, reason), ensure_ascii=False) + "\n")
                    n_rows_written += 1
                    n_errored += 1
                fout.flush()
            print(
                f"[unscheduled] wrote {len(unscheduled_instances)} MEMORY_ERROR rows "
                f"so cache rows match eval instances",
                file=sys.stderr,
            )

        run_status = "complete"
    except BaseException as e:
        run_status = "crashed"
        error_detail = f"{type(e).__name__}: {str(e)[:500]}"
        print(f"\n[crash] {error_detail}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Re-raise after finally runs so the process exits non-zero,
        # but only if it's an unexpected exception. Keyboard interrupt /
        # SIGTERM should still let us write the summary first.
        if not isinstance(e, (KeyboardInterrupt, SystemExit)):
            # Write summary in finally, then re-raise.
            _reraise = e
        else:
            _reraise = None
    finally:
        try:
            fout.close()
        except Exception:
            pass
        # Close the adapter (async HTTP sessions, sqlite handles, etc.)
        # before we exit. Suppresses the aiohttp "Unclosed client session"
        # warning observed in Memobase Phase 2M stderr.
        try:
            close_fn = getattr(adapter, "close", None)
            if close_fn is not None:
                maybe_coro = close_fn()
                if hasattr(maybe_coro, "__await__"):
                    await maybe_coro
        except Exception:
            pass
        # Elapsed is computed INSIDE finally so a NameError / missing
        # variable can never mask the summary write.
        _elapsed_s = int(time.time() - t_start)
        api_stats = None
        try:
            stats_fn = getattr(adapter, "api_stats", None)
            api_stats = stats_fn() if callable(stats_fn) else None
        except Exception:
            api_stats = None
        try:
            print(
                f"\n[{run_status}] {n_rows_written} rows written "
                f"({n_errored} errored) in {_elapsed_s}s → {out_path}",
                file=sys.stderr,
            )
            print(
                f"[{run_status}] ingest: {ingest['messages_succeeded']}/{ingest['messages_submitted']} "
                f"messages succeeded ({ingest['sessions_submitted']} sessions submitted), "
                f"{ingest['messages_failed']} failed, {ingest['flush_errors']} flush errors, "
                f"{ingest['add_exceptions']} add() exceptions",
                file=sys.stderr,
            )
            if ingest["messages_failed"] or ingest["flush_errors"] or ingest["add_exceptions"]:
                print(
                    f"[{run_status}][WARNING] memory ingest lost data; the cache does not hold "
                    f"every submitted session (api_stats={api_stats})",
                    file=sys.stderr,
                )
        except Exception:
            pass
        # Always write a sidecar summary — even on crashed / partial runs
        # so triage can see what happened without having to replay stderr.
        try:
            summary_path.write_text(
                json.dumps(
                    {
                        "status": run_status,
                        "error": error_detail,
                        "system": args.system,
                        "config": args.extractor_config,
                        "extractor_model": args.extractor_model,
                        "extractor_endpoint_label": _endpoint_label(args.extractor_endpoint),
                        "run_dir": str(run_dir),
                        "n_ego_agents": len(ego_agents),
                        "n_days": len(day_order),
                        "n_rows_written": n_rows_written,
                        "n_errored": n_errored,
                        "n_ingest_batches": n_ingest_batches,
                        # submitted vs succeeded (n_ingest_batches counts submissions)
                        "ingest": ingest,
                        "adapter_api_stats": api_stats,
                        "concurrency": int(args.concurrency),
                        "llm_max_tokens": int(args.max_tokens),
                        "elapsed_seconds": _elapsed_s,
                        "top_k": int(args.top_k),
                        "trial_seed": args.trial_seed,
                        "namespace_mode": "ego_agent_id",
                        "output_jsonl": str(out_path),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"[{run_status}] summary → {summary_path}", file=sys.stderr)
        except Exception as _sum_e:
            print(f"[{run_status}] summary write failed: {_sum_e}", file=sys.stderr)

    # If we saved an exception to re-raise above, do it now (after the
    # finally block wrote the summary).
    if run_status == "crashed" and _reraise is not None:
        raise _reraise  # type: ignore[misc]


def _endpoint_label(endpoint: str) -> str:
    """Return a coarse label for paper / cache metadata, hiding raw URLs."""
    e = (endpoint or "").lower()
    if "openrouter.ai" in e:
        return "openrouter"
    if "127.0.0.1" in e or "localhost" in e:
        return "local_sglang"
    if "anthropic.com" in e:
        return "anthropic_api"
    if "openai.com" in e:
        return "openai_api"
    return "external"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--system", required=True, choices=["mem0", "memobase", "memos", "memsearch"])
    p.add_argument("--run-dir", required=True, help="MASim run directory containing corpus_sessions.jsonl + eval_instances/")
    p.add_argument("--extractor-model", required=True,
                   help='LLM id (e.g. "Qwen3-8B-AWQ" or "anthropic/claude-3.5-haiku")')
    p.add_argument("--extractor-endpoint", required=True,
                   help="OpenAI-compatible base URL")
    p.add_argument("--extractor-api-key", default=os.environ.get("OPENROUTER_API_KEY", "EMPTY"),
                   help="bearer token (defaults to $OPENROUTER_API_KEY then EMPTY)")
    p.add_argument("--extractor-config", required=True, choices=["A_paired", "B_remote"],
                   help="cache metadata tag")
    p.add_argument("--output", required=True, help="output JSONL path")
    p.add_argument("--qdrant-path", default="/tmp/mem0_qdrant_memarena",
                   help="local Qdrant on-disk path (per-run unique recommended)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--qa-limit", type=int, default=0,
                   help="for smoke/subset runs: only cache the first N eval instances (0 = all)")
    p.add_argument("--user-limit", type=int, default=0,
                   help="for smoke runs: only ingest first N ego agents (0 = all)")
    p.add_argument("--namespace-prefix", type=str, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--concurrency", type=int, default=8,
                   help="parallel agents per day during ingest + query (default 8, "
                        "matches the setting used for the paper). Higher = faster but "
                        "pushes more load onto SGLang and, with weak extractors that "
                        "emit invalid JSON, can collapse server-side throughput via "
                        "synchronous retry storms.")
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="max_tokens the extractor LLM may emit per call (default 4096). "
                        "Raise for weak/small extractors that truncate JSON output.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trial-seed", type=int, default=None,
                   help="Integer seed for trial provenance (e.g. 1002/1003/1004). "
                        "Recorded in output metadata.")
    p.add_argument("--hw-marker-file", type=str, default=None,
                   help="Path to append JSONL hardware telemetry markers per LLM call "
                        "(spark_remote.md §8b).")
    p.add_argument("--vanilla",
                   action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Phase 7.0 (post-block): run mem0 with all cosmetic "
                        "driver patches disabled — per-run qdrant scratch "
                        "isolation, Qwen3 /no_think prompt override, qdrant "
                        "SERVER-mode auto-detection, and the OpenRouter "
                        "provider pin. The qdrant LocalCollection "
                        "threading.Lock is KEPT because 7.0.2's smoke proved "
                        "it is a load-bearing bug fix for a real mem0 1.0.4 "
                        "async-gather race, not a cosmetic patch. Only "
                        "meaningful with --system mem0.")
    args = p.parse_args()

    # Phase 7.0: set the vanilla env var BEFORE the lazy adapter import
    # inside build_cache() so the module-level monkey-patch install calls
    # in mem0_sdk_adapter.py observe it and skip themselves.
    if args.vanilla:
        os.environ["MEMARENA_MEM0_VANILLA"] = "1"
        os.environ["PHASE5_DISABLE_OPENROUTER_PIN"] = "1"
        print("[vanilla] cosmetic mem0 patches disabled "
              "(qdrant LocalCollection threading.Lock KEPT — bug fix, not monkey-patch)",
              file=sys.stderr)

    asyncio.run(build_cache(args))


if __name__ == "__main__":
    main()
