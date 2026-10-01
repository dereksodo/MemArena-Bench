from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote as _url_quote

from .base import normalize_score
from .http_base import HttpRetrievalAdapter
from ..types import MessageEntry, SearchHit


_MEMOBASE_HEALTH_PATH = "/api/v1/healthcheck"

# Memobase answers most failures with HTTP 200 and the error in the body:
# {"data": null, "errno": 500, "errmsg": "..."}. errno 500 is what v0.0.42
# returns when Postgres runs out of connections ("too many clients"), so it
# and the other 5xx-style codes are retried. 501 means "feature disabled"
# (e.g. event embedding off) and is not retried.
_TRANSIENT_ERRNOS = {500, 502, 503, 504}
_TRANSIENT_ERRMSG = ("too many clients", "connection", "timeout", "timed out", "pool", "temporarily")


class MemobaseAPIError(RuntimeError):
    """A Memobase call whose body reported an error (non-zero errno), after
    the transient-error retries were used up."""

    def __init__(self, op: str, errno: object, errmsg: str) -> None:
        super().__init__(f"memobase {op}: errno={errno} errmsg={errmsg[:200]}")
        self.op = op
        self.errno = errno
        self.errmsg = errmsg


def _body_error(data: Any) -> Optional[Tuple[object, str]]:
    """(errno, errmsg) if a Memobase response body reports an error, else None."""
    if not isinstance(data, dict):
        return None
    errno = data.get("errno")
    if errno not in (0, None, "0"):
        return errno, str(data.get("errmsg") or data.get("message") or "")
    err = data.get("error")
    if err:
        return "error", str(err)
    return None


def _is_transient(errno: object, errmsg: str) -> bool:
    try:
        if int(errno) in _TRANSIENT_ERRNOS:  # type: ignore[arg-type]
            return True
    except (TypeError, ValueError):
        pass
    msg = (errmsg or "").lower()
    return any(k in msg for k in _TRANSIENT_ERRMSG)


class MemobaseAdapter(HttpRetrievalAdapter):
    """Adapter for Memobase (memodb-io/memobase) v0.0.42+ API.

    API flow:
      1. POST /api/v1/users              → create user (returns UUID)
      2. POST /api/v1/blobs/insert/{uid}?wait_process=true
                                          → insert chat blob + flush buffer
      3. GET  /api/v1/users/context/{uid} → retrieve memory context
    """

    def __init__(self, *, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        base_url = str(cfg.get("base_url") or os.getenv("MEMOBASE_BASE_URL") or "").strip()
        api_key = str(cfg.get("api_key") or os.getenv("MEMOBASE_API_TOKEN") or "").strip() or None
        if not base_url:
            raise ValueError("MemobaseAdapter requires base_url (cfg.memobase.base_url or MEMOBASE_BASE_URL)")
        if not api_key:
            raise ValueError("MemobaseAdapter requires api_key (cfg.memobase.api_key or MEMOBASE_API_TOKEN)")

        # Memobase API requires Bearer token format
        if not api_key.lower().startswith("bearer "):
            api_key = f"Bearer {api_key}"

        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_s=int(cfg.get("timeout_s", 120)),
            max_retries=int(cfg.get("max_retries", 3)),
        )
        self._user_ids: Dict[str, str] = {}  # namespace → Memobase UUID
        # Retries for errors reported in the response BODY (errno), on top of
        # the transport retries of _request_with_retry.
        self.errno_retries = max(1, int(cfg.get("errno_retries", 6)))
        self._api_stats: Dict[str, Dict[str, int]] = {}

    # ── errno-checked calls ─────────────────────────────────────────────

    def _stat(self, op: str, key: str) -> None:
        d = self._api_stats.setdefault(op, {"calls": 0, "ok": 0, "errno_responses": 0, "retries": 0, "failed": 0})
        d[key] += 1

    def api_stats(self) -> Dict[str, Dict[str, int]]:
        """Per-operation counters: calls, ok, errno_responses (bodies with an
        error), retries, failed (gave up). build_memory_cache writes them to
        the cache summary."""
        return {op: dict(v) for op, v in self._api_stats.items()}

    def _errno_backoff(self, attempt: int) -> float:
        return min(30.0, 1.0 * (2 ** max(0, attempt - 1)))

    async def _call(
        self,
        method: str,
        path: str,
        *,
        op: str,
        json_body: Optional[dict] = None,
        timeout_s: Optional[int] = None,
    ) -> Dict[str, Any]:
        """HTTP call that also treats an error in the body as a failure.

        HTTP 200 is not success: the body's errno must be 0/absent. Transient
        body errors (errno 500 etc., "too many clients") are retried with
        exponential backoff; the rest, and transient errors that outlast
        ``errno_retries``, raise MemobaseAPIError. Every outcome is counted
        in api_stats().
        """
        self._stat(op, "calls")
        for attempt in range(1, self.errno_retries + 1):
            try:
                data = await self._request_with_retry(method, path, json_body=json_body, timeout_s=timeout_s)
            except Exception:
                self._stat(op, "failed")
                raise
            err = _body_error(data)
            if err is None:
                self._stat(op, "ok")
                return data
            self._stat(op, "errno_responses")
            errno, errmsg = err
            if not _is_transient(errno, errmsg) or attempt >= self.errno_retries:
                self._stat(op, "failed")
                raise MemobaseAPIError(op, errno, errmsg)
            self._stat(op, "retries")
            await asyncio.sleep(self._errno_backoff(attempt))
        raise AssertionError("unreachable")

    async def health(self) -> dict:
        """Memobase's health endpoint is /api/v1/healthcheck, not the
        default /health that HttpRetrievalAdapter probes. Overriding so
        sanity gate G0 doesn't false-fail with a 404."""
        import time
        t0 = time.time()
        try:
            await self._request_with_retry(
                "GET", _MEMOBASE_HEALTH_PATH, timeout_s=min(self.timeout_s, 10)
            )
            return {
                "ok": True,
                "detail": "reachable (healthcheck)",
                "latency_ms": int((time.time() - t0) * 1000),
            }
        except Exception as e:
            return {
                "ok": False,
                "detail": f"{_MEMOBASE_HEALTH_PATH}: {e}",
                "latency_ms": int((time.time() - t0) * 1000),
            }

    async def _ensure_user(self, namespace: str) -> str:
        """Create a Memobase user for the namespace if not already created.

        Under high concurrency the server may return {"data": null,
        "errno": <X>, "errmsg": "..."}. Retry a few times before bailing.
        """
        if namespace in self._user_ids:
            return self._user_ids[namespace]

        last_resp: Any = None
        for attempt in range(1, 6):
            try:
                data = await self._call("POST", "/api/v1/users", op="user_create", json_body={})
            except Exception as e:
                last_resp = f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(min(2.0, 0.25 * attempt))
                continue
            last_resp = data
            inner = data.get("data") if isinstance(data, dict) else None
            uid = inner.get("id", "") if isinstance(inner, dict) else ""
            if uid:
                self._user_ids[namespace] = uid
                return uid
            await asyncio.sleep(min(2.0, 0.25 * attempt))

        raise RuntimeError(
            f"Failed to create Memobase user for namespace={namespace} after 5 retries: {last_resp}"
        )

    async def reset(self, *, namespace: str) -> None:
        """DELETE the memobase user for `namespace` so the next ingest pass
        starts from a clean profile/event store. The pre-release pipeline inherits the
        no-op base; MemArena explicitly cleans up because build_memory_cache
        re-resets every ego up-front."""
        uid = self._user_ids.pop(namespace, None)
        if not uid:
            return None
        try:
            await self._call(
                "DELETE",
                f"/api/v1/users/{uid}",
                op="user_delete",
                timeout_s=min(self.timeout_s, 30),
            )
        except Exception as e:
            print(
                f"[memobase_adapter.reset] ns={namespace}: delete user {uid} failed: "
                f"{type(e).__name__}: {str(e)[:200]}",
                file=sys.stderr,
            )
        return None

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        """Batch-ingest via Memobase's multi-message blob format.

        Two optimizations over the original per-message loop:
          1. Group consecutive same-thread_id messages into ONE blob's
             `blob_data.messages` list. Memobase's API natively accepts
             a list of messages in one blob — we were wasting it by
             always sending lists of length 1.
          2. Fire all per-thread blob inserts CONCURRENTLY via
             asyncio.gather. Memobase's server-side LLM extraction is
             deferred to the final flush, so parallel blob inserts are
             safe and round-trip limited, not LLM-time limited.

        A 12-turn session goes from 12 serial POSTs to 1. An
        agent-day with 10 sessions goes from ~120 serial POSTs to 10
        concurrent POSTs + 1 final flush.
        """
        t0 = time.time()
        uid = await self._ensure_user(namespace)

        if not messages:
            return {
                "namespace": namespace,
                "memobase_user_id": uid,
                "indexed_messages": 0,
                "n_errors": 0,
                "first_errors": [],
                "n_threads": 0,
                "latency_ms": 0,
                "backend_meta": {"adapter": "memobase"},
            }

        # Group by thread_id, preserving order so conversational context
        # in each blob is coherent.
        by_thread: "OrderedDict[str, List[MessageEntry]]" = OrderedDict()
        for m in messages:
            by_thread.setdefault(m.thread_id, []).append(m)

        async def _insert_thread_blob(tid: str, thread_msgs: List[MessageEntry]) -> Tuple[int, Optional[str]]:
            # Map role by speaker identity vs the ego namespace owning this
            # memobase user_id. Memobase's profile extractor only mines
            # facts from role="user" turns; role="assistant" turns are kept
            # for conversational context but are NOT folded into the
            # ego's profile. Without this split, multi-speaker MASim
            # sessions cause cross-user contamination — every
            # participant's statements get extracted as facts about the
            # ego (we observed e.g. abigail_ross's profile listing all
            # 11 corpus users).
            payload_msgs = []
            for m in thread_msgs:
                speaker = (
                    m.meta.get("speaker")
                    or (str(m.user_id) if m.user_id is not None else "unknown")
                )
                role = "user" if speaker == namespace else "assistant"
                payload_msgs.append({
                    "role": role,
                    "alias": speaker,
                    "content": m.text,
                })
            payload: Dict[str, object] = {
                "blob_type": "chat",
                "blob_data": {"messages": payload_msgs},
            }
            if thread_msgs[0].occur_ts:
                payload["created_at"] = thread_msgs[0].occur_ts
            try:
                await self._call(
                    "POST",
                    f"/api/v1/blobs/insert/{uid}",
                    op="insert",
                    json_body=payload,
                    timeout_s=60,
                )
                return (len(thread_msgs), None)
            except Exception as e:
                return (
                    -len(thread_msgs),
                    f"{type(e).__name__}: {str(e)[:200]} "
                    f"(tid={tid} n={len(thread_msgs)})",
                )

        results = await asyncio.gather(
            *(_insert_thread_blob(tid, msgs) for tid, msgs in by_thread.items())
        )

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
                f"[memobase_adapter.add] ns={namespace}: {added} ok, "
                f"{n_errors} errors across {len(by_thread)} threads; "
                f"first: {first_errors}",
                file=sys.stderr,
            )

        # Explicitly flush all pending buffers (wait for server-side LLM
        # processing of the blobs we just inserted). This is one call
        # regardless of how many blobs we pushed.
        flush_err: Optional[str] = None
        try:
            await self._call(
                "POST",
                f"/api/v1/users/buffer/{uid}/chat?wait_process=true",
                op="flush",
                json_body=None,
                timeout_s=1200,  # 20 min — Q3-32B bf16 profile extraction can take 6-16 min
            )
        except Exception as e:
            flush_err = f"{type(e).__name__}: {str(e)[:200]}"
            print(
                f"[memobase_adapter.add] ns={namespace}: buffer flush failed: {flush_err}",
                file=sys.stderr,
            )

        return {
            "namespace": namespace,
            "memobase_user_id": uid,
            "indexed_messages": added,
            "submitted_messages": len(messages),
            "n_errors": n_errors,
            "n_threads": len(by_thread),
            "first_errors": first_errors,
            "flush_error": flush_err,
            "latency_ms": int((time.time() - t0) * 1000),
            "backend_meta": {"adapter": "memobase"},
        }

    async def search(self, *, namespace: str, query: str, top_k: int, **_: Any) -> List[SearchHit]:
        # Scope kwargs (ego_id, query_timestamp, ...) are ignored: the store
        # is namespaced per ego at ingest and holds what was ingested.
        """Hybrid retrieval: aggregate user profile context + per-query event
        gists. The /context endpoint gives a coherent profile snapshot but is
        query-agnostic — every question gets the same blob, so MASim's
        d1_conflict / d2_anaphora / d7_qa families (which require specific
        recallable events) score near zero. Memobase's
        /event_gist/search/{uid} endpoint returns embedding-ranked event
        summaries; combining the two surfaces both stable persona facts and
        query-relevant episodic memories.

        Requires server config `enable_event_embedding: true`. If event
        embedding is disabled the gist search returns errno=501 and we fall
        back to context-only. Any other error (after the transient-error
        retries of _call) raises instead of returning an empty memory.
        """
        uid = self._user_ids.get(namespace)
        if not uid:
            return []

        async def _fetch_context() -> str:
            # A failed fetch raises: an empty context would pass for "nothing
            # remembered" (build_memory_cache records the row as MEMORY_ERROR).
            try:
                data = await self._call(
                    "GET",
                    f"/api/v1/users/context/{uid}?max_token_size=4096",
                    op="context",
                    timeout_s=30,
                )
            except Exception as e:
                print(
                    f"[memobase_adapter.search] ns={namespace}: context fetch failed: "
                    f"{type(e).__name__}: {str(e)[:150]}",
                    file=sys.stderr,
                )
                raise
            return ((data or {}).get("data") or {}).get("context") or ""

        async def _fetch_event_gists() -> list:
            try:
                data = await self._call(
                    "GET",
                    f"/api/v1/users/event_gist/search/{uid}"
                    f"?query={_url_quote(query)}&topk={max(1, int(top_k))}",
                    op="event_gist",
                    timeout_s=30,
                )
            except MemobaseAPIError as e:
                if not _is_transient(e.errno, e.errmsg):
                    # e.g. errno 501: event embedding disabled -> context only.
                    return []
                print(
                    f"[memobase_adapter.search] ns={namespace}: event_gist search failed: {e}",
                    file=sys.stderr,
                )
                raise
            except Exception as e:
                print(
                    f"[memobase_adapter.search] ns={namespace}: event_gist search failed: "
                    f"{type(e).__name__}: {str(e)[:150]}",
                    file=sys.stderr,
                )
                raise
            inner = (data or {}).get("data") or {}
            if isinstance(inner, dict):
                # Memobase v0.0.42 returns {"data":{"gists":[...],"events":[]}}
                return (
                    inner.get("gists")
                    or inner.get("events")
                    or inner.get("event_gists")
                    or inner.get("results")
                    or []
                )
            return inner if isinstance(inner, list) else []

        context_text, gist_rows = await asyncio.gather(_fetch_context(), _fetch_event_gists())

        out: List[SearchHit] = []
        if context_text:
            out.append(
                SearchHit(
                    msg_id=f"memobase_context_{namespace}",
                    score=1.0,
                    text=context_text,
                    occur_ts="",
                    thread_id=namespace,
                    user_id=None,
                )
            )

        for i, row in enumerate(gist_rows):
            if not isinstance(row, dict):
                continue
            text = (
                row.get("gist_data", {}).get("content") if isinstance(row.get("gist_data"), dict) else None
            ) or row.get("content") or row.get("text") or row.get("summary") or ""
            if not text:
                continue
            score = row.get("score") or row.get("similarity")
            try:
                score = float(score) if score is not None else max(0.0, 0.95 - i * 0.05)
            except (TypeError, ValueError):
                score = max(0.0, 0.95 - i * 0.05)
            out.append(
                SearchHit(
                    msg_id=str(row.get("id") or row.get("event_id") or f"memobase_gist_{namespace}_{i}"),
                    score=score,
                    text=str(text),
                    occur_ts=str(row.get("created_at") or row.get("event_timestamp") or ""),
                    thread_id=namespace,
                    user_id=None,
                )
            )

        return out
