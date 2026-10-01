from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from openai import AsyncOpenAI, OpenAI

from MASim.prompts import ANSWERING_DIM_HINTS, ANSWERING_SYSTEM_PARTS
from memarena.text_cleaning import strip_think, think_status

from .adapters.base import PromptStyle
from .types import AnswerRecord, MessageEntry, QAItem, SearchHit
from .types_light import LightAnswerRecord

_THINK_PREFIX_RE = re.compile(r"^Thinking Process:\s*", re.IGNORECASE)

_CHARS_PER_TOKEN = 3.0  # conservative estimate for Qwen/Mistral
TEST_STUB_ANSWER = "I don't know"

# Thread-safe JSONL marker writer for hardware telemetry (spark_remote.md §8b)
_hw_marker_lock = threading.Lock()


def _emit_hw_marker(
    marker_path: Optional[str],
    *,
    instance_id: str,
    phase: str,
    llm_model: str,
    ts_start_ms: int,
    ts_end_ms: int,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    call_idx: int = 0,
) -> None:
    """Append one JSONL row to the hw marker file (if configured)."""
    if not marker_path:
        return
    row = {
        "ts_start_ms": ts_start_ms,
        "ts_end_ms": ts_end_ms,
        "instance_id": instance_id,
        "phase": phase,
        "llm_model": llm_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "call_idx": call_idx,
    }
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _hw_marker_lock:
        with open(marker_path, "a") as f:
            f.write(line)


def _is_mistral_model(model: str) -> bool:
    """Check if model is Mistral/Ministral (no system role support)."""
    m = model.lower()
    return "mistral" in m or "ministral" in m


def _is_gemma2_model(model: str) -> bool:
    """Gemma-2 has hard 8192 context cap; vanilla TEXT_SESSIONS truncation
    must be tightened to leave headroom for the chat template + generation."""
    return "gemma-2" in model.lower()


def _is_no_system_model(model: str) -> bool:
    """Models whose chat template does not accept role='system'.

    Both Mistral and Gemma-2 instruct templates require user-first messages
    with strictly alternating user/assistant turns. The system content must
    be prepended into the first user message via _merge_system_into_user.
    """
    m = model.lower()
    return (
        "mistral" in m
        or "ministral" in m
        or "gemma-2" in m
    )


def _merge_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge system message into first user message for Mistral models.

    Mistral chat templates require strictly alternating user/assistant roles
    and do not support role="system". This prepends the system content to the
    first user message.
    """
    if not messages or messages[0]["role"] != "system":
        return messages
    system_content = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": f"{system_content}\n\n{rest[0]['content']}"}
    else:
        rest.insert(0, {"role": "user", "content": system_content})
    return rest


class AsyncTokenBudgetSemaphore:
    """Async version of TokenBudgetSemaphore: allow concurrent requests as long as
    total in-flight tokens < budget. Always allows at least one request through."""

    def __init__(self, budget: int):
        self._budget = budget
        self._in_flight = 0
        self._cond = asyncio.Condition()

    async def acquire(self, tokens: int) -> None:
        async with self._cond:
            while self._in_flight > 0 and self._in_flight + tokens > self._budget:
                await self._cond.wait()
            self._in_flight += tokens

    async def release(self, tokens: int) -> None:
        async with self._cond:
            self._in_flight -= tokens
            self._cond.notify_all()


def _strip_thinking(text: str) -> str:
    """Remove thinking/reasoning wrappers from LLM output.

    Closed ``<think>`` blocks are removed, and an unclosed one (generation cut
    at max_tokens while thinking) is dropped with everything after it; see
    ``memarena.text_cleaning.strip_think``.
    """
    text = strip_think(text)
    text = _THINK_PREFIX_RE.sub("", text).strip()
    return text


class ThinkingEnabledError(RuntimeError):
    """Raised (with AnswerConfig.fail_on_thinking) when a cell's answers carry
    reasoning although every request sends ``enable_thinking=False``: the
    serving stack ignored the flag."""


@dataclass
class AnswerConfig:
    model: str = "Qwen3-Coder"
    endpoint: Optional[str] = "http://127.0.0.1:8000/v1"
    api_key: Optional[str] = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 400
    context_length: int = 0  # 0 = no limit; otherwise cap prompt to fit this token budget
    timeout_seconds: int = 120
    concurrency: int = 32
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    provider_order: Optional[List[str]] = None
    allow_fallbacks: bool = False
    cache_stats: bool = True

    # backend: llm | openclaw-rag | openclaw-session
    backend: str = "llm"

    # openclaw backend settings
    openclaw_bin: str = "openclaw"
    openclaw_session_id: Optional[str] = None
    openclaw_stateless: bool = False
    openclaw_model: Optional[str] = None
    openclaw_timeout_seconds: int = 60
    openclaw_ingest_chunk_lines: int = 120
    openclaw_inject_soul: bool = False
    openclaw_soul_path: Optional[str] = "SOUL.md"
    openclaw_cleanup_after_run: bool = False
    openclaw_ingest_mode: str = "ack"  # ack | chat
    openclaw_user_context_path: Optional[str] = None
    openclaw_user_context_max_chars: int = 12000
    openclaw_console_log_path: Optional[str] = None
    interleaved_mode: bool = False
    show_progress: bool = True
    verbose: bool = False
    log_every: int = 50
    print_full_io: bool = False

    # two-phase timing: fraction of queries per dimension at low concurrency
    timing_fraction: float = 0.0
    timing_concurrency: int = 4
    timing_only: bool = False

    # separate judge LLM (if empty, falls back to answering model)
    judge_model: str = ""
    judge_endpoint: Optional[str] = None
    judge_api_key: Optional[str] = None

    # secondary judge for cross-judge validation (e.g. GPT-4o-mini)
    secondary_judge_model: str = ""
    secondary_judge_endpoint: Optional[str] = None
    secondary_judge_api_key: Optional[str] = None

    # prompt variant for robustness ablation (overrides dim hints)
    prompt_variant: Optional[str] = None

    # hardware marker file for Spark edge-deployment telemetry (§8b)
    hw_marker_file: Optional[str] = None

    # D6 permission prompt arm: "A" (default, original prompt) | "B"
    # (Constitutional-AI-style permission frame). Only affects d4_permission.
    d6_arm: str = "A"

    # D6 paired-probe protocol mode (only affects d4_permission items):
    #   "third_party" (default) -- asker is the original query_agent from GT.
    #   "self_ego"    -- asker is overridden to ego_agent_id (the user themselves).
    # The self_ego probe is the second half of the paired-probe access-control
    # test: by comparing each item's third_party vs self_ego output, we can
    # tell whether the system actually possesses the fact (would disclose to
    # ego) and whether it gates disclosure on requester identity. Outputs from
    # self_ego runs should be written to a separate directory so the third_party
    # results are not overwritten.
    d6_probe_mode: str = "third_party"

    # G3a: reader-side norm-binding test.
    # When True, injects [access:DENY] or [access:ALLOW] into the prompt for
    # d4_permission items at scoring time, derived from policy_expected in QA
    # metadata. Tests whether the reader can bind explicit access markers to
    # disclosure decisions (without this, F2's claim stands; with it, F2 must soften).
    d6_inject_access_marker: bool = False
    # D6 answer prompts carry the fact owner's contact list with closeness labels
    # (metadata["owner_contacts"], from the run's social graph); a fact, not a policy.
    d6_owner_contacts: bool = True
    # D6 answer prompts name who is asking; False says only "Someone" (identity ablation).
    d6_show_asker: bool = True

    # Test mode preserves the real pipeline shape but replaces all answer-side
    # LLM/OpenClaw calls with a fixed local string.
    test_mode: bool = False

    # Thinking guard. Every request sends enable_thinking=False, but a serving
    # stack can ignore it (the H100 SGLang did). Once at least
    # think_min_answers answers are in and more than think_max_fraction of
    # them carry reasoning, the engine warns loudly on stderr; with
    # fail_on_thinking it raises ThinkingEnabledError and the cell aborts.
    think_max_fraction: float = 0.02
    think_min_answers: int = 20
    fail_on_thinking: bool = False


class AnswerEngine:
    def __init__(
        self,
        cfg: AnswerConfig,
        *,
        prompt_style: PromptStyle = PromptStyle.JSON_CONTEXT,
        corpus_sessions: Optional[Dict[str, Any]] = None,
        ego_session_map: Optional[Dict[str, Any]] = None,
        adapter_name: str = "",
        context_from_hits: bool = False,
    ) -> None:
        self.cfg = cfg
        self._prompt_style = prompt_style
        self._corpus_sessions = corpus_sessions or {}
        self._ego_session_map = ego_session_map or {}
        self._adapter_name = adapter_name.lower()
        # TEXT_SESSIONS adapters whose context IS their search hits (Oracle +
        # distractors, BM25 in TEXT_SESSIONS format). Vanilla / Oracle /
        # omniscient build their context from session ids instead.
        self._context_from_hits = bool(context_from_hits)
        self._turn_index: Optional[Dict[str, tuple]] = None
        self._client: Optional[OpenAI] = None
        self._aclient: Optional[AsyncOpenAI] = None

        self._backend = str(cfg.backend or "llm").strip().lower()
        if self._backend not in {"llm", "openclaw-rag", "openclaw-session"}:
            self._backend = "llm"

        self._openclaw_session_id = cfg.openclaw_session_id or f"memarena-openclaw-{int(time.time())}"
        self._openclaw_model = (str(cfg.openclaw_model).strip() if cfg.openclaw_model else None)
        self._ingest_mode = (str(getattr(cfg, "openclaw_ingest_mode", "ack") or "ack").strip().lower())
        if self._ingest_mode not in {"ack", "chat"}:
            self._ingest_mode = "ack"
        self._session_ingested: Dict[str, bool] = {}
        self._active_namespace: Optional[str] = None
        self._prepared_sessions: set[str] = set()
        self._soul_loaded_sessions: set[str] = set()
        self._protocol_loaded_sessions: set[str] = set()
        self._touched_sessions: set[str] = set()
        self._session_prepare_lock = threading.Lock()
        self._policy_soul_text = self._load_policy_soul_text()
        self._user_context_by_id, self._user_context_blob = self._load_user_context()
        self._openclaw_call_count = 0
        self._llm_cache_hits = 0
        self._llm_cache_total = 0
        self._think_answers = 0
        self._think_detected = 0
        self._think_truncated = 0
        self._think_warned = False

        if self._backend == "openclaw-session":
            # Session-memory baseline must be sequential for stable state.
            self.cfg.concurrency = 1
            self.cfg.openclaw_stateless = False

        if self._backend == "llm" and not bool(getattr(cfg, "test_mode", False)):
            api_key = cfg.api_key or os.getenv("LLM_API_KEY")
            endpoint = cfg.endpoint or os.getenv("LLM_BASE_URL")

            enabled = bool(api_key)
            if enabled:
                kwargs = {"api_key": api_key, "timeout": cfg.timeout_seconds}
                if endpoint:
                    kwargs["base_url"] = endpoint
                self._client = OpenAI(**kwargs)
                self._aclient = AsyncOpenAI(**kwargs)

        self._log(
            "init "
            f"backend={self._backend} model={self.cfg.model} "
            f"openclaw_model={self._openclaw_model or '-'} ingest_mode={self._ingest_mode} "
            f"inject_soul={bool(self.cfg.openclaw_inject_soul)} soul_loaded={bool((self._policy_soul_text or '').strip())} "
            f"user_ctx={len(self._user_context_by_id)} interleaved={self.cfg.interleaved_mode} "
            f"test_mode={bool(getattr(self.cfg, 'test_mode', False))}"
        )
        if self._openclaw_model:
            self._log("note: openclaw_model is ignored in strict benchmark mode (no pre-prompts)")

    def _log(self, msg: str) -> None:
        if not bool(self.cfg.verbose):
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[eval.answer][{ts}] {msg}", file=os.sys.stderr, flush=True)

    def _log_full_io(self, *, session_id: str, prompt: str, raw_output: str) -> None:
        if not bool(self.cfg.print_full_io):
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"\n[eval.answer.full_io][{ts}] session={session_id} INPUT_BEGIN\n"
            f"{prompt}\n"
            f"[eval.answer.full_io][{ts}] INPUT_END",
            file=os.sys.stderr,
            flush=True,
        )
        print(
            f"[eval.answer.full_io][{ts}] session={session_id} OUTPUT_BEGIN\n"
            f"{raw_output}\n"
            f"[eval.answer.full_io][{ts}] OUTPUT_END\n",
            file=os.sys.stderr,
            flush=True,
        )

    def _append_openclaw_console_log(self, *, session_id: str, prompt: str, stdout_text: str, stderr_text: str, phase: str) -> None:
        p = (self.cfg.openclaw_console_log_path or "").strip()
        if not p:
            return
        try:
            path = Path(p).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            blob = (
                f"\n===== [{ts}] phase={phase} session={session_id} =====\n"
                f"PROMPT:\n{prompt}\n"
                f"--- STDOUT ---\n{stdout_text}\n"
                f"--- STDERR ---\n{stderr_text}\n"
            )
            with path.open("a", encoding="utf-8") as f:
                f.write(blob)
        except Exception as e:
            self._log(f"console_log_failed: {type(e).__name__}: {e}")

    def needs_search_context(self) -> bool:
        if self._prompt_style == PromptStyle.TEXT_SESSIONS and not self._context_from_hits:
            return False  # context built from corpus_sessions, not search hits
        return self._backend in {"llm", "openclaw-rag"}

    def supports_stream_ingest(self) -> bool:
        return self._backend in {"openclaw-rag", "openclaw-session"}

    # ── Thinking guard ───────────────────────────────────────────────────────

    def _note_thinking(self, raw_text: str, llm_meta: Mapping[str, Any]) -> tuple[bool, bool]:
        """Classify one raw answer as (think_detected, think_truncated).

        Updates the cell's counters and runs the guard. ``think_truncated``
        means no answer is left outside the reasoning (budget used up while
        thinking); the caller then records an empty prediction.
        """
        detected, truncated = think_status(raw_text)
        if llm_meta.get("reasoning_chars"):
            detected = True
            truncated = not strip_think(raw_text)
        self._think_answers += 1
        self._think_detected += int(detected)
        self._think_truncated += int(truncated)
        self.check_thinking()
        return detected, truncated

    def thinking_report(self) -> Dict[str, Any]:
        n = self._think_answers
        return {
            "answers": n,
            "think_detected": self._think_detected,
            "think_truncated": self._think_truncated,
            "think_fraction": round(self._think_detected / n, 4) if n else 0.0,
            "think_max_fraction": float(self.cfg.think_max_fraction),
        }

    def check_thinking(self, *, final: bool = False) -> None:
        """Warn (or raise, with cfg.fail_on_thinking) when thinking is on.

        Checked after every answer once cfg.think_min_answers are in, and once
        more at the end of the answer stage (``final=True``) for small cells.
        """
        n = self._think_answers
        if n == 0 or (not final and n < max(1, int(self.cfg.think_min_answers))):
            return
        if self._think_detected / n <= float(self.cfg.think_max_fraction):
            return
        msg = (
            f"thinking is ON for model={self.cfg.model}: {self._think_detected}/{n} answers carry "
            f"<think> reasoning ({self._think_truncated} cut inside it and recorded as empty) although "
            f"every request sends enable_thinking=False; the serving stack ignores the flag "
            f"(threshold {float(self.cfg.think_max_fraction):.0%})"
        )
        if bool(self.cfg.fail_on_thinking):
            raise ThinkingEnabledError(msg)
        if not self._think_warned:
            self._think_warned = True
            print(f"[eval.answer][WARNING] {msg}", file=os.sys.stderr, flush=True)

    async def answer_one(self, *, qa: QAItem, ctx_hits: Sequence[SearchHit]) -> AnswerRecord:
        return await self._answer_one(qa=qa, ctx_hits=list(ctx_hits))

    async def ingest_message(self, *, namespace: str, message: MessageEntry) -> Dict[str, object]:
        if not self.supports_stream_ingest():
            return {"enabled": False, "reason": "backend_no_stream_ingest"}

        if bool(getattr(self.cfg, "test_mode", False)):
            sid = self._session_id(namespace) if self._backend == "openclaw-session" else self._openclaw_session_id
            self._active_namespace = namespace
            return {
                "enabled": True,
                "mode": "test_stub",
                "session_id": sid,
                "msg_id": message.msg_id,
                "assistant_reply": TEST_STUB_ANSWER,
            }

        sid = self._session_id(namespace) if self._backend == "openclaw-session" else self._openclaw_session_id
        self._active_namespace = namespace
        prompt = self._build_ingest_prompt(chunk=[self._message_line(message)], chunk_idx=1, total_chunks=1)
        ack = await asyncio.to_thread(self._run_openclaw, sid, prompt)
        return {
            "enabled": True,
            "session_id": sid,
            "msg_id": message.msg_id,
            "assistant_reply": ack,
        }

    async def ingest_messages(self, *, namespace: str, messages: Sequence[MessageEntry]) -> Dict[str, object]:
        """Preload transcript into an OpenClaw session for session-memory baseline."""
        if self._backend != "openclaw-session":
            return {"enabled": False, "reason": "backend_not_openclaw_session"}

        if bool(getattr(self.cfg, "test_mode", False)):
            sid = self._session_id(namespace)
            self._active_namespace = namespace
            self._session_ingested[namespace] = True
            return {
                "enabled": True,
                "mode": "test_stub",
                "session_id": sid,
                "chunks": 0,
                "messages": len(messages),
                "assistant_reply": TEST_STUB_ANSWER,
            }

        if self._session_ingested.get(namespace):
            sid = self._session_id(namespace)
            return {
                "enabled": True,
                "session_id": sid,
                "already_ingested": True,
            }

        sid = self._session_id(namespace)
        self._active_namespace = namespace
        chunk_size = max(1, int(self.cfg.openclaw_ingest_chunk_lines))
        lines = [self._message_line(m) for m in messages]
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

        for idx, chunk in enumerate(chunks, start=1):
            prompt = self._build_ingest_prompt(chunk=chunk, chunk_idx=idx, total_chunks=len(chunks))
            await asyncio.to_thread(self._run_openclaw, sid, prompt)

        self._session_ingested[namespace] = True
        return {
            "enabled": True,
            "session_id": sid,
            "chunks": len(chunks),
            "messages": len(lines),
        }

    async def answer_batch(
        self,
        *,
        qas: Sequence[QAItem],
        contexts: Sequence[List[SearchHit]],
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> List[AnswerRecord]:
        if len(qas) != len(contexts):
            raise ValueError("qas and contexts length mismatch")

        out: List[Optional[AnswerRecord]] = [None] * len(qas)

        if self._prompt_style == PromptStyle.TEXT_SESSIONS and self.cfg.context_length > 0:
            # Token-budget-aware scheduling: pre-compute prompt sizes, use weighted semaphore
            # Scale total in-flight token budget with concurrency so the semaphore
            # doesn't serialize requests when concurrency is high.
            token_budget = int(self.cfg.context_length * max(1, int(self.cfg.concurrency)) * 0.6)
            tsem = AsyncTokenBudgetSemaphore(token_budget)
            # Also enforce flat concurrency cap (critical for timing runs at concurrency=1)
            conc_sem = asyncio.Semaphore(max(1, int(self.cfg.concurrency)))

            # Pre-compute prompt token counts
            token_counts: List[int] = []
            for qa, ctx_hits in zip(qas, contexts):
                try:
                    system_p, user_p = self._build_text_prompt(qa, ctx_hits)
                    n_tokens = int((len(system_p) + len(user_p)) / _CHARS_PER_TOKEN)
                except Exception:
                    n_tokens = 4096  # conservative fallback
                token_counts.append(n_tokens)

            async def _run_budgeted(i: int, qa: QAItem, ctx_hits: List[SearchHit], n_tok: int) -> None:
                async with conc_sem:
                    await tsem.acquire(n_tok)
                    try:
                        out[i] = await self._answer_one(qa=qa, ctx_hits=ctx_hits)
                    finally:
                        await tsem.release(n_tok)
                if progress_cb is not None:
                    progress_cb(1)

            await asyncio.gather(*[
                _run_budgeted(i, qa, contexts[i], token_counts[i])
                for i, qa in enumerate(qas)
            ])
        else:
            # Flat concurrency semaphore for JSON_CONTEXT path
            sem = asyncio.Semaphore(max(1, int(self.cfg.concurrency)))

            async def _run(i: int, qa: QAItem, ctx_hits: List[SearchHit]) -> None:
                async with sem:
                    out[i] = await self._answer_one(qa=qa, ctx_hits=ctx_hits)
                if progress_cb is not None:
                    progress_cb(1)

            await asyncio.gather(*[_run(i, qa, contexts[i]) for i, qa in enumerate(qas)])

        return [x for x in out if x is not None]

    async def answer_batch_lightweight(
        self,
        *,
        qas: Sequence[QAItem],
        contexts: Sequence[List[SearchHit]],
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> List[LightAnswerRecord]:
        full = await self.answer_batch(qas=qas, contexts=contexts, progress_cb=progress_cb)
        return [
            LightAnswerRecord(
                question_id=r.question_id,
                prediction=r.prediction,
                model=r.model,
                raw_response=r.raw_response,
            )
            for r in full
        ]

    async def _answer_one(self, *, qa: QAItem, ctx_hits: List[SearchHit]) -> AnswerRecord:
        if bool(getattr(self.cfg, "test_mode", False)):
            return AnswerRecord(
                question_id=qa.question_id,
                question=qa.question,
                answer=qa.answer,
                prediction=TEST_STUB_ANSWER,
                model=f"{self.cfg.model}:test-stub",
                raw_response=TEST_STUB_ANSWER,
                answer_time_ms=0.0,
                ttft_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # Route to direct-eval path for TEXT_SESSIONS prompt style
        if self._prompt_style == PromptStyle.TEXT_SESSIONS:
            return await self._answer_one_direct(qa=qa, ctx_hits=ctx_hits)

        context_lines = [
            {
                "msg_id": h.msg_id,
                "thread_id": h.thread_id,
                "occur_ts": h.occur_ts,
                "user_id": h.user_id,
                "text": h.text,
            }
            for h in ctx_hits
        ]

        if self._backend in {"openclaw-rag", "openclaw-session"}:
            try:
                if self._backend == "openclaw-rag":
                    raw_text = await asyncio.to_thread(self._ask_openclaw_rag, qa, ctx_hits)
                    model_name = "baseline_simplerag"
                else:
                    raw_text = await asyncio.to_thread(self._ask_openclaw_session, qa)
                    model_name = "baseline_session"

                parsed_answer = self._normalize_answer(raw_text, qa.question_type)
                if not parsed_answer:
                    parsed_answer = "UNKNOWN"
                return AnswerRecord(
                    question_id=qa.question_id,
                    question=qa.question,
                    answer=qa.answer,
                    prediction=parsed_answer,
                    model=model_name,
                    raw_response=raw_text,
                )
            except Exception as e:
                return AnswerRecord(
                    question_id=qa.question_id,
                    question=qa.question,
                    answer=qa.answer,
                    prediction="UNKNOWN",
                    model="openclaw-error",
                    raw_response=f"OPENCLAW_ERROR: {type(e).__name__}: {e}",
                )

        if not self._client:
            # Fallback heuristic for offline smoke testing.
            guess = self._heuristic_answer(qa, ctx_hits)
            return AnswerRecord(
                question_id=qa.question_id,
                question=qa.question,
                answer=qa.answer,
                prediction=guess,
                model="heuristic-offline",
                raw_response=guess,
            )

        # Filter out excluded sessions (e.g. D5 cloze: exclude the tested session)
        exclude_tids = set((qa.metadata or {}).get("exclude_thread_ids") or [])
        if exclude_tids:
            context_lines = [c for c in context_lines if c.get("thread_id") not in exclude_tids]

        # Separate shared context (cacheable prefix) from per-question payload.
        context_prefix = None
        if context_lines:
            context_prefix = json.dumps(
                {"conversation_context": context_lines},
                ensure_ascii=False,
            )

        payload: Dict[str, object] = {
            "task": "memarena_answer",
            "question_type": qa.question_type,
            "question": qa.question,
            "options": qa.options,
            "output_format": {
                "answer": "string",
                "reason": "short string",
            },
        }

        _qa_dim = str((qa.metadata or {}).get("dimension")
                      or (qa.metadata or {}).get("task_family") or "")

        # G3a: inject [access:DENY]/[access:ALLOW] marker for norm-binding test
        if getattr(self.cfg, "d6_inject_access_marker", False) and _qa_dim == "d4_permission":
            _pexp = str((qa.metadata or {}).get("policy_expected") or "")
            if "DENY" in _pexp:
                payload["access_policy"] = "[access:DENY]"
            elif "ALLOW" in _pexp:
                payload["access_policy"] = "[access:ALLOW]"

        if _qa_dim == "d4_permission":
            _meta = qa.metadata or {}
            # Paired-probe protocol: in self_ego mode, override the asker to
            # the ego agent (the memory owner). This isolates "does the system
            # have the fact?" from "is it willing to share with a third party?"
            # See AnswerConfig.d6_probe_mode docstring.
            if str(self.cfg.d6_probe_mode).lower() == "self_ego":
                _asker_id = str(
                    _meta.get("ego_agent_id")
                    or _meta.get("query_agent")
                    or _meta.get("asker_agent_id")
                    or _meta.get("instance_query_agent")
                    or ""
                )
                # Self-probe never uses the "A stranger" framing, even if the
                # original GT marked the item as anonymous_querier.
                if _asker_id:
                    try:
                        from eval.src.scoring_core import _slug_to_name
                        payload["asker"] = _slug_to_name(_asker_id)
                    except Exception:
                        payload["asker"] = _asker_id.replace("_", " ").title()
            else:
                _asker_id = str(
                    _meta.get("query_agent")
                    or _meta.get("asker_agent_id")
                    or _meta.get("instance_query_agent")
                    or ""
                )
                if not getattr(self.cfg, "d6_show_asker", True):
                    pass
                elif _meta.get("anonymous_querier"):
                    payload["asker"] = "A stranger"
                elif _asker_id:
                    try:
                        from eval.src.scoring_core import _slug_to_name
                        payload["asker"] = _slug_to_name(_asker_id)
                    except Exception:
                        payload["asker"] = _asker_id.replace("_", " ").title()

        if _qa_dim and _qa_dim not in ("d4_permission", "d5_cloze"):
            # whose memory is asked, as the TEXT_SESSIONS prompt of Vanilla / Oracle says
            # ("Question being asked to <name>"); D6 carries owner and asker, cloze neither
            _to = str((qa.metadata or {}).get("instance_query_agent")
                      or (qa.metadata or {}).get("ego_agent_id") or "")
            if _to:
                try:
                    from eval.src.scoring_core import _slug_to_name
                    payload["question_asked_to"] = _slug_to_name(_to)
                except Exception:
                    payload["question_asked_to"] = _to.replace("_", " ").title()

        if _qa_dim == "d4_permission" and getattr(self.cfg, "d6_owner_contacts", True):
            _contacts = (qa.metadata or {}).get("owner_contacts")
            _owner_id = str((qa.metadata or {}).get("ego_agent_id") or "")
            if _contacts and _owner_id:
                try:
                    from eval.src.scoring_core import _slug_to_name
                    _owner_name = _slug_to_name(_owner_id)
                except Exception:
                    _owner_name = _owner_id.replace("_", " ").title()
                payload["owner_profile"] = {"owner": _owner_name, "contacts": _contacts}

        # Dimension-aware hints (with optional variant override)
        dim_hints = ANSWERING_DIM_HINTS
        if self.cfg.prompt_variant:
            try:
                from eval.config.prompt_variants import PROMPT_VARIANTS
                dim_hints = PROMPT_VARIANTS.get(self.cfg.prompt_variant, ANSWERING_DIM_HINTS)
            except ImportError:
                pass

        dim = str((qa.metadata or {}).get("dimension") or (qa.metadata or {}).get("task_family") or "")
        if dim == "d3_confabulation":
            eam = str((qa.metadata or {}).get("expected_answer_mode") or "")
            hint_key = "d3_confabulation_answer" if eam == "answer" else "d3_confabulation"
            hint = dim_hints.get(hint_key)
        else:
            hint = dim_hints.get(dim)
        if hint:
            payload["hint"] = hint

        owner = ""
        if dim == "d4_permission":
            ego_id = str((qa.metadata or {}).get("ego_agent_id")
                         or (qa.metadata or {}).get("instance_query_agent") or "")
            if ego_id:
                try:
                    from eval.src.scoring_core import _slug_to_name
                    owner = _slug_to_name(ego_id)
                except Exception:
                    owner = ego_id.replace("_", " ").title()

        system_prompt = self._build_llm_system_prompt(dim=dim, owner=owner)

        # Truncate context to fit within context_length token budget.
        # Truncate at message-object boundaries so serialized JSON stays valid.
        # Use ~2.5 chars/token (conservative for Qwen/Mistral on JSON with timestamps).
        _CPT = 2.5  # chars per token estimate
        if context_prefix and self.cfg.context_length > 0:
            payload_str = json.dumps(payload, ensure_ascii=False)
            overhead_tokens = (
                len(system_prompt) // _CPT
                + len(payload_str) // _CPT
                + 80  # ack message + chat template special tokens
                + int(self.cfg.max_tokens)
            )
            ctx_budget_chars = max(256, self.cfg.context_length - overhead_tokens) * _CPT

            if len(context_prefix) > ctx_budget_chars:
                # Re-truncate from context_lines (keep most recent messages).
                kept: list[dict] = []
                char_total = len('{"conversation_context": []}')
                for msg in reversed(context_lines):
                    entry_json = json.dumps(msg, ensure_ascii=False)
                    entry_chars = len(entry_json) + 2  # comma + space
                    if char_total + entry_chars > ctx_budget_chars and kept:
                        break
                    kept.append(msg)
                    char_total += entry_chars
                context_prefix = json.dumps(
                    {"conversation_context": list(reversed(kept))},
                    ensure_ascii=False,
                )

        try:
            _ts_start = int(time.time() * 1000)
            raw_text, llm_meta = await self._complete_json(
                system_prompt,
                payload,
                context_prefix=context_prefix,
            )
            _ts_end = int(time.time() * 1000)
            _emit_hw_marker(
                self.cfg.hw_marker_file,
                instance_id=qa.question_id,
                phase="answer",
                llm_model=self.cfg.model,
                ts_start_ms=_ts_start,
                ts_end_ms=_ts_end,
                prompt_tokens=llm_meta.get("prompt_tokens"),
                completion_tokens=llm_meta.get("completion_tokens"),
            )
            think_detected, think_truncated = self._note_thinking(raw_text, llm_meta)
            if think_truncated:
                # Cut inside <think>: there is no answer. Never score the
                # reasoning (or a bare "<think>"); the prediction is empty.
                parsed_answer = ""
                self._log(f"think_truncated qid={qa.question_id}: prediction recorded as empty")
            else:
                parsed_answer = self._extract_answer(raw_text, qa.question_type)
                if not parsed_answer:
                    parsed_answer = self._safe_answer_fallback(qa, raw_text)
            return AnswerRecord(
                question_id=qa.question_id,
                question=qa.question,
                answer=qa.answer,
                prediction=parsed_answer,
                model=self.cfg.model,
                raw_response=raw_text,
                answer_time_ms=llm_meta.get("elapsed_ms"),
                ttft_ms=llm_meta.get("ttft_ms"),
                prompt_tokens=llm_meta.get("prompt_tokens"),
                completion_tokens=llm_meta.get("completion_tokens"),
                think_detected=think_detected,
                think_truncated=think_truncated,
            )
        except ThinkingEnabledError:
            raise
        except Exception as e:
            # Bug B fix: do NOT fall back to BM25 evidence text. After all
            # retries are exhausted in _call_llm_with_retry, mark this row
            # as a hard error so the judge stage skips it (pre_scored=0)
            # instead of judging stale BM25 hits as if they were a model
            # answer. See scripts/patch_llm_errors.py for the post-process.
            return AnswerRecord(
                question_id=qa.question_id,
                question=qa.question,
                answer=qa.answer,
                prediction=None,
                model="llm-error",
                raw_response=f"LLM_ERROR: {type(e).__name__}: {e}",
            )

    # ── Direct-eval (TEXT_SESSIONS) prompt path ──────────────────────────────

    def _qa_to_instance(self, qa: QAItem) -> dict:
        """Convert a QAItem back to the instance dict format used by simple_eval."""
        meta = dict(qa.metadata or {})
        instance: Dict[str, Any] = {
            "instance_id": qa.question_id,
            "query": qa.question,
            "dimension": meta.get("dimension", ""),
            "difficulty": meta.get("difficulty", "medium"),
            "context_tier": meta.get("context_tier", "single_session"),
            "ego_agent_id": meta.get("ego_agent_id", meta.get("instance_query_agent", "")),
            "evidence_session_ids": meta.get("evidence_session_ids", meta.get("instance_evidence_sessions", meta.get("evidence_sessions", []))),
            "ground_truth": meta.get("ground_truth", {}),
            "metadata": meta,
        }
        return instance

    def _hits_to_turns(self, hits: Sequence[SearchHit], exclude_tids: set) -> List[dict]:
        """Turn search hits (one per corpus turn, msg_id == turn_id) back into
        the turn dicts simple_eval.build_prompt renders, with the same
        session metadata reconstruct_context injects. Only the hit turns are
        kept, in chronological order."""
        if self._turn_index is None:
            index: Dict[str, tuple] = {}
            for sid, sess in (self._corpus_sessions or {}).items():
                for turn in sess.get("turns", []):
                    tid = str(turn.get("turn_id") or "")
                    if tid:
                        index[tid] = (sid, sess, turn)
            self._turn_index = index
        turns: List[dict] = []
        seen: set = set()
        for h in hits:
            if h.msg_id in seen or (h.thread_id and h.thread_id in exclude_tids):
                continue
            seen.add(h.msg_id)
            found = self._turn_index.get(str(h.msg_id))
            if found is None:
                # Not a corpus turn: keep the hit text as-is.
                turns.append({"session_id": h.thread_id or "unknown", "speaker_id": "unknown",
                              "text": h.text, "timestamp": 0.0})
                continue
            sid, sess, turn = found
            entry = dict(turn)
            entry.setdefault("session_id", sid)
            entry.setdefault("modality", sess.get("modality", "face_to_face"))
            entry.setdefault("location_id", sess.get("location_id", ""))
            entry.setdefault("participants", sess.get("participants", []))
            turns.append(entry)
        turns.sort(key=lambda t: float(t.get("timestamp", 0.0)))
        return turns

    @staticmethod
    def _withhold_evidence(instance: Dict[str, Any], tier: str) -> None:
        """Remove every route by which build_prompt would pull gold evidence
        (evidence_session_ids, metadata.evidence_sessions / session_id, the
        full_ego fallback)."""
        instance["evidence_session_ids"] = []
        instance["context_tier"] = tier
        meta = dict(instance.get("metadata") or {})
        meta["evidence_sessions"] = []
        meta["session_id"] = ""
        instance["metadata"] = meta

    def _build_text_prompt(self, qa: QAItem, ctx_hits: Optional[Sequence[SearchHit]] = None) -> tuple[str, str]:
        """Build system+user prompts using simple_eval's text-session format."""
        from eval.simple_eval import build_prompt

        instance = self._qa_to_instance(qa)

        # Adapters whose context is their search hits (Oracle + distractors,
        # BM25 in TEXT_SESSIONS format): render exactly those turns, never the
        # gold evidence sessions.
        if self._context_from_hits:
            exclude_tids = set((qa.metadata or {}).get("exclude_thread_ids") or [])
            self._withhold_evidence(instance, "retrieved")
            instance["ego_context"] = self._hits_to_turns(ctx_hits or [], exclude_tids)

        # Oracle + retrieval-time access control: D6 DENY items get no
        # evidence at all; ALLOW items are plain Oracle.
        if (
            self._adapter_name == "oracle_gated"
            and str((qa.metadata or {}).get("policy_expected") or "") == "DENY_NO_ACCESS"
        ):
            self._withhold_evidence(instance, "evidence_only")

        # Vanilla backend: use ego session map (recent sessions) instead of
        # ground-truth evidence sessions. Clear evidence_session_ids and set
        # context_tier to full_ego so reconstruct_context falls back to the
        # ego session map.
        if self._adapter_name == "vanilla":
            instance["evidence_session_ids"] = []
            instance["context_tier"] = "full_ego"

        # Omniscient backend: inject ALL corpus session IDs so the model sees
        # every dialogue, not just the ego agent's. Used for ego vs omniscient
        # ablation (TODO #12/#23).
        if self._adapter_name == "omniscient":
            all_sids = list(self._corpus_sessions.keys()) if self._corpus_sessions else []
            instance["evidence_session_ids"] = all_sids
            instance["context_tier"] = "omniscient"

        # Compute effective context budget
        max_tokens = int(self.cfg.max_tokens)
        context_length = int(self.cfg.context_length) if self.cfg.context_length > 0 else 16384
        _CHARS_PER_TOKEN = 3.0
        _overhead_tokens = 300
        _remaining = context_length - max_tokens - _overhead_tokens
        _budget_tokens = max(4096, _remaining)
        _hard_cap = context_length - _overhead_tokens
        _budget_tokens = min(_budget_tokens, _hard_cap)
        max_context_chars = int(_budget_tokens * _CHARS_PER_TOKEN)
        hard_cap_chars = int(_hard_cap * _CHARS_PER_TOKEN)

        # Full-ego dimensions get 4x context
        dim = instance.get("dimension", "")
        tier = instance.get("context_tier", "single_session")
        if self._adapter_name == "vanilla":
            # Vanilla: cap at 8192 tokens regardless of context_length.
            # Gemma-2 has a hard 8192 max_position_embeddings — leave headroom
            # for chat-template tokens (~20) + max generation (400) = ~420.
            # 7700 input + 420 = 8120, leaving 72 token safety buffer.
            vanilla_token_cap = 7700 if _is_gemma2_model(self.cfg.model) else 8192
            eff_ctx = int(vanilla_token_cap * _CHARS_PER_TOKEN)
        elif tier == "full_ego" or dim == "d11_exception":
            eff_ctx = max_context_chars * 4
            if hard_cap_chars > 0:
                eff_ctx = min(eff_ctx, hard_cap_chars)
        else:
            eff_ctx = max_context_chars

        system_p, user_p = build_prompt(
            instance,
            self._corpus_sessions,
            self._ego_session_map,
            max_context_chars=eff_ctx,
            d6_arm=getattr(self.cfg, "d6_arm", "A"),
            owner_contacts=((qa.metadata or {}).get("owner_contacts")
                            if getattr(self.cfg, "d6_owner_contacts", True) else None),
            show_asker=getattr(self.cfg, "d6_show_asker", True),
        )

        # G3a: inject [access:DENY]/[access:ALLOW] marker for norm-binding test
        if getattr(self.cfg, "d6_inject_access_marker", False):
            _dim = str(instance.get("dimension") or "")
            if _dim == "d4_permission":
                _pexp = str(instance.get("metadata", {}).get("policy_expected") or "")
                if "DENY" in _pexp:
                    _marker = "[access:DENY]"
                elif "ALLOW" in _pexp:
                    _marker = "[access:ALLOW]"
                else:
                    _marker = ""
                if _marker:
                    user_p = f"Access policy: {_marker}\n\n{user_p}"

        return system_p, user_p

    async def _call_llm_text(self, system: str, user: str) -> tuple[str, dict]:
        """Call LLM with plain system+user messages (no JSON prefix caching).

        Same retry/streaming logic as _call_llm_with_retry but simpler message structure.
        """
        assert self._aclient is not None
        max_retries = max(1, int(self.cfg.max_retries))
        base_delay = max(0.1, float(self.cfg.retry_base_delay))
        max_delay = max(base_delay, float(self.cfg.retry_max_delay))

        dim_meta = {}
        # Cap max_tokens for certain dimensions
        effective_max_tokens = int(self.cfg.max_tokens)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if _is_no_system_model(self.cfg.model):
            messages = _merge_system_into_user(messages)

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.perf_counter()
                stream = await self._aclient.chat.completions.create(
                    model=self.cfg.model,
                    temperature=float(self.cfg.temperature),
                    max_tokens=effective_max_tokens,
                    messages=messages,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    stream=True,
                    stream_options={"include_usage": True},
                )

                ttft_ms: Optional[float] = None
                chunks: list[str] = []
                usage_prompt: Optional[int] = None
                usage_completion: Optional[int] = None
                reasoning_chars = 0
                async for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        token = getattr(delta, "content", None) or ""
                        if token and ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                        chunks.append(token)
                        # A server with a reasoning parser moves the thinking
                        # out of `content`; count it so the guard still sees it.
                        reasoning_chars += len(getattr(delta, "reasoning_content", None) or "")
                    if getattr(chunk, "usage", None) is not None:
                        usage_prompt = getattr(chunk.usage, "prompt_tokens", None)
                        usage_completion = getattr(chunk.usage, "completion_tokens", None)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                raw_text = "".join(chunks)

                meta: dict = {
                    "elapsed_ms": round(elapsed_ms, 1),
                    "ttft_ms": ttft_ms,
                }
                if reasoning_chars:
                    meta["reasoning_chars"] = reasoning_chars
                if usage_prompt is not None:
                    meta["prompt_tokens"] = usage_prompt
                if usage_completion is not None:
                    meta["completion_tokens"] = usage_completion

                return raw_text, meta
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    break
                await asyncio.sleep(min(max_delay, base_delay * (2 ** (attempt - 1))))

        raise RuntimeError(f"llm_text_call_failed: {last_err}")

    async def _answer_one_direct(self, *, qa: QAItem, ctx_hits: Optional[List[SearchHit]] = None) -> AnswerRecord:
        """Answer using TEXT_SESSIONS prompt path (vanilla/oracle, and the
        TEXT_SESSIONS adapters whose context is their search hits)."""
        from eval.src.scoring_core import extract_answer

        if not self._aclient:
            return AnswerRecord(
                question_id=qa.question_id,
                question=qa.question,
                answer=qa.answer,
                prediction="UNKNOWN",
                model="no-client",
                raw_response="ERROR: no LLM client configured",
            )

        try:
            system_p, user_p = self._build_text_prompt(qa, ctx_hits)
            _ts_start = int(time.time() * 1000)
            raw_text, llm_meta = await self._call_llm_text(system_p, user_p)
            _ts_end = int(time.time() * 1000)
            _emit_hw_marker(
                self.cfg.hw_marker_file,
                instance_id=qa.question_id,
                phase="answer",
                llm_model=self.cfg.model,
                ts_start_ms=_ts_start,
                ts_end_ms=_ts_end,
                prompt_tokens=llm_meta.get("prompt_tokens"),
                completion_tokens=llm_meta.get("completion_tokens"),
            )
            think_detected, think_truncated = self._note_thinking(raw_text, llm_meta)
            if think_truncated:
                # See _answer_one: no answer outside the reasoning. Empty, not
                # "UNKNOWN" (which the scorer would read as an abstention).
                prediction = ""
                self._log(f"think_truncated qid={qa.question_id}: prediction recorded as empty")
            else:
                prediction = extract_answer(raw_text)
                if not prediction:
                    prediction = "UNKNOWN"
            return AnswerRecord(
                question_id=qa.question_id,
                question=qa.question,
                answer=qa.answer,
                prediction=prediction,
                model=self.cfg.model,
                raw_response=raw_text,
                answer_time_ms=llm_meta.get("elapsed_ms"),
                ttft_ms=llm_meta.get("ttft_ms"),
                prompt_tokens=llm_meta.get("prompt_tokens"),
                completion_tokens=llm_meta.get("completion_tokens"),
                think_detected=think_detected,
                think_truncated=think_truncated,
            )
        except ThinkingEnabledError:
            raise
        except Exception as e:
            # Bug B fix: same as the RAG path. Do NOT use a sentinel
            # ("UNKNOWN") that the judge could mistake for a refusal.
            # Mark prediction=None so the post-process patch sets
            # pre_scored.reason='LLM_ERROR' and the row enters the
            # accuracy denominator with score 0.
            return AnswerRecord(
                question_id=qa.question_id,
                question=qa.question,
                answer=qa.answer,
                prediction=None,
                model="llm-error",
                raw_response=f"TEXT_SESSIONS_ERROR: {type(e).__name__}: {e}",
            )

    def _extract_cache_hit(self, resp: Any) -> tuple[bool, int]:
        try:
            usage = getattr(resp, "usage", None)
            cached = int(getattr(usage, "prompt_tokens_details", None).cached_tokens)  # type: ignore[attr-defined]
            return cached > 0, cached
        except Exception:
            return False, 0

    async def _call_llm_with_retry(
        self,
        system_prompt: str,
        user_payload: Mapping[str, object],
        *,
        context_prefix: Optional[str] = None,
    ) -> tuple[str, dict]:
        """Call the LLM with retries (async, streaming for TTFT).

        Returns:
            (response_text, meta) where meta contains ``prompt_tokens``,
            ``completion_tokens``, ``elapsed_ms``, and ``ttft_ms``.
        """
        assert self._aclient is not None
        max_retries = max(1, int(self.cfg.max_retries))
        base_delay = max(0.1, float(self.cfg.retry_base_delay))
        max_delay = max(base_delay, float(self.cfg.retry_max_delay))

        # Build messages with shared context as a cacheable prefix.
        # Structure: system → [context msg → ack] → question msg
        # sglang radix cache will cache the common prefix (system + context)
        # across all requests that share the same corpus.
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if context_prefix:
            messages.append({"role": "user", "content": context_prefix})
            messages.append({"role": "assistant", "content": "I have read and memorized the full conversation context above. Ready for questions."})
        messages.append({"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)})
        if _is_no_system_model(self.cfg.model):
            messages = _merge_system_into_user(messages)

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.perf_counter()
                stream = await self._aclient.chat.completions.create(
                    model=self.cfg.model,
                    temperature=float(self.cfg.temperature),
                    max_tokens=int(self.cfg.max_tokens),
                    messages=messages,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    stream=True,
                    stream_options={"include_usage": True},
                )

                # Consume stream, measuring TTFT on first content chunk
                ttft_ms: Optional[float] = None
                chunks: list[str] = []
                usage_prompt: Optional[int] = None
                usage_completion: Optional[int] = None
                reasoning_chars = 0
                async for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        token = getattr(delta, "content", None) or ""
                        if token and ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                        chunks.append(token)
                        # A server with a reasoning parser moves the thinking
                        # out of `content`; count it so the guard still sees it.
                        reasoning_chars += len(getattr(delta, "reasoning_content", None) or "")
                    # Final chunk carries usage stats
                    if getattr(chunk, "usage", None) is not None:
                        usage_prompt = getattr(chunk.usage, "prompt_tokens", None)
                        usage_completion = getattr(chunk.usage, "completion_tokens", None)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                raw_text = "".join(chunks)

                meta: dict = {
                    "elapsed_ms": round(elapsed_ms, 1),
                    "ttft_ms": ttft_ms,
                }
                if reasoning_chars:
                    meta["reasoning_chars"] = reasoning_chars
                if usage_prompt is not None:
                    meta["prompt_tokens"] = usage_prompt
                if usage_completion is not None:
                    meta["completion_tokens"] = usage_completion

                raw = self._coerce_content(raw_text)
                if raw:
                    return raw, meta
                return raw_text, meta
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    break
                await asyncio.sleep(min(max_delay, base_delay * (2 ** (attempt - 1))))

        raise RuntimeError(f"llm_call_failed: {last_err}")

    async def _complete_json(
        self,
        system_prompt: str,
        payload: Mapping[str, object],
        *,
        context_prefix: Optional[str] = None,
    ) -> tuple[str, dict]:
        return await self._call_llm_with_retry(system_prompt, payload, context_prefix=context_prefix)

    def _ask_openclaw_rag(self, qa: QAItem, ctx_hits: Sequence[SearchHit]) -> str:
        prompt = self._build_openclaw_prompt(qa, ctx_hits)

        if self.cfg.openclaw_stateless:
            sid = f"{self._openclaw_session_id}-{qa.question_id}"
        else:
            sid = self._openclaw_session_id

        return self._run_openclaw(sid, prompt)

    def _ask_openclaw_session(self, qa: QAItem) -> str:
        namespace = self._active_namespace or "default"
        sid = self._session_id(namespace)
        prompt = self._build_openclaw_session_prompt(qa)
        return self._run_openclaw(sid, prompt)

    def _run_openclaw(self, session_id: str, prompt: str) -> str:
        self._ensure_openclaw_session_ready(session_id)
        self._touched_sessions.add(session_id)

        self._openclaw_call_count += 1
        call_no = self._openclaw_call_count
        prompt_head = (prompt or "").splitlines()[0][:120]
        self._log(f"openclaw_call#{call_no} session={session_id} prompt_head={prompt_head!r}")

        cmd = [
            self.cfg.openclaw_bin,
            "agent",
            "--local",
            "--session-id",
            session_id,
            "--message",
            prompt,
            "--json",
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1, int(self.cfg.openclaw_timeout_seconds)),
        )
        self._append_openclaw_console_log(
            session_id=session_id,
            prompt=prompt,
            stdout_text=(proc.stdout or ""),
            stderr_text=(proc.stderr or ""),
            phase="turn",
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            self._log(f"openclaw_call_failed session={session_id} code={proc.returncode} err={err[:240]!r}")
            raise RuntimeError(err or f"openclaw exit={proc.returncode}")

        data = json.loads(proc.stdout or "{}")
        payloads = data.get("payloads") or []
        if not payloads:
            self._log_full_io(session_id=session_id, prompt=prompt, raw_output=(proc.stdout or "").strip())
            return ""

        text = str(payloads[0].get("text") or "").strip()
        self._log_full_io(session_id=session_id, prompt=prompt, raw_output=text)
        if not text:
            return ""

        # Defensive: if agent returns JSON-like answer, extract answer field.
        try:
            obj = json.loads(text)
            if isinstance(obj, Mapping) and obj.get("answer") is not None:
                out = str(obj.get("answer")).strip()
                self._log(f"openclaw_call_done session={session_id} answer={out[:120]!r}")
                return out
        except Exception:
            pass

        out = text.splitlines()[0].strip()
        self._log(f"openclaw_call_done session={session_id} answer={out[:120]!r}")
        return out

    def _ensure_openclaw_session_ready(self, session_id: str) -> None:
        """Strict mode: no per-session pre-prompts of any kind."""
        return

    def _run_openclaw_bootstrap(self, *, session_id: str, prompt: str, err_tag: str, log_full_io: bool = True) -> None:
        cmd = [
            self.cfg.openclaw_bin,
            "agent",
            "--local",
            "--session-id",
            session_id,
            "--message",
            prompt,
            "--json",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1, int(self.cfg.openclaw_timeout_seconds)),
        )
        self._append_openclaw_console_log(
            session_id=session_id,
            prompt=prompt,
            stdout_text=(proc.stdout or ""),
            stderr_text=(proc.stderr or ""),
            phase=f"bootstrap:{err_tag}",
        )
        out_text = self._extract_openclaw_output_text(proc.stdout or "", proc.stderr or "")
        if log_full_io:
            self._log_full_io(session_id=session_id, prompt=prompt, raw_output=out_text)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(err or f"openclaw {err_tag} exit={proc.returncode}")

    def _extract_openclaw_output_text(self, stdout_text: str, stderr_text: str = "") -> str:
        s = (stdout_text or "").strip()
        if not s:
            return (stderr_text or "").strip()
        try:
            obj = json.loads(s)
            payloads = obj.get("payloads") if isinstance(obj, Mapping) else None
            if isinstance(payloads, list) and payloads:
                parts: List[str] = []
                for p in payloads:
                    if isinstance(p, Mapping):
                        t = p.get("text")
                        if t is not None and str(t).strip():
                            parts.append(str(t).strip())
                if parts:
                    return "\n".join(parts)
        except Exception:
            pass
        return s

    def cleanup_openclaw_sessions(self) -> Dict[str, object]:
        if self._backend not in {"openclaw-rag", "openclaw-session"}:
            return {"enabled": False, "reason": "backend_not_openclaw"}
        if not bool(self.cfg.openclaw_cleanup_after_run):
            return {"enabled": False, "reason": "cleanup_disabled"}

        sessions = sorted(self._touched_sessions)
        if not sessions:
            return {"enabled": True, "sessions": 0, "reset_ok": 0, "reset_fail": 0, "details": []}

        details: List[Dict[str, object]] = []
        ok = 0
        fail = 0
        reset_model = (self._openclaw_model or "").strip() or None

        for sid in sessions:
            prompt = f"/new {reset_model} Reply RESET_OK only." if reset_model else "/new Reply RESET_OK only."
            cmd = [
                self.cfg.openclaw_bin,
                "agent",
                "--local",
                "--session-id",
                sid,
                "--message",
                prompt,
                "--json",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(1, int(self.cfg.openclaw_timeout_seconds)),
            )
            if proc.returncode == 0:
                ok += 1
                details.append({"session_id": sid, "status": "ok"})
            else:
                fail += 1
                err = (proc.stderr or proc.stdout or "").strip()
                details.append({"session_id": sid, "status": "error", "error": err})

        rep = {
            "enabled": True,
            "sessions": len(sessions),
            "reset_ok": ok,
            "reset_fail": fail,
            "details": details,
        }
        self._log(f"cleanup sessions={len(sessions)} ok={ok} fail={fail}")
        return rep

    def _session_id(self, namespace: str) -> str:
        ns = (namespace or "default").replace(" ", "_")
        return f"{self._openclaw_session_id}-{ns}"

    def _message_line(self, m: MessageEntry) -> str:
        who = f"u_{m.user_id}" if m.user_id is not None else "u_unknown"
        ts = m.deliver_ts or m.occur_ts
        profile = self._format_user_profile_hint(m.user_id)
        profile_suffix = f" {profile}" if profile else ""
        return f"[{ts}][{m.thread_id}][{who}{profile_suffix}] {m.text}"

    def _build_ingest_prompt(self, *, chunk: Sequence[str], chunk_idx: int, total_chunks: int) -> str:
        body = "\n".join(chunk)
        if self._ingest_mode == "chat":
            tail = (
                "Reply naturally as the assistant in this conversation. "
                "Keep it concise (1-3 sentences), grounded to the message, and do not mention benchmark instructions."
            )
        else:
            tail = "Reply ACK only."
        return (
            f"[INGEST] chunk {chunk_idx}/{total_chunks}\n"
            f"{body}\n\n"
            f"{tail}"
        )

    def _build_openclaw_session_prompt(self, qa: QAItem) -> str:
        options_block = ""
        if qa.options:
            options_block = "\nOptions:\n" + "\n".join(str(x) for x in qa.options)

        return (
            "[QUESTION]\n"
            "Mode: session-memory\n"
            f"Question type: {qa.question_type}\n"
            f"Question: {qa.question}"
            f"{options_block}\n\n"
            "Final answer only:"
        )

    def _build_openclaw_prompt(self, qa: QAItem, hits: Sequence[SearchHit]) -> str:
        ctx_lines: List[str] = []
        for i, h in enumerate(hits, start=1):
            who = f"u_{h.user_id}" if h.user_id is not None else "u_unknown"
            ts = h.occur_ts or ""
            profile = self._format_user_profile_hint(h.user_id)
            profile_suffix = f" {profile}" if profile else ""
            ctx_lines.append(f"[{i}] ({ts}) ({h.thread_id}) ({who}{profile_suffix}) {h.text}")

        context_block = "\n".join(ctx_lines) if ctx_lines else "(no retrieved context)"
        options_block = ""
        if qa.options:
            options_block = "\nOptions:\n" + "\n".join(str(x) for x in qa.options)

        return (
            "[QUESTION]\n"
            "Mode: retrieved-context\n"
            f"Question type: {qa.question_type}\n"
            f"Question: {qa.question}"
            f"{options_block}\n\n"
            f"Context:\n{context_block}\n\n"
            "Final answer only:"
        )

    def _build_llm_system_prompt(self, dim: str = "", owner: str = "") -> str:
        soul = (self._policy_soul_text or "").strip()

        parts = list(ANSWERING_SYSTEM_PARTS)
        if soul:
            parts.append("Policy playbook:\n" + soul)

        if getattr(self.cfg, "d6_arm", "A") == "B" and dim == "d4_permission":
            from MASim.prompts import D6_ARM_B_SYSTEM
            owner_str = owner or "the user"
            parts.append("Policy:\n" + D6_ARM_B_SYSTEM.format(owner=owner_str))
        elif dim == "d4_permission" and owner:
            # the same role as the TEXT_SESSIONS prompt of Vanilla / Oracle
            from eval.simple_eval import d6_assistant_role
            parts.append(d6_assistant_role(owner))
        return "\n".join(parts)

    def _load_policy_soul_text(self) -> str:
        try:
            root = Path(__file__).resolve().parents[2]
            raw_path = (self.cfg.openclaw_soul_path or "SOUL.md").strip()
            p = Path(raw_path)
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                return ""
            text = p.read_text(encoding="utf-8").strip()
            return text[:12000]
        except Exception:
            return ""

    def _load_user_context(self) -> tuple[Dict[int, Dict[str, Any]], str]:
        raw_path = (self.cfg.openclaw_user_context_path or "").strip()
        if not raw_path:
            return {}, ""

        try:
            root = Path(__file__).resolve().parents[2]
            p = Path(raw_path)
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                self._log(f"user_context missing path={p}")
                return {}, ""

            obj = json.loads(p.read_text(encoding="utf-8"))
            users_raw: Any
            if isinstance(obj, Mapping) and isinstance(obj.get("users"), Mapping):
                users_raw = obj.get("users")
            elif isinstance(obj, Mapping):
                users_raw = obj
            elif isinstance(obj, list):
                users_raw = {str(x.get("user_id")): x for x in obj if isinstance(x, Mapping) and x.get("user_id") is not None}
            else:
                users_raw = {}

            by_id: Dict[int, Dict[str, Any]] = {}
            preview_lines: List[str] = []
            for k, v in (users_raw.items() if isinstance(users_raw, Mapping) else []):
                try:
                    uid = int(k)
                except Exception:
                    continue
                rec = dict(v) if isinstance(v, Mapping) else {}
                if not rec:
                    continue

                compact = {
                    "user_id": uid,
                    "name": rec.get("name"),
                    "role": rec.get("role"),
                    "role_labels": rec.get("role_labels"),
                    "team": rec.get("team"),
                    "groups": rec.get("groups"),
                    "permissions": rec.get("permissions") or rec.get("permission") or rec.get("access"),
                }
                compact = {kk: vv for kk, vv in compact.items() if vv not in (None, "", [], {})}
                by_id[uid] = compact
                preview_lines.append(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))

            max_chars = max(1000, int(getattr(self.cfg, "openclaw_user_context_max_chars", 12000) or 12000))
            blob = "\n".join(preview_lines)
            if len(blob) > max_chars:
                blob = blob[: max_chars - 64].rstrip() + "\n... [truncated]"

            return by_id, blob
        except Exception as e:
            self._log(f"user_context load failed: {type(e).__name__}: {e}")
            return {}, ""

    def _format_user_profile_hint(self, user_id: Optional[int]) -> str:
        if user_id is None:
            return ""
        rec = self._user_context_by_id.get(int(user_id)) if self._user_context_by_id else None
        if not rec:
            return ""

        role = rec.get("role")
        team = rec.get("team")
        groups = rec.get("groups")
        perms = rec.get("permissions")

        bits: List[str] = []
        if role:
            bits.append(f"role={role}")
        if team:
            bits.append(f"team={team}")
        if isinstance(groups, list) and groups:
            bits.append(f"groups={len(groups)}")
        if perms:
            bits.append("permissions=present")
        if not bits:
            return ""
        return "{" + ", ".join(bits) + "}"

    def _coerce_content(self, content: object) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for x in content:
                if isinstance(x, Mapping):
                    t = x.get("text")
                    if t is not None:
                        parts.append(str(t))
                elif x is not None:
                    parts.append(str(x))
            return "\n".join(parts).strip()
        if content is None:
            return ""
        return str(content).strip()

    def _safe_answer_fallback(self, qa: QAItem, raw_text: str) -> str:
        s = _strip_thinking((raw_text or "").strip())
        if not s:
            return "UNKNOWN"
        # prefer first non-empty line
        for line in s.splitlines():
            t = line.strip()
            if t:
                return self._normalize_answer(t, qa.question_type)
        return "UNKNOWN"

    def _extract_answer(self, raw_text: str, qtype: str) -> str:
        s = _strip_thinking((raw_text or "").strip())
        if not s:
            return "UNKNOWN"

        candidates = [s]

        # markdown fenced json extraction
        if "```" in s:
            parts = s.split("```")
            for p in parts:
                t = p.strip()
                if not t:
                    continue
                if t.lower().startswith("json"):
                    t = t[4:].strip()
                candidates.append(t)

        for cand in candidates:
            # strict JSON path
            try:
                obj = json.loads(cand)
                if isinstance(obj, Mapping):
                    ans = obj.get("answer")
                    if ans is not None:
                        out = str(ans).strip()
                        if out:
                            return self._normalize_answer(out, qtype)
            except Exception:
                pass

            # loose JSON extraction
            i = cand.find("{")
            j = cand.rfind("}")
            if i >= 0 and j > i:
                sub = cand[i : j + 1]
                try:
                    obj = json.loads(sub)
                    if isinstance(obj, Mapping):
                        ans = obj.get("answer")
                        if ans is not None:
                            out = str(ans).strip()
                            if out:
                                return self._normalize_answer(out, qtype)
                except Exception:
                    pass

        return self._normalize_answer(s.splitlines()[0].strip(), qtype)

    def _normalize_answer(self, text: str, qtype: str) -> str:
        out = text.strip()
        if qtype == "multiple_choice" and out:
            c = out[0].upper()
            if c in {"A", "B", "C", "D"}:
                return c
        return out

    def _heuristic_answer(self, qa: QAItem, hits: Sequence[SearchHit]) -> str:
        if qa.question_type == "multiple_choice":
            # can't reliably solve offline; choose UNKNOWN to avoid false confidence
            return "UNKNOWN"
        if hits:
            return hits[0].text[:200]
        return "UNKNOWN"
