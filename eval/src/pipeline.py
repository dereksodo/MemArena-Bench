from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from .adapters.base import PromptStyle, RetrievalAdapter
from .answering import AnswerConfig, AnswerEngine
from .io import load_messages_jsonl, load_qa, write_json
from .scoring import evaluate_answers, evaluate_answers_hybrid
from .types import AnswerRecord, QAItem, SearchRecord


class _ProgressBar:
    def __init__(self, *, total: int, enabled: bool, prefix: str) -> None:
        self.total = max(0, int(total))
        self.enabled = bool(enabled) and self.total > 0
        self.prefix = str(prefix)
        self.done = 0
        self._last_render_done = -1
        self._render_every = max(1, self.total // 200)

        if self.enabled:
            self._render(force=True)

    def update(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self.done = min(self.total, self.done + max(0, int(n)))
        if self.done == self.total or (self.done - self._last_render_done) >= self._render_every:
            self._render(force=False)

    def close(self) -> None:
        if not self.enabled:
            return
        self.done = self.total
        self._render(force=True)
        print(file=sys.stderr, flush=True)

    def _render(self, *, force: bool) -> None:
        if (not force) and self.done == self._last_render_done:
            return
        width = 24
        ratio = (self.done / self.total) if self.total else 1.0
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        pct = ratio * 100.0
        print(
            f"\r[{self.prefix}] |{bar}| {self.done}/{self.total} {pct:5.1f}%",
            end="",
            file=sys.stderr,
            flush=True,
        )
        self._last_render_done = self.done


class EvalPipeline:
    def __init__(
        self,
        *,
        adapter: RetrievalAdapter,
        answer_cfg: AnswerConfig,
        output_dir: Path,
        masim_run_dir: Optional[Path] = None,
    ) -> None:
        self.adapter = adapter
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.masim_run_dir = masim_run_dir

        # Load corpus data for TEXT_SESSIONS prompts and ego-scoped search.
        prompt_style = adapter.prompt_style
        corpus_sessions: Optional[Dict[str, Any]] = None
        ego_session_map: Optional[Dict[str, Any]] = None
        if masim_run_dir is not None:
            from eval.simple_eval import load_ego_session_map
            ego_session_map = load_ego_session_map(masim_run_dir)
            if prompt_style == PromptStyle.TEXT_SESSIONS:
                from eval.simple_eval import load_corpus_sessions
                corpus_sessions = load_corpus_sessions(masim_run_dir)
        # Pass ego_session_map to adapter for ego-scoped BM25 indexing
        self._ego_session_map: Optional[Dict[str, Any]] = ego_session_map
        if ego_session_map is not None and hasattr(adapter, 'set_ego_session_map'):
            adapter.set_ego_session_map(ego_session_map)
        if corpus_sessions is not None and hasattr(adapter, 'set_corpus_sessions'):
            adapter.set_corpus_sessions(corpus_sessions)

        # Derive adapter name for backend-specific prompt logic
        adapter_name = type(adapter).__name__.lower()
        if "vanilla" in adapter_name:
            adapter_name = "vanilla"
        elif "omniscient" in adapter_name:
            adapter_name = "omniscient"
        elif "oraclegated" in adapter_name:
            adapter_name = "oracle_gated"
        elif "oracle" in adapter_name:
            adapter_name = "oracle"

        self.answer_engine = AnswerEngine(
            answer_cfg,
            prompt_style=prompt_style,
            corpus_sessions=corpus_sessions,
            ego_session_map=ego_session_map,
            adapter_name=adapter_name,
            context_from_hits=bool(getattr(adapter, "text_context_from_hits", False)),
        )

    def _log(self, msg: str) -> None:
        if not bool(self.answer_engine.cfg.verbose):
            return
        ts = datetime.now(timezone.utc).isoformat()
        print(f"[eval.pipeline][{ts}] {msg}", file=sys.stderr, flush=True)

    async def run(
        self,
        *,
        stages: Sequence[str],
        namespace: str,
        messages_path: Optional[Path],
        qa_path: Optional[Path],
        top_k: int,
        qa_limit: Optional[int] = None,
        message_limit: Optional[int] = None,
        resume: bool = True,
        force: bool = False,
        judge_enabled: bool = False,
        judge_runs: int = 3,
        eval_concurrency: Optional[int] = None,
    ) -> Dict[str, object]:
        stages = list(stages)
        result: Dict[str, object] = {}
        interleaved = bool(self.answer_engine.cfg.interleaved_mode)
        show_progress = bool(self.answer_engine.cfg.show_progress)
        log_every = max(1, int(getattr(self.answer_engine.cfg, "log_every", 50)))

        self._log(
            "start "
            f"namespace={namespace} stages={stages} backend={self.answer_engine.cfg.backend} "
            f"interleaved={interleaved} top_k={top_k} resume={resume} force={force} "
            f"judge_enabled={judge_enabled} judge_runs={judge_runs}"
        )

        messages = None
        if any(s in stages for s in ["add", "search", "answer"]) and messages_path is not None:
            messages = load_messages_jsonl(messages_path, limit=message_limit)
            self._log(f"loaded_messages count={len(messages)} from={messages_path}")

        qas: Optional[List[QAItem]] = None
        if any(s in stages for s in ["search", "answer", "evaluate"]):
            if qa_path is None:
                raise ValueError("qa_path is required for search/answer/evaluate")
            qas = load_qa(qa_path, limit=qa_limit)
            self._log(f"loaded_qas count={len(qas)} from={qa_path}")

        run_meta = self._load_run_meta(namespace)
        run_meta.setdefault("namespace", namespace)
        run_meta.setdefault("stages", {})

        if "add" in stages:
            self._log("stage=add start")
            add_path = self._stage_file("add_report", namespace)
            add_checksum = self._stage_checksum(
                {
                    "stage": "add",
                    "namespace": namespace,
                    "messages_path": str(messages_path) if messages_path else None,
                    "message_limit": message_limit,
                    "interleaved": interleaved,
                    "backend": self.answer_engine.cfg.backend,
                    "messages_count": len(messages or []),
                }
            )
            if resume and (not force) and add_path.exists() and self._maybe_skip_stage("add", add_checksum, run_meta):
                add_report = json.loads(add_path.read_text(encoding="utf-8"))
                result["add"] = {**add_report, "resumed": True}
                self._log(f"stage=add resumed path={add_path}")
            else:
                if messages is None:
                    raise ValueError("messages_path is required for add stage")
                if interleaved:
                    add_report = {
                        "mode": "interleaved",
                        "skipped_bulk_add": True,
                        "messages": len(messages),
                        "note": "bulk add skipped to avoid future-context leakage; ingestion happens during answer stage",
                    }
                else:
                    add_report = await self.adapter.add(namespace=namespace, messages=messages)
                    ingest_report = await self.answer_engine.ingest_messages(namespace=namespace, messages=messages)
                    if ingest_report.get("enabled"):
                        add_report = {
                            **add_report,
                            "answer_backend_ingest": ingest_report,
                        }
                write_json(add_path, add_report)
                result["add"] = add_report
                self._log(f"stage=add done messages={add_report.get('messages', 0)} path={add_path}")
                run_meta["stages"]["add"] = {"path": str(add_path), "updated_at": self._now_iso(), "checksum": add_checksum}
                self._save_run_meta(namespace, run_meta)

        search_records: Optional[List[SearchRecord]] = None
        if "search" in stages:
            self._log("stage=search start")
            search_path = self._stage_file("search_results", namespace)
            search_checksum = self._stage_checksum(
                {
                    "stage": "search",
                    "namespace": namespace,
                    "qa_path": str(qa_path) if qa_path else None,
                    "qa_limit": qa_limit,
                    "top_k": top_k,
                    "interleaved": interleaved,
                    "qids": [q.question_id for q in (qas or [])],
                }
            )
            if resume and (not force) and search_path.exists() and self._maybe_skip_stage("search", search_checksum, run_meta):
                raw = json.loads(search_path.read_text(encoding="utf-8"))
                if interleaved:
                    search_records = []
                else:
                    search_records = [
                        SearchRecord(
                            question_id=str(x.get("question_id", "")),
                            query=str(x.get("query", "")),
                            hits=[self._dict_to_hit(h) for h in (x.get("hits") or [])],
                        )
                        for x in raw
                    ]
                result["search"] = {
                    "questions": len(raw) if isinstance(raw, list) else 0,
                    "path": str(search_path),
                    "resumed": True,
                }
                self._log(f"stage=search resumed path={search_path}")
            else:
                if qas is None:
                    raise ValueError("qas missing for search stage")
                if interleaved:
                    search_records = []
                    payload: List[Dict[str, object]] = []
                    write_json(search_path, payload)
                    result["search"] = {
                        "questions": 0,
                        "path": str(search_path),
                        "mode": "interleaved",
                        "note": "search is executed online at each anchored question during answer stage",
                    }
                    self._log(f"stage=search done mode=interleaved path={search_path}")
                else:
                    search_records_out: List[Optional[SearchRecord]] = [None] * len(qas)
                    search_progress = _ProgressBar(total=len(qas), enabled=show_progress, prefix="eval_search")
                    search_sem = asyncio.Semaphore(max(1, int(self.answer_engine.cfg.concurrency)))

                    async def _search_one(idx: int, qa_item: QAItem) -> None:
                        ego_id = self._qa_ego_id(qa_item)
                        async with search_sem:
                            t_s = time.perf_counter()
                            # Pass ego_id for ego-scoped BM25 (search only
                            # within sessions the ego participated in).
                            search_kwargs = {
                                "namespace": namespace,
                                "query": qa_item.question,
                                "top_k": top_k,
                                "instance_id": qa_item.question_id,
                            }
                            if ego_id:
                                search_kwargs["ego_id"] = ego_id
                            query_ts = self._qa_query_timestamp(qa_item)
                            if query_ts is not None:
                                search_kwargs["query_timestamp"] = query_ts
                            exclude_thread_ids = self._qa_exclude_thread_ids(qa_item)
                            if exclude_thread_ids:
                                search_kwargs["exclude_thread_ids"] = exclude_thread_ids
                            # Pass evidence_session_ids for oracle_with_distractors
                            _ev_sids = qa_item.metadata.get("evidence_session_ids")
                            if _ev_sids:
                                search_kwargs["evidence_session_ids"] = _ev_sids
                            # Pass policy_expected for oracle_gated (ALLOW/DENY filter)
                            _policy = qa_item.metadata.get("policy_expected")
                            if _policy:
                                search_kwargs["policy_expected"] = _policy
                            # No TypeError fallback: an adapter that cannot take
                            # the ego / time / exclusion scope must say so in its
                            # signature, not have it silently dropped here.
                            hits = await self.adapter.search(**search_kwargs)
                            s_ms = round((time.perf_counter() - t_s) * 1000.0, 1)
                        search_records_out[idx] = SearchRecord(question_id=qa_item.question_id, query=qa_item.question, hits=hits, search_time_ms=s_ms)
                        search_progress.update(1)

                    t_search_start = time.perf_counter()
                    await asyncio.gather(*[_search_one(i, qa) for i, qa in enumerate(qas)])
                    t_search_elapsed = time.perf_counter() - t_search_start
                    search_records = [r for r in search_records_out if r is not None]
                    search_progress.close()
                    payload = [
                        {
                            "question_id": r.question_id,
                            "query": r.query,
                            "hits": [asdict(h) for h in r.hits],
                        }
                        for r in search_records
                    ]
                    write_json(search_path, payload)
                    result["search"] = {"questions": len(payload), "path": str(search_path), "elapsed_s": round(t_search_elapsed, 1)}
                    self._log(f"stage=search done questions={len(payload)} elapsed={t_search_elapsed:.1f}s path={search_path}")
                run_meta["stages"]["search"] = {"path": str(search_path), "updated_at": self._now_iso(), "checksum": search_checksum}
                self._save_run_meta(namespace, run_meta)

        answer_records: Optional[List[AnswerRecord]] = None
        if "answer" in stages:
            self._log("stage=answer start")
            answer_path = self._stage_file("answer_results", namespace)
            answer_light_path = self._stage_file("answer_results_light", namespace)
            answer_checksum = self._stage_checksum(
                {
                    "stage": "answer",
                    "namespace": namespace,
                    "qa_path": str(qa_path) if qa_path else None,
                    "qa_limit": qa_limit,
                    "messages_path": str(messages_path) if messages_path else None,
                    "message_limit": message_limit,
                    "interleaved": interleaved,
                    "backend": self.answer_engine.cfg.backend,
                    "top_k": top_k,
                    "qids": [q.question_id for q in (qas or [])],
                }
            )
            if resume and (not force) and answer_path.exists() and self._maybe_skip_stage("answer", answer_checksum, run_meta):
                raw = json.loads(answer_path.read_text(encoding="utf-8"))
                answer_records = [AnswerRecord(**x) for x in raw]
                result["answer"] = {
                    "questions": len(answer_records),
                    "path": str(answer_path),
                    "resumed": True,
                    "light_path": str(answer_light_path) if answer_light_path.exists() else None,
                }
                self._log(f"stage=answer resumed path={answer_path}")
            else:
                if qas is None:
                    raise ValueError("qas missing for answer stage")

                io_trace: List[Dict[str, Any]] = []
                turn_idx = 0

            if answer_records is None and interleaved:
                if messages is None:
                    raise ValueError("messages_path is required for interleaved answer stage")

                qas_by_anchor: Dict[str, List[QAItem]] = {}
                no_anchor_qas: List[QAItem] = []
                for qa in qas:
                    anchor_msg_id = self._qa_anchor_msg_id(qa)
                    if anchor_msg_id:
                        qas_by_anchor.setdefault(anchor_msg_id, []).append(qa)
                    else:
                        no_anchor_qas.append(qa)

                answer_records = []
                streamed_messages = 0
                asked_questions = 0
                msg_progress = _ProgressBar(total=len(messages), enabled=show_progress, prefix="eval_stream")
                search_progress = _ProgressBar(
                    total=len(qas),
                    enabled=(show_progress and self.answer_engine.needs_search_context()),
                    prefix="eval_search",
                )
                qa_progress = _ProgressBar(total=len(qas), enabled=show_progress, prefix="eval_answer")

                for m in messages:
                    streamed_messages += 1
                    ingest_reply: Optional[str] = None
                    ingest_meta: Dict[str, object] = {"enabled": False, "reason": "no_stream_ingest"}

                    if self.answer_engine.needs_search_context():
                        await self.adapter.add(namespace=namespace, messages=[m])

                    if self.answer_engine.supports_stream_ingest():
                        ingest_meta = await self.answer_engine.ingest_message(namespace=namespace, message=m)
                        ingest_reply = str(ingest_meta.get("assistant_reply") or "").strip() or None

                    turn_idx += 1
                    io_trace.append(
                        {
                            "turn_index": turn_idx,
                            "ts": self._now_iso(),
                            "turn_type": "input",
                            "msg_id": m.msg_id,
                            "occur_ts": m.occur_ts,
                            "deliver_ts": m.deliver_ts,
                            "thread_id": m.thread_id,
                            "user_id": m.user_id,
                            "input_text": m.text,
                            "assistant_reply": ingest_reply,
                            "backend_ingest": ingest_meta,
                        }
                    )

                    msg_progress.update(1)
                    if streamed_messages == 1 or streamed_messages % log_every == 0:
                        self._log(
                            f"stream msg={streamed_messages}/{len(messages)} msg_id={m.msg_id} "
                            f"pending_q={len(qas_by_anchor.get(m.msg_id, []))}"
                        )

                    pending = qas_by_anchor.get(m.msg_id, [])
                    for qa in pending:
                        s_ms = None
                        if self.answer_engine.needs_search_context():
                            t_s = time.perf_counter()
                            search_kwargs = {
                                "namespace": namespace,
                                "query": qa.question,
                                "top_k": top_k,
                                "instance_id": qa.question_id,
                            }
                            ego_id = self._qa_ego_id(qa)
                            if ego_id:
                                search_kwargs["ego_id"] = ego_id
                            query_ts = self._qa_query_timestamp(qa)
                            if query_ts is not None:
                                search_kwargs["query_timestamp"] = query_ts
                            exclude_thread_ids = self._qa_exclude_thread_ids(qa)
                            if exclude_thread_ids:
                                search_kwargs["exclude_thread_ids"] = exclude_thread_ids
                            hits = await self.adapter.search(**search_kwargs)
                            s_ms = round((time.perf_counter() - t_s) * 1000.0, 1)
                            search_progress.update(1)
                        else:
                            hits = []
                        rec = await self.answer_engine.answer_one(qa=qa, ctx_hits=hits)
                        rec.search_time_ms = s_ms
                        answer_records.append(rec)
                        asked_questions += 1
                        qa_progress.update(1)
                        self._log(
                            f"ask qid={qa.question_id} anchor={self._qa_anchor_msg_id(qa) or '-'} "
                            f"hits={len(hits)} pred={(rec.prediction or '')[:80]!r}"
                        )

                        turn_idx += 1
                        io_trace.append(
                            {
                                "turn_index": turn_idx,
                                "ts": self._now_iso(),
                                "turn_type": "question",
                                "question_id": qa.question_id,
                                "anchor_msg_id": self._qa_anchor_msg_id(qa),
                                "question_type": qa.question_type,
                                "question": qa.question,
                                "gold_answer": qa.answer,
                                "expected_answer_mode": qa.metadata.get("expected_answer_mode"),
                                "policy_expected": qa.metadata.get("policy_expected"),
                                "retrieval_hits": [self._hit_to_dict(h) for h in hits],
                                "assistant_reply": rec.prediction,
                                "assistant_raw": rec.raw_response,
                                "assistant_model": rec.model,
                            }
                        )

                # Fallback: ask unanchored or unmatched-anchor questions at end of stream.
                answered_ids = {a.question_id for a in answer_records}
                tail_qas = [qa for qa in qas if qa.question_id not in answered_ids] + [qa for qa in no_anchor_qas if qa.question_id not in answered_ids]
                dedup_tail: List[QAItem] = []
                seen_tail = set()
                for qa in tail_qas:
                    if qa.question_id in seen_tail:
                        continue
                    seen_tail.add(qa.question_id)
                    dedup_tail.append(qa)

                for qa in dedup_tail:
                    s_ms = None
                    if self.answer_engine.needs_search_context():
                        t_s = time.perf_counter()
                        search_kwargs = {
                            "namespace": namespace,
                            "query": qa.question,
                            "top_k": top_k,
                            "instance_id": qa.question_id,
                        }
                        ego_id = self._qa_ego_id(qa)
                        if ego_id:
                            search_kwargs["ego_id"] = ego_id
                        query_ts = self._qa_query_timestamp(qa)
                        if query_ts is not None:
                            search_kwargs["query_timestamp"] = query_ts
                        exclude_thread_ids = self._qa_exclude_thread_ids(qa)
                        if exclude_thread_ids:
                            search_kwargs["exclude_thread_ids"] = exclude_thread_ids
                        hits = await self.adapter.search(**search_kwargs)
                        s_ms = round((time.perf_counter() - t_s) * 1000.0, 1)
                        search_progress.update(1)
                    else:
                        hits = []
                    rec = await self.answer_engine.answer_one(qa=qa, ctx_hits=hits)
                    rec.search_time_ms = s_ms
                    answer_records.append(rec)
                    asked_questions += 1
                    qa_progress.update(1)
                    self._log(
                        f"ask_tail qid={qa.question_id} anchor={self._qa_anchor_msg_id(qa) or '-'} "
                        f"hits={len(hits)} pred={(rec.prediction or '')[:80]!r}"
                    )

                    turn_idx += 1
                    io_trace.append(
                        {
                            "turn_index": turn_idx,
                            "ts": self._now_iso(),
                            "turn_type": "question",
                            "question_id": qa.question_id,
                            "anchor_msg_id": self._qa_anchor_msg_id(qa),
                            "question_type": qa.question_type,
                            "question": qa.question,
                            "gold_answer": qa.answer,
                            "expected_answer_mode": qa.metadata.get("expected_answer_mode"),
                            "policy_expected": qa.metadata.get("policy_expected"),
                            "retrieval_hits": [self._hit_to_dict(h) for h in hits],
                            "assistant_reply": rec.prediction,
                            "assistant_raw": rec.raw_response,
                            "assistant_model": rec.model,
                            "tail_question": True,
                        }
                    )

                msg_progress.close()
                search_progress.close()
                qa_progress.close()

                answer_path = self._stage_file("answer_results", namespace)
                write_json(answer_path, [asdict(x) for x in answer_records])
                io_path = self._stage_file("input_output", namespace)
                io_payload = {
                    "namespace": namespace,
                    "mode": "interleaved",
                    "turns": io_trace,
                }
                write_json(io_path, io_payload)
                write_json(self.output_dir / "input_output.json", io_payload)

                result["answer"] = {
                    "questions": len(answer_records),
                    "path": str(answer_path),
                    "mode": "interleaved",
                    "streamed_messages": streamed_messages,
                    "asked_questions": asked_questions,
                    "input_output_path": str(io_path),
                }
                self._log(
                    f"stage=answer done mode=interleaved streamed={streamed_messages} asked={asked_questions} "
                    f"path={answer_path}"
                )
            elif answer_records is None:
                if search_records is None:
                    if self.answer_engine.needs_search_context():
                        search_path = self._stage_file("search_results", namespace)
                        if not search_path.exists():
                            raise ValueError("search results missing; run search stage first")
                        raw = json.loads(search_path.read_text(encoding="utf-8"))
                        search_records = [
                            SearchRecord(
                                question_id=str(x.get("question_id", "")),
                                query=str(x.get("query", "")),
                                hits=[
                                    self._dict_to_hit(h) for h in (x.get("hits") or [])
                                ],
                            )
                            for x in raw
                        ]
                    else:
                        search_records = [
                            SearchRecord(question_id=qa.question_id, query=qa.question, hits=[])
                            for qa in qas
                        ]

                ctx_by_qid = {r.question_id: r.hits for r in search_records}
                search_time_by_qid = {r.question_id: r.search_time_ms for r in search_records if r.search_time_ms is not None}
                contexts = [ctx_by_qid.get(qa.question_id, []) for qa in qas]

                # Two-phase timing support: split queries per dimension
                timing_frac = float(getattr(self.answer_engine.cfg, "timing_fraction", 0.0) or 0.0)
                timing_conc = max(1, int(getattr(self.answer_engine.cfg, "timing_concurrency", 4) or 4))
                bulk_conc = max(1, int(self.answer_engine.cfg.concurrency))

                if timing_frac > 0:
                    import random as _random
                    _random.seed(42)
                    from collections import defaultdict as _ddict
                    by_dim_idx: dict = _ddict(list)
                    for i, qa in enumerate(qas):
                        dim = (qa.metadata or {}).get("dimension") or (qa.metadata or {}).get("task_family") or ""
                        by_dim_idx[dim].append(i)
                    timing_set: set = set()
                    for dim, idxs in by_dim_idx.items():
                        n_t = max(1, int(len(idxs) * timing_frac))
                        # Take the LAST n_t instances per dimension (deterministic,
                        # same IDs across all trials — no random sampling).
                        timing_set.update(idxs[-n_t:])
                    timing_indices = sorted(timing_set)
                    bulk_indices = [i for i in range(len(qas)) if i not in timing_set]
                    self._log(f"two-phase: {len(timing_indices)} timing (conc={timing_conc}), "
                              f"{len(bulk_indices)} bulk (conc={bulk_conc})")
                else:
                    timing_indices = []
                    bulk_indices = list(range(len(qas)))

                qa_progress = _ProgressBar(total=len(qas), enabled=show_progress, prefix="eval_answer")
                answer_records_out: list = [None] * len(qas)

                # Phase 1: timing sample at low concurrency
                if timing_indices:
                    timing_qas = [qas[i] for i in timing_indices]
                    timing_ctxs = [contexts[i] for i in timing_indices]
                    old_conc = self.answer_engine.cfg.concurrency
                    self.answer_engine.cfg.concurrency = timing_conc
                    timing_results = await self.answer_engine.answer_batch(
                        qas=timing_qas, contexts=timing_ctxs, progress_cb=qa_progress.update,
                    )
                    self.answer_engine.cfg.concurrency = old_conc
                    for j, idx in enumerate(timing_indices):
                        rec = timing_results[j]
                        rec.timing_reliable = True
                        qid = qas[idx].question_id
                        if qid in search_time_by_qid:
                            rec.search_time_ms = search_time_by_qid[qid]
                        answer_records_out[idx] = rec

                # Phase 2: bulk at high concurrency
                timing_only = bool(getattr(self.answer_engine.cfg, "timing_only", False))
                if bulk_indices and not timing_only:
                    bulk_qas = [qas[i] for i in bulk_indices]
                    bulk_ctxs = [contexts[i] for i in bulk_indices]
                    bulk_results = await self.answer_engine.answer_batch(
                        qas=bulk_qas, contexts=bulk_ctxs, progress_cb=qa_progress.update,
                    )
                    for j, idx in enumerate(bulk_indices):
                        rec = bulk_results[j]
                        rec.timing_reliable = (timing_frac == 0)
                        qid = qas[idx].question_id
                        if qid in search_time_by_qid:
                            rec.search_time_ms = search_time_by_qid[qid]
                        answer_records_out[idx] = rec

                answer_records = [r for r in answer_records_out if r is not None]
                qa_progress.close()
                self._log(f"answer_batch done questions={len(answer_records)}")

                if messages is not None:
                    for m in messages:
                        turn_idx += 1
                        io_trace.append(
                            {
                                "turn_index": turn_idx,
                                "ts": self._now_iso(),
                                "turn_type": "input",
                                "msg_id": m.msg_id,
                                "occur_ts": m.occur_ts,
                                "deliver_ts": m.deliver_ts,
                                "thread_id": m.thread_id,
                                "user_id": m.user_id,
                                "input_text": m.text,
                                "assistant_reply": None,
                                "backend_ingest": {"mode": "bulk_or_non_interleaved"},
                            }
                        )

                rec_by_qid = {r.question_id: r for r in answer_records}
                for qa, hits in zip(qas, contexts):
                    rec = rec_by_qid.get(qa.question_id)
                    if rec is None:
                        continue
                    turn_idx += 1
                    io_trace.append(
                        {
                            "turn_index": turn_idx,
                            "ts": self._now_iso(),
                            "turn_type": "question",
                            "question_id": qa.question_id,
                            "anchor_msg_id": self._qa_anchor_msg_id(qa),
                            "question_type": qa.question_type,
                            "question": qa.question,
                            "gold_answer": qa.answer,
                            "expected_answer_mode": qa.metadata.get("expected_answer_mode"),
                            "policy_expected": qa.metadata.get("policy_expected"),
                            "retrieval_hits": [self._hit_to_dict(h) for h in hits],
                            "assistant_reply": rec.prediction,
                            "assistant_raw": rec.raw_response,
                            "assistant_model": rec.model,
                        }
                    )
                    if turn_idx == 1 or turn_idx % log_every == 0:
                        self._log(
                            f"ask qid={qa.question_id} hits={len(hits)} pred={(rec.prediction or '')[:80]!r}"
                        )

                answer_path = self._stage_file("answer_results", namespace)
                write_json(answer_path, [asdict(x) for x in answer_records])
                io_path = self._stage_file("input_output", namespace)
                io_payload = {
                    "namespace": namespace,
                    "mode": "non_interleaved",
                    "turns": io_trace,
                }
                write_json(io_path, io_payload)
                write_json(self.output_dir / "input_output.json", io_payload)
                result["answer"] = {
                    "questions": len(answer_records),
                    "path": str(answer_path),
                    "input_output_path": str(io_path),
                }
                self._log(f"stage=answer done mode=non_interleaved questions={len(answer_records)} path={answer_path}")

        if "answer" in stages and answer_records is not None and not bool((result.get("answer") or {}).get("resumed")):
            answer_light_path = self._stage_file("answer_results_light", namespace)
            light_payload = [
                {
                    "question_id": r.question_id,
                    "prediction": r.prediction,
                    "model": r.model,
                    "raw_response": r.raw_response,
                }
                for r in answer_records
            ]
            write_json(answer_light_path, light_payload)
            thinking = self.answer_engine.thinking_report()
            if isinstance(result.get("answer"), dict):
                result["answer"]["light_path"] = str(answer_light_path)
                result["answer"]["thinking"] = thinking
            run_meta["stages"]["answer"] = {
                "path": str(self._stage_file("answer_results", namespace)),
                "light_path": str(answer_light_path),
                "updated_at": self._now_iso(),
                "checksum": answer_checksum,
                "thinking": thinking,
            }
            self._save_run_meta(namespace, run_meta)
            # Final guard, after the answers are on disk: warns (or raises,
            # with fail_on_thinking) if the server ignored enable_thinking=False.
            self.answer_engine.check_thinking(final=True)

        if "evaluate" in stages:
            self._log("stage=evaluate start")
            eval_path = self._stage_file("evaluation_results", namespace)
            evaluate_checksum = self._stage_checksum(
                {
                    "stage": "evaluate",
                    "namespace": namespace,
                    "qa_path": str(qa_path) if qa_path else None,
                    "qa_limit": qa_limit,
                    "judge_enabled": judge_enabled,
                    "judge_runs": judge_runs,
                    "answer_path": str(self._stage_file("answer_results", namespace)),
                    "qids": [q.question_id for q in (qas or [])],
                }
            )
            if resume and (not force) and eval_path.exists() and self._maybe_skip_stage("evaluate", evaluate_checksum, run_meta):
                eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
                result["evaluate"] = {
                    **(eval_payload.get("summary") or {}),
                    "path": str(eval_path),
                    "resumed": True,
                }
                self._log(f"stage=evaluate resumed path={eval_path}")
            else:
                if qas is None:
                    raise ValueError("qas missing for evaluate stage")
                if answer_records is None:
                    answer_path = self._stage_file("answer_results", namespace)
                    if not answer_path.exists():
                        raise ValueError("answer results missing; run answer stage first")
                    raw = json.loads(answer_path.read_text(encoding="utf-8"))
                    answer_records = [AnswerRecord(**x) for x in raw]

                # Use a dedicated judge client when --judge-model is specified,
                # otherwise fall back to the answering engine's client.
                _cfg = self.answer_engine.cfg
                if bool(getattr(_cfg, "test_mode", False)):
                    _judge_client = None
                    _judge_model = ""
                elif _cfg.judge_model:
                    from openai import OpenAI as _OpenAI
                    _j_endpoint = _cfg.judge_endpoint or _cfg.endpoint
                    _j_api_key = _cfg.judge_api_key or _cfg.api_key or "EMPTY"
                    _judge_client = _OpenAI(
                        api_key=_j_api_key,
                        base_url=_j_endpoint,
                        timeout=_cfg.timeout_seconds,
                    )
                    _judge_model = _cfg.judge_model
                else:
                    _judge_client = getattr(self.answer_engine, "_client", None)
                    _judge_model = _cfg.model if _judge_client else ""

                # Secondary judge client (cross-judge validation, e.g. GPT-4o-mini)
                _secondary_judge_client = None
                _secondary_judge_model = ""
                if (not bool(getattr(_cfg, "test_mode", False))) and _cfg.secondary_judge_model:
                    from openai import OpenAI as _OpenAI2
                    _sj_endpoint = _cfg.secondary_judge_endpoint or "https://api.openai.com/v1"
                    _sj_api_key = _cfg.secondary_judge_api_key or ""
                    if not _sj_api_key:
                        self._log("WARNING: secondary judge configured but no API key provided")
                    else:
                        _secondary_judge_client = _OpenAI2(
                            api_key=_sj_api_key,
                            base_url=_sj_endpoint,
                            timeout=_cfg.timeout_seconds,
                        )
                        _secondary_judge_model = _cfg.secondary_judge_model
                        self._log(f"secondary judge: {_secondary_judge_model} @ {_sj_endpoint}")

                # Load corpus sessions for evidence-grounded scoring
                _corpus_sessions = None
                if self.masim_run_dir is not None:
                    from .masim_loader import load_corpus_sessions_dict
                    _corpus_sessions = load_corpus_sessions_dict(self.masim_run_dir)
                    self._log(f"loaded corpus_sessions for evidence scoring: {len(_corpus_sessions)} sessions")

                _eval_conc = eval_concurrency if eval_concurrency is not None else int(self.answer_engine.cfg.concurrency)
                _eval_conc = max(1, _eval_conc)
                eval_progress = _ProgressBar(total=len(qas), enabled=show_progress, prefix="eval_judge")
                if judge_enabled:
                    eval_payload = await evaluate_answers_hybrid(
                        qas=qas,
                        answers=answer_records,
                        judge_enabled=judge_enabled,
                        judge_runs=judge_runs,
                        judge_client=_judge_client,
                        judge_model=_judge_model,
                        corpus_sessions=_corpus_sessions,
                        concurrency=_eval_conc,
                        progress_cb=eval_progress.update,
                        secondary_judge_client=_secondary_judge_client,
                        secondary_judge_model=_secondary_judge_model,
                    )
                else:
                    eval_payload = await evaluate_answers(
                        qas=qas,
                        answers=answer_records,
                        judge_client=_judge_client,
                        judge_model=_judge_model,
                        corpus_sessions=_corpus_sessions,
                        concurrency=_eval_conc,
                        progress_cb=eval_progress.update,
                        secondary_judge_client=_secondary_judge_client,
                        secondary_judge_model=_secondary_judge_model,
                    )
                eval_progress.close()
                # Judge provenance in the file itself (as scripts/llmjudge.py does).
                eval_payload.setdefault("summary", {}).update({
                    "judge_model": _judge_model or None,
                    "secondary_judge_model": _secondary_judge_model or None,
                })
                write_json(eval_path, eval_payload)
                result["evaluate"] = {
                    **eval_payload["summary"],
                    "path": str(eval_path),
                }
                self._log(
                    f"stage=evaluate done accuracy={eval_payload.get('summary', {}).get('accuracy')} "
                    f"policy_accuracy={eval_payload.get('summary', {}).get('policy_accuracy')} path={eval_path}"
                )
                run_meta["stages"]["evaluate"] = {"path": str(eval_path), "updated_at": self._now_iso(), "checksum": evaluate_checksum}
                self._save_run_meta(namespace, run_meta)

        cleanup_report = self.answer_engine.cleanup_openclaw_sessions()
        if cleanup_report:
            cleanup_path = self._stage_file("openclaw_cleanup", namespace)
            write_json(cleanup_path, cleanup_report)
            result["openclaw_cleanup"] = {
                **cleanup_report,
                "path": str(cleanup_path),
            }
            self._log(f"cleanup report path={cleanup_path} data={cleanup_report}")

        await self.adapter.close()
        self._log("done")
        return result

    def _run_meta_path(self, namespace: str) -> Path:
        return self.output_dir / f"run_meta_{namespace}.json"

    def _load_run_meta(self, namespace: str) -> Dict[str, Any]:
        p = self._run_meta_path(namespace)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_run_meta(self, namespace: str, meta: Dict[str, Any]) -> None:
        write_json(self._run_meta_path(namespace), meta)

    def _stage_checksum(self, payload: Any) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _maybe_skip_stage(self, stage_name: str, checksum: str, meta: Dict[str, Any]) -> bool:
        st = ((meta.get("stages") if isinstance(meta, dict) else {}) or {}).get(stage_name)
        if not isinstance(st, dict):
            return False
        old = st.get("checksum")
        return bool(old and str(old) == str(checksum))

    def _stage_file(self, stage_prefix: str, namespace: str) -> Path:
        return self.output_dir / f"{stage_prefix}_{namespace}.json"

    @staticmethod
    def _qa_anchor_msg_id(qa: QAItem) -> Optional[str]:
        md = qa.metadata if isinstance(qa.metadata, dict) else {}
        oracle = md.get("oracle") if isinstance(md.get("oracle"), dict) else {}
        anchor = oracle.get("anchor_msg_id")
        if anchor is None:
            anchor = md.get("anchor_msg_id")
        s = str(anchor).strip() if anchor is not None else ""
        return s or None

    @staticmethod
    def _qa_ego_id(qa: QAItem) -> str:
        md = qa.metadata if isinstance(qa.metadata, dict) else {}
        for key in (
            "ego_agent_id",
            "query_agent",
            "instance_query_agent",
            "answerer_agent_id",
            "instance_answerer_agent_id",
            "target_agent",
            "instance_target_agent",
        ):
            value = md.get(key)
            s = str(value or "").strip()
            if s and s not in {"__assistant__", "assistant"}:
                return s
        return ""

    @staticmethod
    def _qa_query_timestamp(qa: QAItem) -> Optional[object]:
        md = qa.metadata if isinstance(qa.metadata, dict) else {}
        for key in ("query_timestamp", "instance_query_timestamp", "anchor_timestamp", "instance_anchor_timestamp"):
            value = md.get(key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _qa_exclude_thread_ids(qa: QAItem) -> List[str]:
        md = qa.metadata if isinstance(qa.metadata, dict) else {}
        raw = md.get("exclude_thread_ids") or md.get("instance_exclude_thread_ids") or []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(x) for x in raw if str(x)]
        return []

    @staticmethod
    def _hit_to_dict(h) -> Dict[str, object]:
        return {
            "msg_id": h.msg_id,
            "score": h.score,
            "text": h.text,
            "occur_ts": h.occur_ts,
            "thread_id": h.thread_id,
            "user_id": h.user_id,
        }

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _dict_to_hit(d: Dict[str, object]):
        from .types import SearchHit

        return SearchHit(
            msg_id=str(d.get("msg_id", "")),
            score=float(d.get("score", 0.0)),
            text=str(d.get("text", "")),
            occur_ts=str(d.get("occur_ts", "")),
            thread_id=str(d.get("thread_id", "")),
            user_id=(int(d["user_id"]) if d.get("user_id") is not None else None),
        )
