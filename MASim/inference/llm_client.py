"""Unified OpenAI-compatible LLM client for vLLM/SGLang backends.

Uses thread-local client instances for safe concurrent access
from ThreadPoolExecutor.
"""

from __future__ import annotations

import json as _json
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from MASim.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LLMClientConfig:
    """Configuration for the LLM client."""
    endpoint: str = "http://127.0.0.1:30000/v1"
    api_key: str = "EMPTY"
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout_seconds: int = 60
    max_retries: int = 2
    concurrency: int = 32
    # When True, max_tokens is never sent to the API — the backend generates
    # until EOS or its own server-side limit.  Per-call max_tokens hints are
    # also ignored.  Set this when the model/backend handles length itself.
    unlimited_tokens: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMClientConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LLMClient:
    """Thread-safe OpenAI-compatible LLM client.

    Compatible with vLLM and SGLang serving endpoints.
    Uses thread-local storage for client instances.
    """

    def __init__(self, cfg: LLMClientConfig):
        self.cfg = cfg
        self._thread_local = threading.local()
        self._call_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._lock = threading.Lock()
        self._timing_records: List[Dict] = []
        self._detail_log_path: Optional[Path] = None
        self._detail_log_file = None

    def set_detail_log(self, path: str | Path) -> None:
        """Enable per-call detail logging to a JSONL file.

        Each LLM call will append a record with full system_prompt,
        user_prompt, completion text, timing, and tags.
        """
        self._detail_log_path = Path(path)
        self._detail_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._detail_log_file = open(self._detail_log_path, "a", encoding="utf-8")
        log.info("LLM detail logging enabled → %s", self._detail_log_path)

    def _write_detail(
        self, system_prompt: str, user_prompt: str, completion: str,
        wall_s: float, prompt_tokens: int, completion_tokens: int,
        tags: Optional[Dict[str, Any]],
    ) -> None:
        """Append one detail record (thread-safe)."""
        if self._detail_log_file is None:
            return
        rec: Dict[str, Any] = {
            "ts": _time.time(),
            "wall_s": round(wall_s, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "completion": completion,
        }
        if tags:
            rec.update(tags)
        line = _json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:
            self._detail_log_file.write(line)
            self._detail_log_file.flush()

    def _get_client(self) -> OpenAI:
        """Get or create a thread-local OpenAI client."""
        client = getattr(self._thread_local, "client", None)
        if client is None:
            endpoint = self.cfg.endpoint
            if not endpoint.endswith("/v1"):
                endpoint = endpoint.rstrip("/") + "/v1"
            client = OpenAI(
                base_url=endpoint,
                api_key=self.cfg.api_key,
                timeout=self.cfg.timeout_seconds,
            )
            self._thread_local.client = client
        return client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
        stop: Optional[list] = None,
    ) -> str:
        """Generate a single completion.

        Args:
            system_prompt: System message setting the context.
            user_prompt: User message with the specific request.
            max_tokens: Per-call override; falls back to config value when None.

        Returns:
            Generated text response.
        """
        client = self._get_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        effective_max_tokens = max_tokens if max_tokens is not None else self.cfg.max_tokens

        for attempt in range(self.cfg.max_retries + 1):
            try:
                call_kwargs: Dict[str, Any] = {
                    "model": self.cfg.model,
                    "messages": messages,
                    "temperature": self.cfg.temperature,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
                if not self.cfg.unlimited_tokens:
                    call_kwargs["max_tokens"] = effective_max_tokens
                if stop:
                    call_kwargs["stop"] = stop
                t0 = _time.perf_counter()
                response = client.chat.completions.create(**call_kwargs)
                wall_s = _time.perf_counter() - t0
                with self._lock:
                    self._call_count += 1
                    if response.usage:
                        self._prompt_tokens += response.usage.prompt_tokens or 0
                        self._completion_tokens += response.usage.completion_tokens or 0
                    record: Dict[str, Any] = {
                        "ts": _time.time(),
                        "wall_s": round(wall_s, 4),
                        "prompt_tokens": response.usage.prompt_tokens or 0 if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens or 0 if response.usage else 0,
                    }
                    if tags:
                        record.update(tags)
                    self._timing_records.append(record)
                completion_text = response.choices[0].message.content or ""
                self._write_detail(
                    system_prompt, user_prompt, completion_text,
                    wall_s,
                    response.usage.prompt_tokens or 0 if response.usage else 0,
                    response.usage.completion_tokens or 0 if response.usage else 0,
                    tags,
                )
                return completion_text
            except Exception as e:
                if attempt == self.cfg.max_retries:
                    log.error("LLM call failed after %d retries: %s", self.cfg.max_retries + 1, e)
                    raise
                log.warning("LLM call attempt %d failed: %s", attempt + 1, e)

        return ""  # unreachable

    def generate_batch(self, prompts: List[Dict[str, str]]) -> List[str]:
        """Generate completions for a batch of prompts concurrently.

        Args:
            prompts: List of dicts with 'system' and 'user' keys.
                     Optional 'max_tokens' key overrides the config value per prompt.

        Returns:
            List of generated responses.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [""] * len(prompts)
        batch_size = len(prompts)

        def _gen(idx: int, prompt: Dict[str, str]) -> tuple:
            prompt_tags = dict(prompt.get("tags") or {})
            prompt_tags["batch_size"] = batch_size
            prompt_tags["batch_idx"] = idx
            text = self.generate(
                prompt["system"], prompt["user"],
                max_tokens=prompt.get("max_tokens"),
                tags=prompt_tags,
            )
            return idx, text

        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as ex:
            futures = {ex.submit(_gen, i, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                try:
                    idx, text = fut.result()
                    results[idx] = text
                except Exception as e:
                    idx = futures[fut]
                    log.error("Batch generation failed for prompt %d: %s", idx, e)
                    results[idx] = ""

        return results

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    def get_usage_stats(self) -> Dict[str, int]:
        """Return all usage counters as a dict."""
        return {
            "llm_calls": self._call_count,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
        }

    def write_timing_log(self, path) -> None:
        """Dump per-call timing records to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for rec in self._timing_records:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        # Close detail log if open
        if self._detail_log_file is not None:
            self._detail_log_file.close()
            self._detail_log_file = None
