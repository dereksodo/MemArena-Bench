"""Shared text cleanup helpers for LLM outputs."""

from __future__ import annotations

import re


_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_OPEN_RE = re.compile(r"<think(?:ing)?\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```\s*$", re.DOTALL)
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:rephrased\s+question|paraphrased\s+question|rewritten\s+question|question)\s*:\s*",
    re.IGNORECASE,
)


def strip_think(text: object) -> str:
    """Remove Qwen3-style reasoning from LLM text; keep what follows it.

    * closed ``<think>...</think>`` blocks are removed;
    * an orphan ``</think>`` (the chat template opened the block, so the output
      starts inside it) drops everything up to it;
    * an unclosed ``<think>`` (generation hit ``max_tokens`` while still
      thinking) drops everything from it to the end, so the reasoning (or a
      bare ``<think>``) is never taken as the answer.
    """
    s = _THINK_BLOCK_RE.sub("", str(text or ""))
    closes = list(_THINK_CLOSE_RE.finditer(s))
    if closes:
        s = s[closes[-1].end():]
    m = _THINK_OPEN_RE.search(s)
    if m:
        s = s[: m.start()]
    return s.strip()


def think_status(text: object) -> tuple[bool, bool]:
    """Return ``(has_think, truncated_in_think)`` for raw LLM output.

    ``truncated_in_think``: the output carries reasoning and nothing is left
    outside it, i.e. the token budget ran out while (or right after) thinking,
    typically an unclosed ``<think>``. There is no answer to score.
    """
    s = str(text or "")
    has_think = bool(_THINK_OPEN_RE.search(s) or _THINK_CLOSE_RE.search(s))
    return has_think, has_think and not strip_think(s)


def clean_llm_text(text: object) -> str:
    """Remove reasoning wrappers and common response scaffolding from LLM text.

    Qwen3-style reasoning models often return ``<think>...</think>`` before the
    actual answer (``strip_think`` also drops an unclosed block). This helper
    is intentionally plain-text oriented: it keeps the final user-visible
    content and removes only wrappers/prefixes that are not part of the
    benchmark question or model answer.
    """
    cleaned = strip_think(str(text or "").strip())

    match = _FENCE_RE.match(cleaned)
    if match:
        cleaned = strip_think(match.group(1).strip())

    cleaned = _LEADING_LABEL_RE.sub("", cleaned).strip()

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()

    return cleaned
