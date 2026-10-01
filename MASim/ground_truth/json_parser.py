"""Robust JSON parsing for LLM-generated MASim eval instances."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from memarena.text_cleaning import clean_llm_text


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


def clean_llm_json_text(text: str) -> str:
    cleaned = clean_llm_text(text)

    match = _FENCE_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()

    # Some models put the fence first and the reasoning block inside it.
    return clean_llm_text(cleaned)


def _extract_balanced_json(text: str) -> str | None:
    for start, first in enumerate(text):
        if first not in "[{":
            continue
        stack = ["]" if first == "[" else "}"]
        in_string = False
        escaped = False
        for idx in range(start + 1, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                stack.append("]" if char == "[" else "}")
            elif char in "]}":
                if not stack or char != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    return text[start:idx + 1]
    return None


def parse_llm_json(text: str) -> Any:
    """Parse JSON from an LLM completion with Qwen3 reasoning wrappers."""
    cleaned = clean_llm_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as direct_error:
        snippet = _extract_balanced_json(cleaned)
        if snippet is None:
            raise ValueError(f"LLM response does not contain JSON: {cleaned[:200]!r}") from direct_error
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as snippet_error:
            raise ValueError(f"LLM response contains malformed JSON: {snippet[:200]!r}") from snippet_error


def parse_json_object(text: str) -> Dict[str, Any]:
    data = parse_llm_json(text)
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object, got {type(data).__name__}")
    return data


def parse_json_array(text: str) -> List[Dict[str, Any]]:
    data = parse_llm_json(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise TypeError(f"expected JSON array/object, got {type(data).__name__}")
