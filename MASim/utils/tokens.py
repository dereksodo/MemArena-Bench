"""Token counting utilities using tiktoken."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def _get_encoding(model: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(model)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count the number of tokens in a string."""
    enc = _get_encoding(encoding)
    return len(enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, encoding: str = "cl100k_base") -> str:
    """Truncate text to fit within a token budget."""
    enc = _get_encoding(encoding)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
