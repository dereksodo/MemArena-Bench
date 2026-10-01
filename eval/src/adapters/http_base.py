from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import aiohttp

from .base import RetrievalAdapter, normalize_score
from ..types import SearchHit


class HttpRetrievalAdapter(RetrievalAdapter):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_s: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip() or None
        self.timeout_s = max(1, int(timeout_s))
        self.max_retries = max(1, int(max_retries))
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # No session-level timeout — use per-request timeouts only
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    def _backoff(self, attempt: int) -> float:
        return min(8.0, 0.5 * (2 ** max(0, attempt - 1)))

    async def _request_with_retry(
        self,
        method: str,
        path_or_url: str,
        *,
        json_body: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout_s: Optional[int] = None,
    ) -> Dict[str, Any]:
        session = await self._get_session()
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}/{path_or_url.lstrip('/')}"
        req_timeout = aiohttp.ClientTimeout(total=max(1, int(timeout_s or self.timeout_s)))

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with session.request(
                    method.upper(),
                    url,
                    json=json_body,
                    headers=headers,
                    timeout=req_timeout,
                ) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
                    if not text.strip():
                        return {}
                    try:
                        return await resp.json()
                    except Exception:
                        return {"raw": text}
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(self._backoff(attempt))

        raise RuntimeError(f"request_failed: {last_error}")

    def _normalize_hit(self, raw: dict) -> SearchHit:
        return SearchHit(
            msg_id=str(raw.get("msg_id") or raw.get("id") or ""),
            score=normalize_score(float(raw.get("score", 0.0)), min_value=0.0, max_value=1.0),
            text=str(raw.get("text") or raw.get("content") or ""),
            occur_ts=str(raw.get("occur_ts") or raw.get("timestamp") or ""),
            thread_id=str(raw.get("thread_id") or raw.get("group_id") or ""),
            user_id=(int(raw["user_id"]) if raw.get("user_id") is not None else None),
        )

    async def health(self) -> dict:
        t0 = time.time()
        try:
            await self._request_with_retry("GET", "/health", timeout_s=min(self.timeout_s, 10))
            return {
                "ok": True,
                "detail": "reachable",
                "latency_ms": int((time.time() - t0) * 1000),
            }
        except Exception as e:
            return {
                "ok": False,
                "detail": str(e),
                "latency_ms": int((time.time() - t0) * 1000),
            }

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
