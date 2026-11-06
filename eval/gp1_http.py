"""Logs GP1's Groq requests by wrapping requests.post inside the running process (§9.4), for gp1_answer.py and
gp1_truncation.py.

GP1 throws the response away after reading the text, and baseline/ may not change for this (D12), so usage,
finish_reason, latency and status can only be taken here. GP1's own logic, retries and limits stay as they are. The
wrapper adds one thing: it waits when the minute's tokens are nearly gone. GP1 has no limiter, and a 429 caused by the
pace of an experiment would say nothing about the system (§9.4: requests are spaced). Ollama calls pass through.
"""
import time
from typing import Any

import requests

GROQ_HOST = "api.groq.com"
DAILY_RETRY_AFTER = 60.0  # as in the agent's client: a longer retry-after is the daily quota, not the minute (§9.5)
TOKENS_NEEDED = 2000  # a GP1 request's prompt plus its max_tokens, at most about this


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RequestLog:
    def __init__(self, post: Any = None, sleep: Any = time.sleep, clock: Any = time.monotonic) -> None:
        self._post = post or requests.post
        self._sleep = sleep
        self._clock = clock
        self.calls: list[dict[str, Any]] = []
        self.daily_quota = False
        self.limits: dict[str, str] = {}
        self._free_tokens: float | None = None
        self._free_at = 0.0
        self._limit = 8000.0

    def install(self) -> None:
        requests.post = self.post

    def _pace(self) -> float:
        """Seconds waited for the tokens of the next request, from the last x-ratelimit headers."""
        if self._free_tokens is None:
            return 0.0
        free = self._free_tokens + (self._clock() - self._free_at) * self._limit / 60
        if free >= TOKENS_NEEDED:
            return 0.0
        wait = (TOKENS_NEEDED - free) * 60 / self._limit
        self._sleep(wait)
        return wait

    def post(self, url: str, **options: Any) -> Any:
        if GROQ_HOST not in url:
            return self._post(url, **options)
        waited = self._pace()
        started = self._clock()
        try:
            response = self._post(url, **options)
        except Exception as exc:
            self.calls.append({"http_status": None, "error": type(exc).__name__, "waited_s": round(waited, 2),
                               "latency_ms": int((self._clock() - started) * 1000)})
            raise
        self.calls.append(self._record(response, waited, int((self._clock() - started) * 1000)))
        return response

    def _record(self, response: Any, waited: float, latency_ms: int) -> dict[str, Any]:
        headers = response.headers
        self.limits = {k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit")}
        remaining, limit = _number(headers.get("x-ratelimit-remaining-tokens")), _number(headers.get("x-ratelimit-limit-tokens"))
        if remaining is not None:
            self._free_tokens, self._free_at = remaining, self._clock()
        if limit:
            self._limit = limit
        call: dict[str, Any] = {"http_status": response.status_code, "waited_s": round(waited, 2), "latency_ms": latency_ms}
        if response.status_code != 200:
            call["retry_after"] = _number(headers.get("retry-after"))
            call["error"] = response.text[:300]  # the body only: headers can carry cookies
            if response.status_code == 429 and (call["retry_after"] or 0) > DAILY_RETRY_AFTER:
                self.daily_quota = True
            return call
        try:
            body = response.json()
            choice = body["choices"][0]
            usage = body.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            call["error"] = "unreadable body"
            return call
        call.update(
            finish_reason=choice.get("finish_reason"),
            content_length=len(choice["message"].get("content") or ""),
            prompt_tokens=usage.get("prompt_tokens") or 0,
            completion_tokens=usage.get("completion_tokens") or 0,
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        )
        return call


def failure(calls: list[dict[str, Any]]) -> str | None:
    """Why the last request of a message failed, if it did. GP1 hides this: on an API error `_handle_query` answers with
    the raw summary, which holds correct numbers, and a failed request must count as wrong (§9.3)."""
    if not calls:
        return None
    last = calls[-1]
    if last.get("http_status") != 200:
        return f"api: {last.get('http_status')} {last.get('error', '')}".strip()
    if "error" in last:
        return f"api: {last['error']}"
    if not last.get("content_length"):
        return f"empty reply (finish_reason={last.get('finish_reason')})"
    return None
