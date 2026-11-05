"""Groq client (§6.3): native tool calling, or a JSON-in-text fallback with the same schemas.

- Retries timeouts, connection errors, 5xx and 429 with exponential backoff and jitter; a 429
  waits for `retry-after` when the header is present. A `retry-after` over a minute is the daily
  quota: it stops at once.
- Never returns None (A1): after the last attempt it raises a typed `LLMError`.
- A limiter per model keeps requests under the tokens-per-minute limit, from a rolling window
  of its own usage and the `x-ratelimit-*` headers, so that 429s are avoided rather than retried.
  Each request reserves its prompt estimate plus the whole of `max_tokens`.
- A reply cut by `max_tokens` is an error (`LLMTruncatedError`), not a short answer.
- One client serves every API request: its shared state is guarded by locks, and a waiting
  request sleeps outside them.
- Every attempt is reported through `on_attempt`; the agent loop writes it to `llm_calls`.
"""
import json
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

GROQ_URL = "https://api.groq.com/openai/v1"
MAX_RETRY_WAIT = 60.0  # a longer retry-after means the daily quota, not the minute
_PRIVATE_HEADERS = {"authorization", "cookie", "set-cookie"}
_UNSET: Any = object()


# ---------- errors ----------

class LLMError(Exception):
    kind = "error"
    retryable = False
    usage: "Usage | None" = None  # what the failed call still cost, when known

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class LLMTimeoutError(LLMError):
    kind = "timeout"
    retryable = True


class LLMServerError(LLMError):
    kind = "server_error"
    retryable = True


class LLMRateLimitError(LLMError):
    kind = "rate_limited"
    retryable = True

    def __init__(self, message: str, http_status: int | None = 429, retry_after: float | None = None) -> None:
        super().__init__(message, http_status)
        self.retry_after = retry_after


class LLMRequestError(LLMError):
    """A 4xx other than 429 (bad request, unknown model, invalid key): not retried."""

    kind = "request_error"


class LLMToolValidationError(LLMRequestError):
    """Groq's 400 `tool_use_failed`: the model's tool call did not match the schema. The loop lets the model fix it."""

    kind = "tool_use_failed"


class LLMTruncatedError(LLMError):
    """finish_reason = length: the reply hit `max_tokens`, so it cannot be trusted as complete. Not retried, because
    the same request would be cut in the same place."""

    kind = "truncated"


class LLMResponseError(LLMError):
    """HTTP 200 with a body that cannot be read."""

    kind = "bad_response"


# ---------- results ----------

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0  # part of prompt_tokens: read from Groq's prompt cache

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens,
                     self.reasoning_tokens + other.reasoning_tokens, self.cached_tokens + other.cached_tokens)

    @classmethod
    def from_api(cls, usage: dict | None) -> "Usage":
        usage = usage or {}
        completion = usage.get("completion_tokens_details") or {}
        prompt = usage.get("prompt_tokens_details") or {}
        return cls(usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0,
                   completion.get("reasoning_tokens") or 0, prompt.get("cached_tokens") or 0)

    def as_dict(self) -> dict:
        return {"prompt": self.prompt_tokens, "completion": self.completion_tokens,
                "reasoning": self.reasoning_tokens, "cached": self.cached_tokens, "total": self.total_tokens}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None  # None when the model's arguments are not a JSON object
    raw_arguments: str
    error: str | None = None


@dataclass
class Attempt:
    model: str
    attempt: int
    status: str  # ok, timeout, server_error, rate_limited, request_error, bad_response, truncated
    http_status: int | None
    latency_ms: int
    usage: Usage | None = None
    error: str | None = None
    waited_ms: int = 0  # time the limiter held the request back
    retry_after: float | None = None  # from a 429


@dataclass
class LLMResponse:
    model: str
    content: str | None
    tool_calls: list[ToolCall]
    usage: Usage
    latency_ms: int
    attempts: int
    assistant_message: dict  # to append to the conversation before the tool results
    finish_reason: str | None = None


# ---------- rate limiting ----------

_DURATION = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def parse_duration(value: str | None) -> float | None:
    """Groq's reset values ('2.25s', '1m26.4s', '7.66ms') or retry-after seconds → seconds."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    match = _DURATION.match(value.strip())
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds, millis = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


class RateLimiter:
    """Keeps one model under its tokens-per-minute limit, with as little waiting as possible.

    - After each response, `x-ratelimit-remaining-tokens` gives the free budget; Groq refills it
      continuously at about limit/60 tokens a second, so a request waits only for the missing part.
    - Before any header is seen, a rolling 60 s window of the limiter's own usage is used.
    - Prompt estimates are calibrated per model: a moving average of actual ÷ estimated tokens, plus a
      small safety margin, because the raw estimate overshot by about 1.7× in step 5a.
    """

    def __init__(self, tokens_per_minute: int = 8000, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 margin: float = 0.1, smoothing: float = 0.3) -> None:
        self.limit = tokens_per_minute
        self.window = window
        self.margin = margin
        self.smoothing = smoothing
        self.ratio = 1.0  # moving average of actual / raw estimate
        self.observations = 0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._events: deque[list] = deque()  # [timestamp, tokens, seconds waited], used until the first headers
        self._budget: float | None = None  # free tokens at _budget_at, from the headers
        self._budget_at = 0.0
        self.waited_total = 0.0

    def calibrate(self, raw_estimate: int) -> int:
        return int(raw_estimate * self.ratio * (1 + self.margin)) + 1

    def observe(self, raw_estimate: int, actual_tokens: int) -> None:
        if raw_estimate <= 0 or actual_tokens <= 0:
            return
        ratio = actual_tokens / raw_estimate
        with self._lock:
            self.ratio = ratio if self.observations == 0 else (1 - self.smoothing) * self.ratio + self.smoothing * ratio
            self.ratio = max(self.ratio, 0.1)
            self.observations += 1

    def _used(self, now: float) -> int:
        while self._events and self._events[0][0] + self.window <= now:
            self._events.popleft()
        return sum(event[1] for event in self._events)

    def _available(self, now: float) -> float:
        return min(self.limit, self._budget + (now - self._budget_at) * self.limit / self.window)

    def _delay(self, now: float, estimate: int) -> float:
        """Seconds until `estimate` tokens fit, or 0 after reserving them from the budget."""
        needed = min(estimate, self.limit)
        if self._budget is not None:
            available = self._available(now)
            if needed <= available:
                self._budget, self._budget_at = available - needed, now
                return 0.0
            return max((needed - available) * self.window / self.limit, 0.05)
        if self._used(now) + estimate <= self.limit or not self._events:
            return 0.0
        return max(self._events[0][0] + self.window - now, 0.05)

    def acquire(self, estimate: int) -> list:
        """Wait until `estimate` tokens fit, then reserve them. Returns the reservation for `settle`;
        its third item is the time this request waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                delay = self._delay(now, estimate)
                if not delay:
                    event = [now, estimate, waited]
                    self._events.append(event)
                    return event
                self.waited_total += delay
            self._sleep(delay)
            waited += delay

    @staticmethod
    def settle(event: list, tokens: int) -> None:
        """Replace the reservation with the tokens actually used."""
        event[1] = tokens

    def update(self, headers: httpx.Headers | dict) -> None:
        limit = headers.get("x-ratelimit-limit-tokens")
        remaining = headers.get("x-ratelimit-remaining-tokens")
        with self._lock:
            if limit and limit.isdigit():
                self.limit = int(limit)
            if remaining is not None and remaining.lstrip("-").isdigit():
                self._budget, self._budget_at = float(remaining), self._clock()


def estimate_prompt_tokens(messages: Any, tools: Any = None) -> int:
    """Rough size: about 3 characters per token."""
    return len(json.dumps([messages, tools], ensure_ascii=False)) // 3


# ---------- JSON fallback (TOOL_MODE=json) ----------

JSON_MODE_INSTRUCTIONS = (
    "Tools are described below as JSON schemas. Answer with one JSON object only: either "
    '{"tool_calls": [{"name": "<tool>", "arguments": {...}}]} to call tools, or {"reply": "<text>"} '
    "to answer the user. Tool results come back in messages that start with TOOL RESULT."
)


def _to_json_mode(messages: list[dict], tools: list[dict]) -> list[dict]:
    schemas = json.dumps([t["function"] for t in tools], ensure_ascii=False, separators=(",", ":"))
    names: dict[str, str] = {}
    converted = []
    for message in messages:
        role = message["role"]
        if role == "system":
            converted.append({"role": "system",
                              "content": f"{message['content']}\n\n{JSON_MODE_INSTRUCTIONS}\nTools: {schemas}"})
        elif role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                names[call["id"]] = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"])
                except (TypeError, ValueError):
                    arguments = call["function"]["arguments"]
                calls.append({"name": call["function"]["name"], "arguments": arguments})
            converted.append({"role": "assistant", "content": json.dumps({"tool_calls": calls}, ensure_ascii=False)})
        elif role == "tool":
            name = names.get(message.get("tool_call_id"), "tool")
            converted.append({"role": "user", "content": f"TOOL RESULT {name}: {message['content']}"})
        else:
            converted.append(message)
    return converted


def _parse_arguments(raw: Any) -> tuple[dict | None, str, str | None]:
    if isinstance(raw, dict):
        return raw, json.dumps(raw, ensure_ascii=False), None
    try:
        value = json.loads(raw) if raw else {}
    except (TypeError, ValueError) as exc:
        return None, str(raw), f"arguments are not valid JSON: {exc}"
    if not isinstance(value, dict):
        return None, str(raw), "arguments must be a JSON object"
    return value, raw, None


def _native_message(message: dict) -> tuple[str | None, list[ToolCall], dict]:
    calls = []
    for raw in message.get("tool_calls") or []:
        arguments, text, error = _parse_arguments(raw["function"].get("arguments"))
        calls.append(ToolCall(raw["id"], raw["function"]["name"], arguments, text, error))
    assistant = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        assistant["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"}}
            for c in message["tool_calls"]
        ]
    return message.get("content"), calls, assistant


def _json_mode_message(message: dict) -> tuple[str | None, list[ToolCall], dict]:
    content = message.get("content") or ""
    try:
        data = json.loads(content)
    except ValueError:
        data = None
    if not isinstance(data, dict) or not data.get("tool_calls"):
        reply = data.get("reply") if isinstance(data, dict) and "reply" in data else content
        return reply, [], {"role": "assistant", "content": reply}
    calls, wire = [], []
    for index, raw in enumerate(data["tool_calls"], start=1):
        raw = raw if isinstance(raw, dict) else {}
        arguments, text, error = _parse_arguments(raw.get("arguments", {}))
        call_id = f"call_{index}"
        calls.append(ToolCall(call_id, str(raw.get("name", "")), arguments, text, error))
        wire.append({"id": call_id, "type": "function", "function": {"name": str(raw.get("name", "")), "arguments": text}})
    return None, calls, {"role": "assistant", "content": None, "tool_calls": wire}


# ---------- client ----------

def _error_code(response: httpx.Response) -> str | None:
    try:
        return (response.json().get("error") or {}).get("code")
    except ValueError:
        return None


def _rate_limit_text(response: httpx.Response) -> str:
    """The error and all the response headers, so that the reason of a 429 can be read later."""
    headers = {k: v for k, v in response.headers.items() if k.lower() not in _PRIVATE_HEADERS}
    return f"{_error_text(response)} | headers: {json.dumps(headers, ensure_ascii=False)}"


def _error_text(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        return f"{error.get('code') or response.status_code}: {error.get('message', '')}"[:500]
    except ValueError:
        return response.text[:500]


class GroqClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = GROQ_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_retry_wait: float = MAX_RETRY_WAIT,
        reasoning_effort: str | None = "low",
        max_tokens: int = 800,
        tool_mode: str = "native",
        tokens_per_minute: int = 8000,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if tool_mode not in ("native", "json"):
            raise ValueError(f"unknown tool mode {tool_mode!r}")
        self.max_retries = max_retries
        self.max_retry_wait = max_retry_wait
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.tool_mode = tool_mode
        self.tokens_per_minute = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport,
                                  headers={"Authorization": f"Bearer {api_key}"})
        self._limiters_lock = threading.Lock()
        self.limiters: dict[str, RateLimiter] = {}
        self.last_rate_limits: dict[str, dict[str, str]] = {}

    @classmethod
    def from_settings(cls, settings: Any, **overrides: Any) -> "GroqClient":
        if not settings.groq_key_present:
            raise LLMRequestError("GROQ_API_KEY is not set")
        options = {"timeout": settings.llm_timeout, "reasoning_effort": settings.reasoning_effort,
                   "max_tokens": settings.llm_max_tokens, "tool_mode": settings.tool_mode, **overrides}
        return cls(settings.groq_api_key.get_secret_value(), **options)

    def limiter(self, model: str) -> RateLimiter:
        with self._limiters_lock:
            if model not in self.limiters:
                self.limiters[model] = RateLimiter(self.tokens_per_minute, clock=self._clock, sleep=self._sleep)
            return self.limiters[model]

    def _payload(self, model: str, messages: list[dict], tools: list[dict], temperature: float,
                 reasoning_effort: str | None) -> dict:
        payload: dict[str, Any] = {"model": model, "temperature": temperature, "max_tokens": self.max_tokens}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if self.tool_mode == "json":
            payload["messages"] = _to_json_mode(messages, tools) if tools else messages
            payload["response_format"] = {"type": "json_object"}
        else:
            payload["messages"] = messages
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
        return payload

    def _backoff(self, attempt: int) -> float:
        return 2 ** (attempt - 1) + self._jitter()

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        on_attempt: Callable[[Attempt], None] | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = _UNSET,
    ) -> LLMResponse:
        effort = self.reasoning_effort if reasoning_effort is _UNSET else reasoning_effort
        payload = self._payload(model, messages, tools or [], temperature, effort)
        limiter = self.limiter(model)
        raw_prompt = estimate_prompt_tokens(payload["messages"], payload.get("tools"))
        estimate = limiter.calibrate(raw_prompt) + self.max_tokens

        for attempt in range(1, self.max_retries + 2):
            reservation = limiter.acquire(estimate)
            waited_ms = int(reservation[2] * 1000)
            started = self._clock()
            response: httpx.Response | None = None
            wait: float | None = None
            usage: Usage | None = None
            try:
                response = self._http.post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                error: LLMError = LLMTimeoutError(f"timeout: {exc}")
            except httpx.HTTPError as exc:
                error = LLMServerError(f"connection error: {exc}")
            else:
                limiter.update(response.headers)
                self.last_rate_limits[model] = {k: v for k, v in response.headers.items()
                                                if k.startswith("x-ratelimit") or k == "retry-after"}
                if response.status_code == 200:
                    try:
                        result = self._read(model, response, attempt, int((self._clock() - started) * 1000))
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        error = LLMResponseError(f"unreadable response: {exc}", 200)
                    else:
                        usage = result.usage
                        limiter.settle(reservation, usage.total_tokens)
                        limiter.observe(raw_prompt, usage.prompt_tokens)
                        if result.finish_reason != "length":
                            if on_attempt:
                                on_attempt(Attempt(model, attempt, "ok", 200, result.latency_ms, usage,
                                                   waited_ms=waited_ms))
                            return result
                        error = LLMTruncatedError(f"the reply was cut at max_tokens={self.max_tokens}", 200)
                        error.usage = usage
                elif response.status_code == 429:
                    wait = parse_duration(response.headers.get("retry-after"))
                    error = LLMRateLimitError(_rate_limit_text(response), 429, wait)
                elif response.status_code >= 500:
                    error = LLMServerError(_error_text(response), response.status_code)
                elif _error_code(response) == "tool_use_failed":
                    error = LLMToolValidationError(_error_text(response), response.status_code)
                else:
                    error = LLMRequestError(_error_text(response), response.status_code)

            latency_ms = int((self._clock() - started) * 1000)
            if usage is None:
                limiter.settle(reservation, 0 if response is None or response.status_code == 429 else estimate)
            if on_attempt:
                on_attempt(Attempt(model, attempt, error.kind, error.http_status, latency_ms, usage, str(error),
                                   waited_ms=waited_ms, retry_after=wait))
            if not error.retryable or attempt > self.max_retries:
                raise error
            if wait is not None and wait > self.max_retry_wait:
                raise error
            self._sleep(wait if wait is not None else self._backoff(attempt))
        raise AssertionError("unreachable")

    def _read(self, model: str, response: httpx.Response, attempts: int, latency_ms: int) -> LLMResponse:
        body = response.json()
        choice = body["choices"][0]
        reader = _json_mode_message if self.tool_mode == "json" else _native_message
        content, calls, assistant = reader(choice["message"])
        return LLMResponse(model, content, calls, Usage.from_api(body.get("usage")), latency_ms, attempts, assistant,
                           choice.get("finish_reason"))

    def close(self) -> None:
        self._http.close()
