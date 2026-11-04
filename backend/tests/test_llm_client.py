import json

import httpx
import pytest

from app.llm.client import (
    GroqClient, LLMRateLimitError, LLMRequestError, LLMServerError, LLMTimeoutError, LLMToolValidationError,
    RateLimiter, parse_duration,
)

TOOL_CALL = {"id": "c1", "type": "function",
             "function": {"name": "add_expense", "arguments": '{"amount": 5000, "category": "بنزين ومواصلات"}'}}
USAGE = {"prompt_tokens": 200, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 20}}
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "دفعت 5000 بنزين"}]
TOOLS = [{"type": "function", "function": {"name": "add_expense", "parameters": {"type": "object"}}}]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


def ok(message=None, headers=None):
    message = message or {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
    return httpx.Response(200, json={"choices": [{"message": message}], "usage": USAGE}, headers=headers or {})


def make(responses, clock=None, **options):
    """A client whose transport returns `responses` in order (an exception instance is raised)."""
    clock = clock or FakeClock()
    sent: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = GroqClient("test-key", transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep,
                        jitter=lambda: 0.0, **options)
    return client, sent, clock


def chat(client, attempts=None):
    return client.chat("openai/gpt-oss-20b", MESSAGES, TOOLS, on_attempt=attempts.append if attempts is not None else None)


def test_success_payload_and_parsing():
    client, sent, _ = make([ok()])
    attempts = []
    result = chat(client, attempts)
    body = json.loads(sent[0].content)
    assert body["reasoning_effort"] == "low" and body["temperature"] == 0 and body["tool_choice"] == "auto"
    assert body["max_tokens"] == 1024 and body["tools"] == TOOLS
    assert sent[0].headers["authorization"] == "Bearer test-key"
    call = result.tool_calls[0]
    assert (call.name, call.arguments) == ("add_expense", {"amount": 5000, "category": "بنزين ومواصلات"})
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.reasoning_tokens) == (200, 50, 20)
    assert [(a.status, a.http_status) for a in attempts] == [("ok", 200)]
    assert result.assistant_message["tool_calls"][0]["id"] == "c1"


def test_429_waits_for_retry_after():
    client, sent, clock = make([httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow"}}),
                                ok()])
    attempts = []
    chat(client, attempts)
    assert clock.sleeps == [2.0]
    assert [a.status for a in attempts] == ["rate_limited", "ok"]


def test_5xx_is_retried_with_backoff():
    client, _, clock = make([httpx.Response(503), httpx.Response(502), ok()])
    chat(client)
    assert clock.sleeps == [1.0, 2.0]


def test_final_failure_raises_a_typed_error_never_none():
    client, sent, clock = make([httpx.Response(500)] * 4)
    attempts = []
    with pytest.raises(LLMServerError):
        chat(client, attempts)
    assert len(sent) == 4 and clock.sleeps == [1.0, 2.0, 4.0]
    assert [a.status for a in attempts] == ["server_error"] * 4


def test_timeouts_are_retried_then_raised():
    client, sent, _ = make([httpx.ReadTimeout("slow")] * 4)
    with pytest.raises(LLMTimeoutError):
        chat(client)
    assert len(sent) == 4


def test_400_is_not_retried():
    client, sent, clock = make([httpx.Response(400, json={"error": {"code": "model_not_found", "message": "no"}})])
    with pytest.raises(LLMRequestError, match="model_not_found"):
        chat(client)
    assert len(sent) == 1 and clock.sleeps == []


def test_tool_use_failed_is_classified_and_not_retried():
    body = {"error": {"code": "tool_use_failed", "message": "Tool call validation failed: bad enum"}}
    client, sent, _ = make([httpx.Response(400, json=body)])
    attempts = []
    with pytest.raises(LLMToolValidationError):
        chat(client, attempts)
    assert len(sent) == 1 and attempts[0].status == "tool_use_failed"


def test_long_retry_after_stops_at_once():
    client, sent, clock = make([httpx.Response(429, headers={"retry-after": "3600"})])
    with pytest.raises(LLMRateLimitError):
        chat(client)
    assert len(sent) == 1 and clock.sleeps == []


def test_rate_limit_retries_can_be_disabled():
    client, sent, _ = make([httpx.Response(429, headers={"retry-after": "1"})], max_rate_limit_retries=0)
    with pytest.raises(LLMRateLimitError):
        chat(client)
    assert len(sent) == 1


def test_invalid_tool_arguments_are_reported_not_raised():
    bad = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c9", "type": "function", "function": {"name": "add_expense", "arguments": "{amount: 5"}}]}
    client, _, _ = make([ok(bad)])
    call = chat(client).tool_calls[0]
    assert call.arguments is None and "not valid JSON" in call.error


def test_rate_limit_headers_are_kept_and_respected():
    headers = {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "0",
               "x-ratelimit-reset-tokens": "5s", "x-ratelimit-remaining-requests": "998"}
    client, _, clock = make([ok(headers=headers), ok()])
    chat(client)
    assert client.last_rate_limits["openai/gpt-oss-20b"]["x-ratelimit-remaining-requests"] == "998"
    chat(client)
    assert clock.sleeps == [5.0], "the second request waits for the token window to reset"


def test_limiter_rolling_window():
    clock = FakeClock()
    limiter = RateLimiter(1000, clock=clock, sleep=clock.sleep)
    first = limiter.acquire(600)
    limiter.settle(first, 700)
    clock.now += 10
    limiter.acquire(400)
    assert clock.sleeps == [50.0], "waits until the first request leaves the 60 s window"


def test_json_fallback_mode():
    content = json.dumps({"tool_calls": [{"name": "get_total", "arguments": {"period": "this_month"}}]})
    client, sent, _ = make([ok({"role": "assistant", "content": content})], tool_mode="json")
    history = MESSAGES + [
        {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok":true}'},
    ]
    result = client.chat("openai/gpt-oss-20b", history, TOOLS)
    body = json.loads(sent[0].content)
    assert "tools" not in body and body["response_format"] == {"type": "json_object"}
    assert "Tools:" in body["messages"][0]["content"]
    assert body["messages"][3] == {"role": "user", "content": 'TOOL RESULT add_expense: {"ok":true}'}
    assert (result.tool_calls[0].name, result.tool_calls[0].arguments) == ("get_total", {"period": "this_month"})

    client, _, _ = make([ok({"role": "assistant", "content": '{"reply": "أهلين"}'})], tool_mode="json")
    assert client.chat("m", MESSAGES, TOOLS).content == "أهلين"


@pytest.mark.parametrize("value, seconds", [
    ("2", 2.0), ("2.257s", 2.257), ("1m26.4s", 86.4), ("7.66ms", 0.00766), ("1h2m3s", 3723.0), ("", None), ("x", None),
])
def test_parse_duration(value, seconds):
    assert parse_duration(value) == (pytest.approx(seconds) if seconds is not None else None)
