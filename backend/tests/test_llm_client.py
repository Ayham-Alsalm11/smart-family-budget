"""The Groq client with a mocked transport: retries, the 429 policy, the limiter and token usage (§6.3)."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from app.llm.client import (
    GroqClient, LLMRateLimitError, LLMRequestError, LLMServerError, LLMTimeoutError, LLMToolValidationError,
    LLMTruncatedError, RateLimiter, Usage, estimate_prompt_tokens, parse_duration,
)
from app.settings import Settings

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
    assert body["max_tokens"] == 800 and body["tools"] == TOOLS
    assert sent[0].headers["authorization"] == "Bearer test-key"
    call = result.tool_calls[0]
    assert (call.name, call.arguments) == ("add_expense", {"amount": 5000, "category": "بنزين ومواصلات"})
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.reasoning_tokens) == (200, 50, 20)
    assert [(a.status, a.http_status) for a in attempts] == [("ok", 200)]
    assert result.assistant_message["tool_calls"][0]["id"] == "c1"


def test_cached_tokens_are_read_and_added_up():
    body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {**USAGE, "prompt_tokens_details": {"cached_tokens": 150}}}
    client, _, _ = make([httpx.Response(200, json=body), ok()])
    first, second = chat(client), chat(client)
    assert first.usage.cached_tokens == 150
    assert second.usage.cached_tokens == 0, "zero when Groq does not report it"
    assert (first.usage + first.usage).cached_tokens == 300
    assert first.usage.as_dict()["cached"] == 150 and first.usage.total_tokens == 250, "part of the prompt, not extra"


@pytest.mark.parametrize("usage", [{"prompt_tokens_details": None}, {"prompt_tokens_details": {}}, {}, None])
def test_cached_tokens_missing_from_the_response(usage):
    assert Usage.from_api(usage).cached_tokens == 0


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


def test_short_429s_are_retried_up_to_three_times():
    short = httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "slow"}})
    client, sent, clock = make([short, short, short, ok()])
    attempts = []
    result = chat(client, attempts)
    assert result.content is None and len(sent) == 4 and clock.sleeps == [1.0, 1.0, 1.0]
    assert [a.status for a in attempts] == ["rate_limited"] * 3 + ["ok"]


def test_stops_after_the_retries_fail():
    short = httpx.Response(429, headers={"retry-after": "1"})
    client, sent, clock = make([short] * 4)
    with pytest.raises(LLMRateLimitError) as caught:
        chat(client)
    assert len(sent) == 4 and clock.sleeps == [1.0] * 3 and caught.value.retry_after == 1


@pytest.mark.parametrize("seconds, requests", [("60", 2), ("61", 1)])
def test_a_minute_is_the_longest_wait(seconds, requests):
    client, sent, clock = make([httpx.Response(429, headers={"retry-after": seconds}), ok()][:requests])
    attempts = []
    if requests == 2:
        chat(client, attempts)
        assert clock.sleeps == [60.0]
    else:
        with pytest.raises(LLMRateLimitError) as caught:
            chat(client, attempts)
        assert clock.sleeps == [] and caught.value.retry_after == 61
    assert len(sent) == requests and attempts[0].retry_after == int(seconds)


def test_a_429_keeps_its_text_and_all_headers_but_never_the_key():
    response = httpx.Response(429, json={"error": {"code": "rate_limit_exceeded", "message": "Limit 8000, Used 7900"}},
                              headers={"retry-after": "3600", "x-ratelimit-remaining-tokens": "12", "x-request-id": "r1",
                                       "set-cookie": "private=1"})
    client, _, _ = make([response])
    attempts = []
    with pytest.raises(LLMRateLimitError):
        chat(client, attempts)
    text = attempts[0].error
    assert "rate_limit_exceeded" in text and "Limit 8000, Used 7900" in text
    assert '"x-ratelimit-remaining-tokens": "12"' in text and '"x-request-id": "r1"' in text
    assert '"retry-after": "3600"' in text
    assert "private=1" not in text and "test-key" not in text


def test_a_429_without_retry_after_backs_off():
    client, sent, clock = make([httpx.Response(429), ok()])
    chat(client)
    assert clock.sleeps == [1.0] and len(sent) == 2


def truncated(usage=USAGE):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "نص مقطو"},
                                                  "finish_reason": "length"}], "usage": usage})


def test_a_reply_cut_by_max_tokens_is_a_clear_error():
    client, sent, clock = make([truncated()])
    attempts = []
    with pytest.raises(LLMTruncatedError, match="max_tokens=800") as caught:
        chat(client, attempts)
    assert len(sent) == 1 and clock.sleeps == [], "the same request would be cut again"
    assert [(a.status, a.http_status) for a in attempts] == [("truncated", 200)]
    assert attempts[0].usage.completion_tokens == 50 and caught.value.usage.prompt_tokens == 200


def test_a_finished_reply_is_not_an_error():
    body = {"choices": [{"message": {"role": "assistant", "content": "تم"}, "finish_reason": "stop"}], "usage": USAGE}
    client, _, _ = make([httpx.Response(200, json=body)])
    assert chat(client).finish_reason == "stop"


def test_max_tokens_default_is_800():
    assert Settings(_env_file=None).llm_max_tokens == 800


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
    limiter = client.limiter("openai/gpt-oss-20b")
    needed = limiter.calibrate(estimate_prompt_tokens(MESSAGES, TOOLS)) + 800
    chat(client)
    assert clock.sleeps == [pytest.approx(needed * 60 / 8000, abs=0.01)], "the prompt estimate plus all of max_tokens"


def test_calibration_follows_the_prompt_not_the_completion():
    client, _, _ = make([ok(), ok()])
    chat(client)
    limiter = client.limiter("openai/gpt-oss-20b")
    assert limiter.ratio == pytest.approx(200 / estimate_prompt_tokens(MESSAGES, TOOLS))


def test_limiter_calibrates_on_actual_usage():
    limiter = RateLimiter(8000)
    assert limiter.calibrate(3000) == 3301, "uncalibrated: raw estimate plus the 10 % margin"
    limiter.observe(3000, 1650)
    assert limiter.calibrate(3000) == int(3000 * 0.55 * 1.1) + 1
    limiter.observe(3000, 1800)
    assert limiter.ratio == pytest.approx(0.7 * 0.55 + 0.3 * 0.6)


def test_limiter_waits_only_for_the_missing_part():
    clock = FakeClock()
    limiter = RateLimiter(8000, clock=clock, sleep=clock.sleep)
    limiter.update({"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "1000"})
    limiter.acquire(2000)
    assert clock.sleeps == [7.5], "1,000 missing tokens at 8,000 a minute"
    clock.now += 30
    limiter.acquire(4000)
    assert clock.sleeps == [7.5], "30 s refilled 4,000 tokens"


def test_parallel_requests_share_the_budget_without_overspending():
    """Twenty threads through one limiter: the budget never goes below zero."""
    clock = FakeClock()
    limiter = RateLimiter(6000, clock=clock, sleep=clock.sleep)
    limiter.update({"x-ratelimit-limit-tokens": "6000", "x-ratelimit-remaining-tokens": "4000"})
    start = threading.Barrier(20)

    def take(_):
        start.wait()
        return limiter.acquire(500)

    with ThreadPoolExecutor(20) as pool:
        events = list(pool.map(take, range(20)))
    assert sum(e[1] for e in events) == 10000 and limiter._budget >= 0
    assert sum(1 for e in events if e[2] == 0) >= 8, "4,000 tokens let at least eight through at once"
    assert limiter.waited_total > 0, "the requests past the budget had to wait"


def test_a_waiting_request_does_not_block_the_others():
    """The sleep happens outside the lock, so that a response can still update the limiter."""
    released = threading.Event()
    sleeping = threading.Event()

    def sleep(_seconds):
        sleeping.set()
        assert released.wait(5)

    limiter = RateLimiter(6000, sleep=sleep)
    limiter.update({"x-ratelimit-limit-tokens": "6000", "x-ratelimit-remaining-tokens": "0"})
    waiting = threading.Thread(target=limiter.acquire, args=(5000,))
    waiting.start()
    assert sleeping.wait(5)
    limiter.update({"x-ratelimit-remaining-tokens": "6000"})
    limiter.observe(100, 100)
    released.set()
    waiting.join(5)
    assert not waiting.is_alive()


def test_each_request_reports_its_own_wait():
    clock = FakeClock()
    limiter = RateLimiter(6000, clock=clock, sleep=clock.sleep)
    limiter.update({"x-ratelimit-limit-tokens": "6000", "x-ratelimit-remaining-tokens": "0"})
    assert limiter.acquire(1000)[2] == pytest.approx(10.0)
    clock.now += 60
    assert limiter.acquire(1000)[2] == 0


def test_all_threads_get_the_same_limiter():
    client, _, _ = make([])
    start = threading.Barrier(16)

    def get(_):
        start.wait()
        return client.limiter("openai/gpt-oss-20b")

    with ThreadPoolExecutor(16) as pool:
        assert len({id(limiter) for limiter in pool.map(get, range(16))}) == 1


def test_simulated_groq_bucket_no_429_and_little_waiting():
    """A fake server with Groq's continuous refill: 20 calls of 1,650 real tokens, with a prompt estimated far higher."""
    clock = FakeClock()
    state = {"bucket": 8000.0, "at": clock.now, "rejected": 0}
    actual = 1650

    def handler(request):
        state["bucket"] = min(8000.0, state["bucket"] + (clock.now - state["at"]) * 8000 / 60)
        state["at"] = clock.now
        if state["bucket"] < actual:
            state["rejected"] += 1
            return httpx.Response(429, headers={"retry-after": "5"})
        state["bucket"] -= actual
        body = {"choices": [{"message": {"role": "assistant", "content": "تمام"}}],
                "usage": {"prompt_tokens": actual - 50, "completion_tokens": 50}}
        return httpx.Response(200, json=body, headers={"x-ratelimit-limit-tokens": "8000",
                                                       "x-ratelimit-remaining-tokens": str(int(state["bucket"]))})

    client = GroqClient("k", transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep,
                        jitter=lambda: 0.0)
    messages = [{"role": "system", "content": "x" * 8100}, {"role": "user", "content": "مرحبا"}]
    assert estimate_prompt_tokens(messages) > 2600
    for _ in range(20):
        client.chat("m", messages)
    assert state["rejected"] == 0
    minimum = (20 * actual - 8000) * 60 / 8000  # seconds the bucket needs to refill, whatever the client does
    assert minimum <= sum(clock.sleeps) <= minimum * 1.2


def test_a_server_that_holds_max_tokens_never_rejects():
    """The likely cause of the 429 in step 6: Groq needs prompt + max_tokens to be free, then charges what was used."""
    clock = FakeClock()
    state = {"bucket": 8000.0, "at": clock.now, "rejected": 0}

    def handler(request):
        state["bucket"] = min(8000.0, state["bucket"] + (clock.now - state["at"]) * 8000 / 60)
        state["at"] = clock.now
        body = json.loads(request.content)
        if state["bucket"] < 1850 + body["max_tokens"]:
            state["rejected"] += 1
            return httpx.Response(429, headers={"retry-after": "1", "x-ratelimit-limit-tokens": "8000",
                                                "x-ratelimit-remaining-tokens": str(int(state["bucket"]))})
        state["bucket"] -= 1900
        reply = {"choices": [{"message": {"role": "assistant", "content": "تمام"}}],
                 "usage": {"prompt_tokens": 1850, "completion_tokens": 50}}
        return httpx.Response(200, json=reply, headers={"x-ratelimit-limit-tokens": "8000",
                                                        "x-ratelimit-remaining-tokens": str(int(state["bucket"]))})

    client = GroqClient("k", transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep,
                        jitter=lambda: 0.0)
    messages = [{"role": "system", "content": "x" * 5400}, {"role": "user", "content": "مرحبا"}]
    client.chat("m", messages)  # the first call teaches the limiter how large the prompt really is
    state["rejected"] = 0
    for _ in range(12):
        client.chat("m", messages)
    assert state["rejected"] == 0


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
