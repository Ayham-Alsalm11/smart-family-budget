"""scripts/run_scenarios.py with a scripted model: no Groq, no Ollama."""
import json
from types import SimpleNamespace

import httpx
import pytest

from app.auth import passwords
from app.llm.client import GroqClient, LLMRateLimitError
from scripts import run_scenarios

from fake_llm import FakeLLM, call, respond

DAILY = "tokens per day exceeded | headers: {\"retry-after\": \"3600\"}"


def scenario(name):
    return {"id": name, "who": "manager", "setup": [], "turns": [{"text": f"مرحبا {name}", "expect": {"ended_by": "reply"}}]}


class ScriptedLLM(FakeLLM):
    def __init__(self, responses=()):
        super().__init__(list(responses))
        self.last_rate_limits = {"openai/gpt-oss-120b": {"x-ratelimit-remaining-requests": "900"},
                                 "openai/gpt-oss-20b": {"x-ratelimit-remaining-requests": "800"}}

    def close(self):
        pass

    def messages_sent(self):
        return [request[-1]["content"] for request in self.requests]


class StubOllama:
    def __init__(self, *_args):
        pass

    def warmup(self):
        return {"ok": True, "seconds": 0.0, "dimensions": 8}

    def embed(self, _text):
        return [0.5] * 8


@pytest.fixture
def harness(tmp_path, monkeypatch):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": [scenario(n) for n in ("S1", "S2", "S3")]}), encoding="utf-8")
    monkeypatch.setattr(run_scenarios, "SCENARIOS", path)
    monkeypatch.setattr(run_scenarios, "BACKEND_DIR", tmp_path)
    monkeypatch.setattr(run_scenarios, "OllamaEmbedder", StubOllama)
    monkeypatch.setattr(passwords, "ROUNDS", 4)

    def run(argv, llm):
        monkeypatch.setattr(run_scenarios, "GroqClient", SimpleNamespace(from_settings=lambda _settings: llm))
        return run_scenarios.main(argv)

    return SimpleNamespace(run=run, folder=tmp_path / "data" / "measurements")


def results(harness):
    path = next(harness.folder.glob("scenarios_*.json"))
    return path, json.loads(path.read_text(encoding="utf-8"))


def stopped_at_s2(harness, retry_after=3600):
    llm = ScriptedLLM([respond(content="أهلا"), LLMRateLimitError(DAILY, 429, retry_after)])
    code = harness.run(["--models", "main"], llm)
    return code, llm


def test_a_full_run_marks_every_scenario_done(harness):
    assert harness.run(["--models", "main"], ScriptedLLM([respond(content="أهلا") for _ in range(3)])) == 0
    _, report = results(harness)
    run = report["runs"]["main"]
    assert [(s["id"], s["done"]) for s in run["scenarios"]] == [("S1", True), ("S2", True), ("S3", True)]
    assert run["summary"]["scenarios_done"] == 3 and run["summary"]["rate_limited_attempts"] == 0
    assert run["rate_limits"]["x-ratelimit-remaining-requests"] == "900"
    assert "stopped" not in report


def test_the_daily_quota_stops_cleanly_and_says_where(harness, capsys):
    code, llm = stopped_at_s2(harness)
    assert code == 2 and len(llm.requests) == 2, "S3 is not tried, and nothing is retried"
    path, report = results(harness)
    stopped = report["stopped"]
    assert (stopped["reason"], stopped["model"], stopped["scenario"], stopped["turn"]) == ("daily_quota", "main", "S2", 1)
    assert stopped["retry_after"] == 3600 and "tokens per day exceeded" in stopped["error"]
    assert stopped["rate_limits"]["x-ratelimit-remaining-requests"] == "900"
    printed = capsys.readouterr().out
    assert "daily_quota in S2.1" in printed and f"--resume {path}" in printed


def test_each_scenario_is_on_disk_when_it_ends(harness):
    stopped_at_s2(harness)
    _, report = results(harness)
    first, second = report["runs"]["main"]["scenarios"]
    assert (first["id"], first["done"]) == ("S1", True) and first["turns"][0]["reply"] == "أهلا"
    assert (second["id"], second["done"]) == ("S2", False)
    assert second["turns"][0]["rate_limit_events"][0]["retry_after"] == 3600


def test_retries_that_fail_are_told_apart_from_the_daily_quota(harness):
    code, _ = stopped_at_s2(harness, retry_after=1)
    assert code == 2 and results(harness)[1]["stopped"]["reason"] == "retries_failed"


def test_resume_continues_from_the_first_unfinished_scenario(harness):
    stopped_at_s2(harness)
    path, _ = results(harness)
    llm = ScriptedLLM([respond(content="أهلا"), respond(content="أهلا")])
    assert harness.run(["--resume", str(path)], llm) == 0

    assert llm.messages_sent() == ["مرحبا S2", "مرحبا S3"], "S1 is not run again"
    assert len(list(harness.folder.glob("scenarios_*.json"))) == 1, "the same file continues"
    report = json.loads(path.read_text(encoding="utf-8"))
    run = report["runs"]["main"]
    assert [(s["id"], s["done"]) for s in run["scenarios"]] == [("S1", True), ("S2", True), ("S3", True)]
    assert [s["id"] for s in run["abandoned"]] == ["S2"] and "stopped" not in report
    assert report["earlier_stops"][0]["scenario"] == "S2"


def test_resume_keeps_the_date_the_models_and_the_scenario_list(harness):
    llm = ScriptedLLM([respond(content="أهلا"), LLMRateLimitError(DAILY, 429, 3600)])
    harness.run(["--today", "2026-10-10", "--models", "main,small", "--only", "S1,S3"], llm)
    path, _ = results(harness)
    assert llm.requests[0][0]["content"].count("2026-10-10") == 1

    resumed = ScriptedLLM([respond(content="أهلا") for _ in range(3)])
    assert harness.run(["--resume", str(path)], resumed) == 0
    assert resumed.messages_sent() == ["مرحبا S3", "مرحبا S1", "مرحبا S3"]
    assert "2026-10-10" in resumed.requests[0][0]["content"]
    report = json.loads(path.read_text(encoding="utf-8"))
    assert {name: [s["id"] for s in run["scenarios"]] for name, run in report["runs"].items()} == {
        "main": ["S1", "S3"], "small": ["S1", "S3"]}


def test_a_stop_on_the_second_model_keeps_the_first(harness):
    llm = ScriptedLLM([respond(content="أهلا") for _ in range(3)] + [LLMRateLimitError(DAILY, 429, 3600)])
    assert harness.run([], llm) == 2
    path, report = results(harness)
    assert report["stopped"]["model"] == "small"
    resumed = ScriptedLLM([respond(content="أهلا") for _ in range(3)])
    assert harness.run(["--resume", str(path)], resumed) == 0
    assert resumed.requests[0] and len(resumed.requests) == 3, "only the second model's three scenarios"


def test_resuming_a_finished_file_calls_nothing(harness):
    harness.run(["--models", "main"], ScriptedLLM([respond(content="أهلا") for _ in range(3)]))
    path, _ = results(harness)
    llm = ScriptedLLM()
    assert harness.run(["--resume", str(path)], llm) == 0 and llm.requests == []


def test_abandoned_turns_still_count_as_spent(harness):
    harness_llm = ScriptedLLM([respond(content="أهلا"), respond(call("get_total", period="this_month")),
                               LLMRateLimitError(DAILY, 429, 3600)])
    assert harness.run(["--models", "main"], harness_llm) == 2
    path, _ = results(harness)
    harness.run(["--resume", str(path)], ScriptedLLM([respond(content="أهلا"), respond(content="أهلا")]))
    summary = json.loads(path.read_text(encoding="utf-8"))["runs"]["main"]["summary"]
    assert summary["tokens_in_abandoned_turns"] == 120 and summary["tokens"] == 360


def test_a_short_429_is_retried_and_only_counted(harness):
    answers = [httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "minute limit"}})]

    def handler(_request):
        if answers:
            return answers.pop()
        body = {"choices": [{"message": {"role": "assistant", "content": "أهلا"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        return httpx.Response(200, json=body)

    client = GroqClient("k", transport=httpx.MockTransport(handler), sleep=lambda _seconds: None, jitter=lambda: 0.0)
    assert harness.run(["--models", "main"], client) == 0
    _, report = results(harness)
    assert "stopped" not in report
    run = report["runs"]["main"]
    turn = run["scenarios"][0]["turns"][0]
    assert run["summary"]["rate_limited_attempts"] == 1 and turn["model_calls"] == 1
    assert turn["rate_limit_events"][0]["retry_after"] == 1 and "minute limit" in turn["rate_limit_events"][0]["error"]
    assert [c["status"] for c in turn["calls"]] == ["rate_limited", "ok"]


def test_the_scenario_file_is_loaded_from_the_repository():
    data = json.loads(run_scenarios.SCENARIOS.read_text(encoding="utf-8"))
    assert [s["id"] for s in data["scenarios"]] == [f"S{n}" for n in range(1, 11)]
