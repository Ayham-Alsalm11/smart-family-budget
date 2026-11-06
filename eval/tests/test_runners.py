"""The RQ2 runner and the GP1 wrapper with fake systems, without Groq (§9.4, §9.5)."""
import json
import sys
from datetime import date

import pytest

from eval import EVAL_DIR
from eval.check_truncation import choose_sentences, summarise
from eval.generate_eval_db import write_records
from eval.run_rq2 import (DailyQuota, WrongDay, build_item, load_database, read_items, run_items, summarise as summarise_rq2,
                          summary_markdown)

sys.path.insert(0, str(EVAL_DIR))  # gp1_answer.py imports gp1_http as a script does

import gp1_answer  # noqa: E402
import gp1_http  # noqa: E402

TODAY = date(2026, 11, 6)
RECORDS = {
    "today": TODAY.isoformat(), "family": {"monthly_income": 500_000}, "categories": [],
    "expenses": [{"date": "2026-11-01", "category": "إيجار", "amount": 125_000}],
}
MANIFEST = {"today": TODAY.isoformat(), "records_sha256": "abc"}
QUESTIONS = [{"id": "P01", "type": "total_this_month", "question": "قديش؟"},
             {"id": "P02", "type": "total_last_month", "question": "قديش؟"}]
TOKENS = {"prompt": 90, "completion": 10, "reasoning": 2, "cached": 0, "total": 100}


def fake_answer(reply="125,000", error=None, seconds=3.0, waited=1.0):
    return {"reply": reply, "error": error, "tokens": TOKENS, "requests": 2, "seconds": seconds, "waited_s": waited}


def always(answer):
    return lambda question: answer


def test_load_database_refuses_another_day(tmp_path):
    digest = write_records(RECORDS, tmp_path / "records.json")
    (tmp_path / "manifest.json").write_text(json.dumps({"today": "2026-11-05", "records_sha256": digest}), encoding="utf-8")
    with pytest.raises(WrongDay, match="2026-11-05"):
        load_database(tmp_path, TODAY)


def test_load_database_refuses_changed_records(tmp_path):
    write_records(RECORDS, tmp_path / "records.json")
    (tmp_path / "manifest.json").write_text(json.dumps({"today": TODAY.isoformat(), "records_sha256": "x"}), encoding="utf-8")
    with pytest.raises(WrongDay, match="does not match"):
        load_database(tmp_path, TODAY)


def test_load_database_accepts_the_right_day(tmp_path):
    digest = write_records(RECORDS, tmp_path / "records.json")
    (tmp_path / "manifest.json").write_text(json.dumps({"today": TODAY.isoformat(), "records_sha256": digest}), encoding="utf-8")
    assert load_database(tmp_path, TODAY)[1]["today"] == TODAY.isoformat()


def test_a_failure_is_wrong_even_with_the_right_number():
    """GP1 answers with its raw summary when the API fails, and that summary holds the right number."""
    item = build_item(QUESTIONS[0], "gp1", {"value": 125_000}, fake_answer(error="api: 429"), MANIFEST)
    assert item["numbers"] == ["125000"] and not item["strict"] and not item["lenient"]


def test_every_pair_is_written_and_scored(tmp_path):
    path = tmp_path / "items.jsonl"
    added = run_items(QUESTIONS, RECORDS, MANIFEST, {"agent": always(fake_answer()), "gp1": always(fake_answer("ما بعرف"))}, path)
    items = read_items(path)
    assert added == 4 and [(i["id"], i["system"]) for i in items] == [
        ("P01", "agent"), ("P01", "gp1"), ("P02", "agent"), ("P02", "gp1")]
    assert items[0]["strict"] and not items[1]["strict"] and items[0]["reference"] == 125_000
    assert items[2]["reference"] == 0


def test_a_stopped_run_continues_from_the_first_missing_item(tmp_path):
    path = tmp_path / "items.jsonl"
    calls = []

    def stops_at_the_second(question):
        calls.append(question["id"])
        if question["id"] == "P02":
            raise DailyQuota("limit")
        return fake_answer()

    with pytest.raises(DailyQuota):
        run_items(QUESTIONS, RECORDS, MANIFEST, {"agent": stops_at_the_second, "gp1": always(fake_answer())}, path)
    assert len(read_items(path)) == 2

    calls.clear()
    def agent(question):
        calls.append(question["id"])
        return fake_answer()

    assert run_items(QUESTIONS, RECORDS, MANIFEST, {"agent": agent, "gp1": always(fake_answer())}, path) == 2
    assert calls == ["P02"] and len(read_items(path)) == 4


def test_the_summary_counts_failures_as_wrong_and_leaves_waits_out(tmp_path):
    path = tmp_path / "items.jsonl"
    run_items(QUESTIONS[:1], RECORDS, MANIFEST,
              {"agent": always(fake_answer()), "gp1": always(fake_answer(error="crash", seconds=5, waited=0))}, path)
    summary = summarise_rq2(read_items(path))
    assert summary["agent"]["strict_accuracy"] == 1.0 and summary["gp1"]["strict_accuracy"] == 0.0
    assert summary["gp1"]["api_failure_rate"] == 1.0 and summary["agent"]["api_failure_rate"] == 0.0
    assert summary["agent"]["mean_seconds"] == 2.0 and summary["agent"]["tokens_per_question"] == 100
    assert summary["agent"]["by_type"]["total_this_month"]["strict"] == 1
    config = {"run_id": "x", "model": "m", "reasoning_effort": "low", "git_commit": "0" * 40, "questions_file": "q",
              "segments": [{"today": "2026-11-06"}]}
    assert "total_this_month | 1/1 | 1/1 | 0/1 | 0/1" in summary_markdown(config, summary)


# ---------- the GP1 wrapper and answer step ----------

class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, text

    def json(self):
        return self._body


def good(remaining="7000", content="ok", finish="stop"):
    return FakeResponse(body={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                              "usage": {"prompt_tokens": 800, "completion_tokens": 60,
                                        "completion_tokens_details": {"reasoning_tokens": 40},
                                        "prompt_tokens_details": {"cached_tokens": 512}}},
                        headers={"x-ratelimit-remaining-tokens": remaining, "x-ratelimit-limit-tokens": "8000"})


URL = "https://api.groq.com/openai/v1/chat/completions"


def test_usage_and_finish_reason_are_logged():
    log = gp1_http.RequestLog(post=lambda url, **options: good())
    assert log.post(URL, json={}).status_code == 200
    call = log.calls[0]
    assert (call["finish_reason"], call["prompt_tokens"], call["reasoning_tokens"], call["cached_tokens"]) == ("stop", 800, 40, 512)


def test_ollama_requests_pass_through_unlogged():
    log = gp1_http.RequestLog(post=lambda url, **options: FakeResponse())
    log.post("http://localhost:11434/api/embed", json={})
    assert log.calls == []


def test_it_waits_when_the_minutes_tokens_are_nearly_gone():
    slept = []
    now = [100.0]
    log = gp1_http.RequestLog(post=lambda url, **options: good(remaining="500"), sleep=slept.append, clock=lambda: now[0])
    log.post(URL, json={})
    log.post(URL, json={})
    assert slept == [pytest.approx((2000 - 500) * 60 / 8000)] and log.calls[1]["waited_s"] > 0


def test_a_long_retry_after_is_the_daily_quota_and_a_short_one_is_not():
    short = gp1_http.RequestLog(post=lambda url, **options: FakeResponse(429, headers={"retry-after": "12"}, text="slow"))
    short.post(URL, json={})
    long = gp1_http.RequestLog(post=lambda url, **options: FakeResponse(429, headers={"retry-after": "3600"}, text="tpd"))
    long.post(URL, json={})
    assert not short.daily_quota and long.daily_quota


def test_failure_of_the_last_request():
    assert gp1_http.failure([]) is None
    assert gp1_http.failure([{"http_status": 429, "error": "x"}, {"http_status": 200, "content_length": 5}]) is None
    assert gp1_http.failure([{"http_status": 429, "error": "x"}]).startswith("api: 429")
    assert gp1_http.failure([{"http_status": None, "error": "Timeout"}]).startswith("api: None")
    assert "empty reply" in gp1_http.failure([{"http_status": 200, "content_length": 0, "finish_reason": "length"}])


class FakeEngine:
    def __init__(self, result=None, raises=None, log=None):
        self.result, self.raises, self.log = result, raises, log

    def process_message(self, text):
        if self.log:
            self.log.post(URL, json={})
        if self.raises:
            raise self.raises
        return self.result


QUESTION = {"id": "P01", "question": "قديش؟"}


def test_an_answer_with_its_calls():
    log = gp1_http.RequestLog(post=lambda url, **options: good())
    item = gp1_answer.answer(FakeEngine({"type": "query", "response": "125,000"}, log=log), log, QUESTION)
    assert item["reply"] == "125,000" and item["error"] is None and len(item["calls"]) == 1


def test_a_crash_is_recorded_as_a_failure():
    """A1: `_parse_json(None)` raises outside GP1's own try."""
    log = gp1_http.RequestLog(post=lambda url, **options: good())
    item = gp1_answer.answer(FakeEngine(raises=AttributeError("NoneType")), log, QUESTION)
    assert item["gp1_type"] == "exception" and item["error"].startswith("AttributeError")


def test_the_raw_summary_after_an_api_error_is_a_failure():
    log = gp1_http.RequestLog(post=lambda url, **options: FakeResponse(429, headers={"retry-after": "5"}, text="slow"))
    item = gp1_answer.answer(FakeEngine({"type": "query", "response": "المصروفات: 125,000"}, log=log), log, QUESTION)
    assert item["error"].startswith("api: 429")


def test_gp1s_error_type_is_a_failure():
    log = gp1_http.RequestLog(post=lambda url, **options: good())
    item = gp1_answer.answer(FakeEngine({"type": "error", "response": "لم أستطع فهم الرد."}, log=log), log, QUESTION)
    assert item["error"] == "لم أستطع فهم الرد."


def test_an_added_expense_is_reported():
    log = gp1_http.RequestLog(post=lambda url, **options: good())
    item = gp1_answer.answer(FakeEngine({"type": "expense", "response": "تم"}, log=log), log, QUESTION, lambda: [5000.0])
    assert item["expense_added"] == [5000.0]


# ---------- the truncation check ----------

def test_ten_sentences_spread_over_the_dataset():
    chosen = choose_sentences()
    assert len(chosen) == 10 and len({row["sentence"] for row in chosen}) == 10
    assert {"expense"} < {row["expected_type"] for row in chosen}


def test_the_truncation_summary():
    items = [
        {"valid_json": True, "finish_reason": "stop", "crash": None, "reasoning_tokens": 40, "completion_tokens": 80, "total_tokens": 900},
        {"valid_json": False, "finish_reason": "length", "crash": None, "reasoning_tokens": 200, "completion_tokens": 200, "total_tokens": 1000},
    ]
    summary = summarise(items)
    assert (summary["valid_json"], summary["cut_at_max_tokens"], summary["reasoning_tokens_max"]) == (1, 1, 200)
