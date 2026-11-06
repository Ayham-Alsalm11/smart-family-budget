"""The ReAct loop with a scripted model: steps, limits, logging and tool rejection (§6.1, §6.2)."""
import json

import httpx
from sqlalchemy import select

from app.agent.grounding import extract_numbers, ungrounded_numbers
from app.agent.loop import (
    EMPTY_REPLY_REQUEST, REPLY_EMPTY, REPLY_LLM_ERROR, REPLY_STEP_LIMIT, REPLY_TOOL_REJECTED, REPLY_TRUNCATED, run_turn,
)
from app.db.models import Expense, LlmCall, Message, PendingAction, ToolLog
from app.llm.client import GroqClient, LLMRateLimitError, LLMToolValidationError, ToolCall
from app.services import expenses

from fake_llm import FakeLLM, call, respond

FOOD = "أكل وشرب"


def run(session, ctx, message, responses, **kwargs):
    llm = FakeLLM(responses)
    return run_turn(session, ctx, message, llm, model="fake-model", **kwargs), llm


def test_multi_step_example(session, manager_ctx):
    """Report Part 1 §4.3.1: two expenses, then check_budget, then a reply built from the results."""
    message = "اشتريت خضرا بـ 3000 ولحمة بـ 20000، وقلي إذا تجاوزت ميزانية الأكل"
    turn, llm = run(session, manager_ctx, message, [
        respond(call("add_expense", amount=3000, category=FOOD, description="خضرا"),
                call("add_expense", amount=20000, category=FOOD, description="لحمة")),
        respond(call("check_budget", category=FOOD)),
        respond(content="سجّلت 3000 و20,000. صرفت 23000 من حد 100000، ما تجاوزت الميزانية."),
    ])

    assert turn.ended_by == "reply"
    assert turn.llm_steps == 3
    assert [(s.step, s.tool, s.status) for s in turn.steps] == [
        (1, "add_expense", "ok"), (1, "add_expense", "ok"), (2, "check_budget", "ok")]
    assert turn.ungrounded == []
    assert turn.usage.total_tokens == 3 * 120

    rows = session.scalars(select(Expense)).all()
    assert [r.amount for r in rows] == [3000, 20000]
    assert all(r.created_by_agent and r.source_text == message for r in rows)

    # The second request carries the assistant's tool calls and both tool results.
    second = llm.requests[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "tool"]
    assert '"ok":true' in second[3]["content"]

    stored = session.scalars(select(Message).order_by(Message.id)).all()
    assert [(m.role, m.content) for m in stored] == [("user", message), ("assistant", turn.reply)]
    assert stored[1].tool_summary == "add_expense → #1; add_expense → #2; check_budget → ok"
    assert len(session.scalars(select(LlmCall)).all()) == 3
    assert [t.status for t in session.scalars(select(ToolLog))] == ["ok", "ok", "ok"]


def test_invalid_arguments_go_back_to_the_model(session, member_ctx):
    turn, llm = run(session, member_ctx, "دفعت 5000 بنزين", [
        respond(call("add_expense", amount=5000, category="بنزين")),
        respond(call("add_expense", amount=5000, category="بنزين ومواصلات")),
        respond(content="تمام، سجّلت 5000."),
    ])
    assert [s.status for s in turn.steps] == ["invalid_args", "ok"]
    assert session.scalars(select(Expense)).all()[0].category.name_ar == "بنزين ومواصلات"
    assert '"code":"invalid_args"' in llm.requests[1][-1]["content"]
    assert len(session.scalars(select(Expense)).all()) == 1


def test_unreadable_arguments_and_unknown_tool(session, member_ctx):
    broken = ToolCall("bad_1", "add_expense", None, "{amount: 5", "arguments are not valid JSON")
    turn, _ = run(session, member_ctx, "x", [
        respond(broken, call("transfer_money", amount=5)),
        respond(content="ما فيني ساعدك بهالشي."),
    ])
    assert [s.status for s in turn.steps] == ["invalid_args", "invalid_args"]
    assert turn.steps[0].arguments == {"raw": "{amount: 5"}


def test_forbidden_is_returned_to_the_model(session, member_ctx):
    turn, llm = run(session, member_ctx, "كم صرف بابا؟", [
        respond(call("get_breakdown", period="this_month", by="member")),
        respond(content="عذراً، هالمعلومة مو متاحة لحسابك."),
    ])
    assert turn.steps[0].status == "forbidden"
    assert '"code":"forbidden"' in llm.requests[1][-1]["content"]
    assert session.scalars(select(ToolLog)).one().status == "forbidden"


def test_ask_clarification_ends_the_turn(session, member_ctx):
    turn, llm = run(session, member_ctx, "دفعت 50 لأبو أحمد", [
        respond(call("ask_clarification", question="على شو دفعت الـ 50؟", missing_fields=["category"]),
                call("add_expense", amount=50, category="متفرقات")),
    ])
    assert turn.ended_by == "clarification"
    assert turn.reply == "على شو دفعت الـ 50؟"
    assert len(llm.requests) == 1
    assert [s.tool for s in turn.steps] == ["ask_clarification"]
    assert session.scalars(select(Expense)).all() == []


def test_step_limit(session, member_ctx):
    turn, llm = run(session, member_ctx, "x", [respond(call("list_expenses")) for _ in range(6)], max_steps=6)
    assert turn.ended_by == "step_limit"
    assert turn.reply == REPLY_STEP_LIMIT
    assert len(llm.requests) == 6 and len(turn.steps) == 6


def test_ungrounded_number_is_logged(session, member_ctx):
    expenses.add_expense(session, member_ctx, amount=5000, category=FOOD)
    turn, _ = run(session, member_ctx, "كم صرفت؟", [
        respond(call("get_total", period="this_month")),
        respond(content="صرفت 5,000 ليرة، يعني تقريباً ١٢٬٣٤٥ بالأسبوع."),
        respond(content="صرفت 5,000 ليرة، يعني تقريباً ١٢٬٣٤٥ بالأسبوع."),
    ])
    assert turn.ungrounded == ["12345"]
    log = session.scalars(select(ToolLog).where(ToolLog.tool_name == "grounding")).one()
    assert (log.tool_name, log.arguments) == ("grounding", {"numbers": ["12345"]})


def test_llm_failure_is_a_counted_failure(session, member_ctx):
    turn, _ = run(session, member_ctx, "مرحبا", [LLMRateLimitError("rate limited", 429)])
    assert turn.ended_by == "llm_error"
    assert turn.reply == REPLY_LLM_ERROR
    assert session.scalars(select(LlmCall)).one().status == "rate_limited"
    assert len(session.scalars(select(Message)).all()) == 2


def rejected(tool="ask_clarification"):
    return LLMToolValidationError(
        f"tool_use_failed: Tool call validation failed: tool call validation failed: parameters for tool {tool} "
        "did not match schema: errors: [`/missing_fields/0`: value must be one of 'amount', 'category']", 400)


def test_provider_rejection_is_corrected_by_the_model(session, member_ctx):
    turn, llm = run(session, member_ctx, "دفعت 50 لأبو أحمد", [
        rejected(),
        respond(call("ask_clarification", question="على شو دفعت الـ 50؟", missing_fields=["category"])),
    ])
    assert turn.ended_by == "clarification" and turn.reply == "على شو دفعت الـ 50؟"
    assert [(s.step, s.tool, s.status) for s in turn.steps] == [(1, "ask_clarification", "invalid_args"),
                                                                 (2, "ask_clarification", "ok")]
    feedback = llm.requests[1][-1]
    assert feedback["role"] == "user" and feedback["content"].startswith("Your last tool call was rejected: parameters")
    assert "Fix the arguments or omit optional ones" in feedback["content"]
    logs = session.scalars(select(ToolLog).order_by(ToolLog.id)).all()
    assert [(log.tool_name, log.status) for log in logs] == [("ask_clarification", "invalid_args"),
                                                            ("ask_clarification", "ok")]
    assert session.scalars(select(LlmCall).order_by(LlmCall.id)).first().status == "tool_use_failed"


def test_provider_rejection_gives_up_after_two_corrections(session, member_ctx):
    turn, llm = run(session, member_ctx, "امسح آخر مصروف", [rejected(), rejected("list_expenses"), rejected()])
    assert turn.ended_by == "tool_rejected"
    assert turn.reply == REPLY_TOOL_REJECTED != REPLY_LLM_ERROR
    assert len(llm.requests) == 3
    assert [log.status for log in session.scalars(select(ToolLog))] == ["invalid_args"] * 3
    assert session.scalars(select(Message).where(Message.role == "assistant")).one().content == REPLY_TOOL_REJECTED


# ---------- empty replies ----------

EMPTY = respond(content=None)


def test_an_empty_reply_is_asked_for_again_once(session, member_ctx):
    turn, llm = run(session, member_ctx, "مرحبا", [EMPTY, respond(content="أهلا")])
    assert (turn.reply, turn.ended_by, turn.llm_steps, turn.empty_retried) == ("أهلا", "reply", 2, True)
    assert llm.requests[1][-1] == {"role": "user", "content": EMPTY_REPLY_REQUEST}
    assert llm.requests[1][-2] == {"role": "user", "content": "مرحبا"}, "the empty message is not in the conversation"


def test_the_retry_may_call_a_tool(session, member_ctx):
    turn, _ = run(session, member_ctx, "كم؟", [EMPTY, respond(call("get_total", period="this_month")), respond(content="صفر")])
    assert [s.tool for s in turn.steps] == ["get_total"] and turn.reply == "صفر"


def test_empty_twice_keeps_the_current_reply(session, member_ctx):
    turn, llm = run(session, member_ctx, "مرحبا", [EMPTY, EMPTY, respond(content="لا يجب أن يصل")])
    assert (turn.reply, turn.ended_by) == (REPLY_EMPTY, "reply") and len(llm.requests) == 2 and len(llm.responses) == 1


def test_only_one_retry_in_a_turn(session, member_ctx):
    turn, llm = run(session, member_ctx, "كم؟", [EMPTY, respond(call("get_total", period="this_month")), EMPTY,
                                                 respond(content="لا يجب أن يصل")])
    assert turn.reply == REPLY_EMPTY and len(llm.requests) == 3


def test_no_retry_on_the_last_step(session, member_ctx):
    turn, llm = run(session, member_ctx, "مرحبا", [EMPTY], max_steps=1)
    assert turn.reply == REPLY_EMPTY and not turn.empty_retried and len(llm.requests) == 1


def test_the_retry_counts_in_the_step_limit(session, member_ctx):
    turn, llm = run(session, member_ctx, "كم؟", [EMPTY, respond(call("get_total", period="this_month"))], max_steps=2)
    assert (turn.ended_by, turn.reply) == ("step_limit", REPLY_STEP_LIMIT) and len(llm.requests) == 2


def test_white_space_is_an_empty_reply(session, member_ctx):
    turn, llm = run(session, member_ctx, "مرحبا", [respond(content="  \n "), respond(content="أهلا")])
    assert turn.reply == "أهلا" and len(llm.requests) == 2


def test_an_empty_answer_to_a_repair_keeps_the_first_reply(session, member_ctx):
    turn, _ = run(session, member_ctx, "كم؟", [respond(content="صرفت 777"), EMPTY, EMPTY])
    assert (turn.reply, turn.ended_by, turn.repaired, turn.empty_retried) == ("صرفت 777", "reply", True, True)
    assert turn.ungrounded == ["777"]


def groq_replies(*messages, finish_reason="stop"):
    queue = list(messages)

    def handler(_request):
        body = {"choices": [{"message": queue.pop(0), "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 1700, "completion_tokens": 30,
                          "completion_tokens_details": {"reasoning_tokens": 5}}}
        return httpx.Response(200, json=body)

    return GroqClient("k", transport=httpx.MockTransport(handler))


def test_what_an_empty_reply_held_is_saved_in_llm_calls(session, member_ctx):
    empty = {"role": "assistant", "content": "", "reasoning": "abcd", "channel": "commentary", "annotations": []}
    client = groq_replies(empty, {"role": "assistant", "content": "أهلا"})
    turn = run_turn(session, member_ctx, "مرحبا", client, model="fake-model")
    assert turn.reply == "أهلا"
    first, second = session.scalars(select(LlmCall).order_by(LlmCall.id)).all()
    assert (first.status, first.http_status, first.completion_tokens) == ("empty", 200, 30)
    assert json.loads(first.error) == {"finish_reason": "stop", "content_length": 0, "reasoning_length": 4,
                                       "unexpected_fields": ["annotations", "channel"]}
    assert (second.status, second.error) == ("ok", None)
    assert turn.llm_calls[0]["status"] == "empty"


def test_white_space_length_is_saved(session, member_ctx):
    client = groq_replies({"role": "assistant", "content": "  "}, {"role": "assistant", "content": "أهلا"})
    run_turn(session, member_ctx, "مرحبا", client, model="fake-model")
    assert json.loads(session.scalars(select(LlmCall)).first().error)["content_length"] == 2


def test_a_tool_call_without_content_is_not_empty(session, member_ctx):
    tool_call = {"id": "c1", "type": "function", "function": {"name": "get_total", "arguments": '{"period": "this_month"}'}}
    client = groq_replies({"role": "assistant", "content": None, "tool_calls": [tool_call]},
                          {"role": "assistant", "content": "تم"})
    turn = run_turn(session, member_ctx, "كم؟", client, model="fake-model")
    assert not turn.empty_retried and {c.status for c in session.scalars(select(LlmCall))} == {"ok"}


def test_a_cut_reply_is_recorded_and_explained(session, member_ctx):
    body = {"choices": [{"message": {"role": "assistant", "content": "نص مقطو"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1700, "completion_tokens": 600,
                      "completion_tokens_details": {"reasoning_tokens": 500}}}
    client = GroqClient("k", transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=body)))
    turn = run_turn(session, member_ctx, "x", client, model="fake-model")
    assert (turn.reply, turn.ended_by) == (REPLY_TRUNCATED, "llm_error") and "truncated" in turn.error
    assert turn.usage.total_tokens == 2300, "the call was paid for"
    row = session.scalars(select(LlmCall)).one()
    assert (row.status, row.http_status, row.completion_tokens, row.reasoning_tokens) == ("truncated", 200, 600, 500)
    assert "max_tokens" in row.error


def test_the_text_and_headers_of_a_429_reach_llm_calls(session, member_ctx):
    def handler(_request):
        return httpx.Response(429, headers={"retry-after": "3600", "x-ratelimit-remaining-tokens": "12"},
                              json={"error": {"message": "tokens per day"}})

    client = GroqClient("secret-key", transport=httpx.MockTransport(handler))
    turn = run_turn(session, member_ctx, "x", client, model="fake-model")
    row = session.scalars(select(LlmCall)).one()
    assert "tokens per day" in row.error and "x-ratelimit-remaining-tokens" in row.error
    assert "secret-key" not in row.error
    assert turn.llm_calls[-1]["retry_after"] == 3600 and "tokens per day" in turn.llm_calls[-1]["error"]


def test_pending_action_and_alerts_reach_the_turn(session, member_ctx):
    mine = expenses.add_expense(session, member_ctx, amount=79000, category=FOOD)
    turn, _ = run(session, member_ctx, f"امسح #{mine['id']}", [
        respond(call("add_expense", amount=2000, category=FOOD),
                call("delete_expense", expense_id=mine["id"])),
        respond(content="بدك امسحه؟"),
    ])
    assert turn.alerts and turn.alerts[0]["status"] == "warning"
    assert turn.pending_action["action"] == "delete_expense"
    assert session.get(PendingAction, turn.pending_action["confirmation_id"]).message_id == turn.message_id


def test_number_extraction():
    assert extract_numbers("٥٠٠٠ و 5,000 و 1,234,567.5 و 2026-11-04") == {5000, 1234567.5, 2026, 11, 4}
    assert ungrounded_numbers("الحد 80% والمصروف 21.4", ['{"percent_used":80.0,"spent":21.40}']) == []
