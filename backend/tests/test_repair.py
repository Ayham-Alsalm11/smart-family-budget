"""One repair attempt for numbers the tool results do not support (§6.9), numbers from the user's message, and the
year corrected in code before the check."""
from decimal import Decimal

from sqlalchemy import select

from app.agent.grounding import ungrounded_numbers
from app.agent.loop import REPAIR_REQUEST, TurnOptions, run_turn
from app.db.models import Message, ToolLog
from app.llm.client import LLMRateLimitError
from app.services import expenses

from conftest import TODAY
from fake_llm import FakeLLM, call, respond

FOOD = "أكل وشرب"
INVENTED = "صرفت 5,000 وبتوفر 777"
GOOD = "صرفت 5,000"


def turn(session, ctx, text, responses, **kwargs):
    llm = FakeLLM(responses)
    return run_turn(session, ctx, text, llm, model="fake", **kwargs), llm


def with_total(session, ctx):
    expenses.add_expense(session, ctx, amount=5000, category=FOOD)
    return respond(call("get_total", period="this_month"))


def repair_log(session):
    return session.scalars(select(ToolLog).where(ToolLog.tool_name == "grounding_repair")).all()


def test_first_reply_ungrounded_then_a_grounded_one(session, member_ctx):
    done, llm = turn(session, member_ctx, "كم؟", [with_total(session, member_ctx), respond(content=INVENTED),
                                                  respond(content=GOOD)])
    assert done.reply == GOOD and done.ended_by == "reply" and done.llm_steps == 3 and done.repaired
    assert (done.first_reply, done.first_ungrounded, done.ungrounded) == (INVENTED, ["777"], [])
    request = llm.requests[2]
    assert request[-2] == {"role": "assistant", "content": INVENTED}
    assert request[-1] == {"role": "user", "content": REPAIR_REQUEST.format(numbers="777")}
    assert "Your reply has numbers that are not in the tool results: 777." in request[-1]["content"]
    assert session.scalars(select(Message).order_by(Message.id.desc())).first().content == GOOD

    log = repair_log(session)[0]
    assert log.status == "ok"
    assert log.arguments == {"first_reply": INVENTED, "first_ungrounded": ["777"], "ungrounded_after": []}
    assert session.scalars(select(ToolLog).where(ToolLog.tool_name == "grounding")).all() == []


def test_the_second_reply_stands_even_if_still_ungrounded(session, member_ctx):
    second = "صرفت 5,000 وبتوفر 888"
    done, llm = turn(session, member_ctx, "كم؟", [with_total(session, member_ctx), respond(content=INVENTED),
                                                  respond(content=second)])
    assert done.reply == second and done.ungrounded == ["888"] and len(llm.requests) == 3
    log = repair_log(session)[0]
    assert log.status == "ungrounded" and log.arguments["ungrounded_after"] == ["888"]
    assert session.scalars(select(ToolLog).where(ToolLog.tool_name == "grounding")).one().arguments == {"numbers": ["888"]}


def test_only_one_attempt_per_turn(session, member_ctx):
    done, llm = turn(session, member_ctx, "كم؟", [
        with_total(session, member_ctx), respond(content=INVENTED), respond(content=INVENTED),
        respond(content="لا يجب أن يصل")])
    assert len(llm.requests) == 3 and done.reply == INVENTED and len(llm.responses) == 1


def test_switched_off(session, member_ctx):
    done, llm = turn(session, member_ctx, "كم؟", [with_total(session, member_ctx), respond(content=INVENTED)],
                     options=TurnOptions(grounding_repair=False))
    assert len(llm.requests) == 2 and done.reply == INVENTED and done.ungrounded == ["777"] and not done.repaired
    assert repair_log(session) == []


def test_the_model_may_call_a_tool_during_the_repair(session, member_ctx):
    expenses.add_expense(session, member_ctx, amount=777, category=FOOD)
    done, llm = turn(session, member_ctx, "كم؟", [
        respond(content="صرفت 777"), respond(call("get_total", period="this_month")), respond(content="صرفت 777")])
    assert done.reply == "صرفت 777" and done.ungrounded == [] and [s.tool for s in done.steps] == ["get_total"]
    assert len(llm.requests) == 3


def test_the_attempt_counts_in_the_step_limit(session, member_ctx):
    done, llm = turn(session, member_ctx, "كم؟", [
        respond(content=INVENTED), respond(call("get_total", period="this_month")),
        respond(call("get_total", period="this_month"))], max_steps=3)
    assert len(llm.requests) == 3 and done.llm_steps == 3


def test_no_attempt_on_the_last_step(session, member_ctx):
    done, llm = turn(session, member_ctx, "كم؟", [with_total(session, member_ctx), respond(content=INVENTED)], max_steps=2)
    assert done.reply == INVENTED and not done.repaired and done.ungrounded == ["777"] and len(llm.requests) == 2


def test_the_first_reply_survives_a_failed_repair(session, member_ctx):
    done, _ = turn(session, member_ctx, "كم؟", [with_total(session, member_ctx), respond(content=INVENTED),
                                                LLMRateLimitError("slow down", 429)])
    assert (done.reply, done.ended_by, done.error) == (INVENTED, "reply", None) and done.ungrounded == ["777"]


def test_the_first_reply_survives_the_step_limit(session, member_ctx):
    done, _ = turn(session, member_ctx, "كم؟", [
        respond(content=INVENTED), respond(call("get_total", period="this_month"))], max_steps=2)
    assert (done.reply, done.ended_by) == (INVENTED, "reply")


def test_a_clarification_can_end_the_repair(session, member_ctx):
    done, _ = turn(session, member_ctx, "كم؟", [
        respond(content=INVENTED), respond(call("ask_clarification", question="أي شهر؟", missing_fields=["period"]))])
    assert done.ended_by == "clarification" and done.reply == "أي شهر؟" and done.ungrounded == []


def test_no_attempt_without_numbers(session, member_ctx):
    done, llm = turn(session, member_ctx, "مرحبا", [respond(content="أهلا فيك")])
    assert len(llm.requests) == 1 and not done.repaired


# ---------- numbers from the user's message ----------

def test_a_number_the_user_said_is_not_invented(session, member_ctx):
    done, llm = turn(session, member_ctx, "إذا صرفت 777 شو بصير؟", [respond(content="إذا صرفت 777 بينقص رصيدك")])
    assert done.ungrounded == [] and not done.repaired and len(llm.requests) == 1


def test_only_the_current_message_counts(session, member_ctx):
    turn(session, member_ctx, "صرفت 777", [respond(content="أهلا")])
    done, _ = turn(session, member_ctx, "كم؟", [respond(content="777"), respond(content="ما بعرف")])
    assert done.repaired and done.first_ungrounded == ["777"]


def test_user_numbers_in_the_grounding_check():
    assert ungrounded_numbers("دفعت 15000", [], "فاتورة 15 ألف") == []
    assert ungrounded_numbers("دفعت 15 ألف", [], "١٥٬٠٠٠ كهربا") == []
    assert ungrounded_numbers("زاد 20%", [], "إذا زاد 20% شو بصير") == []
    assert ungrounded_numbers("بسنة 2025", [], "قارن مع 2025") == []
    assert ungrounded_numbers("دفعت 16000", [], "فاتورة 15 ألف") == [Decimal(16000)]
    assert ungrounded_numbers("زاد 20", [], "إذا زاد 20% شو بصير") == [Decimal(20)]
    assert ungrounded_numbers("دفعت 777", []) == [Decimal(777)]


def test_a_wrong_year_is_corrected_without_a_repair(session, member_ctx):
    expenses.add_expense(session, member_ctx, amount=5000, category=FOOD)
    right = TODAY.isoformat()
    wrong = f"{TODAY.year - 2}{right[4:]}"
    result, llm = turn(session, member_ctx, "شو آخر مصروف؟", [
        respond(call("list_expenses", period="this_month")), respond(content=f"آخر مصروف 5,000 بتاريخ {wrong}")])
    assert len(llm.requests) == 2 and not result.repaired and not result.ungrounded
    assert result.reply.endswith(right) and result.year_corrected == [{"from": wrong, "to": right}]
    logged = session.scalars(select(ToolLog).where(ToolLog.tool_name == "year_corrected")).one()
    assert logged.arguments == {"dates": [{"from": wrong, "to": right}]}
