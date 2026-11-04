"""Conversation memory (§6.8), tool_summary with expense ids, and linked clarification (§6.6)."""
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.agent.loop import TurnOptions, run_turn
from app.agent.memory import MAX_REPLY_CHARS, memory_for_model, recent_messages
from app.db.models import Expense, Message, utcnow
from app.llm.client import estimate_prompt_tokens
from app.settings import Settings

from conftest import FakeEmbedder, FakeVectorStore, ctx_for, make_family, make_member
from fake_llm import FakeLLM, call, respond

FOOD = "أكل وشرب"


def turn(session, ctx, text, responses, options=TurnOptions(), **kwargs):
    llm = FakeLLM(responses)
    result = run_turn(session, ctx, text, llm, model="fake", options=options, **kwargs)
    return result, llm


def store(session, ctx, role, content, summary=None, ended_by=None, age_hours=0):
    row = Message(family_id=ctx.family_id, member_id=ctx.member_id, role=role, content=content, tool_summary=summary,
                  ended_by=ended_by, created_at=utcnow() - timedelta(hours=age_hours))
    session.add(row)
    session.commit()
    return row


def sent(llm, index=0):
    """What request `index` carried between the system prompt and the new message."""
    return llm.requests[index][1:-1]


# ---------- what the model sees ----------

def test_memory_sits_between_the_system_prompt_and_the_new_message(session, member_ctx):
    store(session, member_ctx, "user", "أول")
    store(session, member_ctx, "assistant", "رد أول", "get_total → ok", "reply")
    store(session, member_ctx, "user", "تاني")
    store(session, member_ctx, "assistant", "رد تاني", None, "reply")
    _, llm = turn(session, member_ctx, "جديد", [respond(content="تم")])
    request = llm.requests[0]
    assert request[0]["role"] == "system" and request[-1] == {"role": "user", "content": "جديد"}
    assert request[1:-1] == [{"role": "user", "content": "أول"},
                             {"role": "assistant", "content": "رد أول (get_total → ok)"},
                             {"role": "user", "content": "تاني"},
                             {"role": "assistant", "content": "رد تاني"}]


def test_memory_is_the_members_own_only(session, member_ctx, other_member_ctx, manager_ctx):
    store(session, other_member_ctx, "user", "سر الفرد الآخر")
    store(session, manager_ctx, "user", "سر المسؤول")
    other_family = make_member(session, make_family(session), "stranger", "manager")
    store(session, ctx_for(other_family), "user", "عائلة أخرى")
    store(session, member_ctx, "user", "رسالتي")
    _, llm = turn(session, member_ctx, "جديد", [respond(content="تم")])
    assert sent(llm) == [{"role": "user", "content": "رسالتي"}]


def test_memory_covers_the_last_24_hours_only(session, member_ctx):
    store(session, member_ctx, "user", "قديمة", age_hours=25)
    store(session, member_ctx, "user", "حديثة", age_hours=23)
    _, llm = turn(session, member_ctx, "جديد", [respond(content="تم")])
    assert [m["content"] for m in sent(llm)] == ["حديثة"]


def test_memory_keeps_the_last_ten_in_order(session, member_ctx):
    for n in range(14):
        store(session, member_ctx, "user", f"رسالة {n}")
    _, llm = turn(session, member_ctx, "جديد", [respond(content="تم")])
    assert [m["content"] for m in sent(llm)] == [f"رسالة {n}" for n in range(4, 14)]


def test_memory_settings_are_used(session, member_ctx):
    for n in range(5):
        store(session, member_ctx, "user", f"رسالة {n}", age_hours=n * 2)
    _, llm = turn(session, member_ctx, "جديد", [respond(content="تم")], TurnOptions(memory_messages=3, memory_hours=5))
    assert [m["content"] for m in sent(llm)] == ["رسالة 0", "رسالة 1", "رسالة 2"]


def test_token_ceiling_drops_the_oldest_first(session, member_ctx):
    for n in range(6):
        store(session, member_ctx, "user", f"{n} " + "كلمة طويلة " * 30)
    _, llm = turn(session, member_ctx, "جديد", [respond(content="تم")], TurnOptions(memory_max_tokens=250))
    kept = sent(llm)
    assert 0 < len(kept) < 6 and estimate_prompt_tokens(kept) <= 250
    assert kept[-1]["content"].startswith("5 "), "the newest message stays"
    assert [m["content"][0] for m in kept] == sorted(m["content"][0] for m in kept)


def test_a_single_huge_message_cannot_pass_the_ceiling(session, member_ctx):
    store(session, member_ctx, "user", "ك" * 1000)
    assert memory_for_model(recent_messages(session, member_ctx, 10, 24), 100) == []


def test_long_assistant_replies_are_cut_but_keep_their_summary(session, member_ctx):
    store(session, member_ctx, "assistant", "ر" * 500, "list_expenses → #3", "reply")
    store(session, member_ctx, "user", "ك" * 500)
    _, llm = turn(session, member_ctx, "جديد", [respond(content="تم")])
    reply, user = sent(llm)
    assert reply["content"] == "ر" * MAX_REPLY_CHARS + "… (list_expenses → #3)"
    assert user["content"] == "ك" * 500, "only assistant replies are cut"


def test_memory_can_be_switched_off(session, member_ctx):
    store(session, member_ctx, "user", "قديمة")
    _, llm = turn(session, member_ctx, "جديد", [respond(content="تم")], TurnOptions(memory_messages=0))
    assert sent(llm) == []


def test_options_come_from_the_settings():
    options = TurnOptions.from_settings(Settings(_env_file=None, memory_messages=0, memory_hours=2, memory_max_tokens=9,
                                                 grounding_repair=False))
    assert options == TurnOptions(0, 2, 9, False)
    assert TurnOptions.from_settings(Settings(_env_file=None)) == TurnOptions(10, 24, 600, True)


def test_a_turn_remembers_the_turn_before_it(session, member_ctx):
    turn(session, member_ctx, "كم صرفنا؟", [respond(call("get_total", period="this_month")), respond(content="صفر")])
    _, llm = turn(session, member_ctx, "وبالشهر الماضي؟", [respond(content="تم")])
    assert sent(llm) == [{"role": "user", "content": "كم صرفنا؟"},
                         {"role": "assistant", "content": "صفر (get_total → ok)"}]


# ---------- tool_summary ----------

def summary(session):
    return session.scalars(select(Message).where(Message.role == "assistant")).all()[-1].tool_summary


def add(session, ctx, n):
    from app.services.expenses import add_expense
    return [add_expense(session, ctx, amount=100 + i, category=FOOD)["id"] for i in range(n)]


def test_summary_names_added_expenses(session, member_ctx):
    turn(session, member_ctx, "x", [respond(call("add_expense", amount=5, category=FOOD),
                                            call("add_expense", amount=6, category=FOOD)), respond(content="تم")])
    assert summary(session) == "add_expense → #1; add_expense → #2"


def test_summary_of_a_list_has_at_most_five_ids_newest_first(session, member_ctx):
    ids = add(session, member_ctx, 7)
    turn(session, member_ctx, "x", [respond(call("list_expenses")), respond(content="تم")])
    assert summary(session) == "list_expenses → " + ", ".join(f"#{i}" for i in reversed(ids[2:]))


def test_summary_of_a_search_names_the_matches(session, member_ctx):
    embedder, vectors = FakeEmbedder(), FakeVectorStore()
    from app.services.expenses import add_expense
    found = add_expense(session, member_ctx, amount=5, category=FOOD, embedder=embedder, vector_store=vectors)
    turn(session, member_ctx, "x", [respond(call("search_expenses", query="خبز")), respond(content="تم")],
         embedder=embedder, vector_store=vectors)
    assert summary(session) == f"search_expenses → #{found['id']}"


@pytest.mark.parametrize("tool, expected", [("list_expenses", "ok"), ("get_total", "ok")])
def test_summary_without_ids_keeps_the_status(session, member_ctx, tool, expected):
    arguments = {"period": "this_month"}
    turn(session, member_ctx, "x", [respond(call(tool, **arguments)), respond(content="تم")])
    assert summary(session) == f"{tool} → {expected}"


def test_summary_of_a_refused_call_keeps_the_status(session, member_ctx):
    turn(session, member_ctx, "x", [respond(call("list_expenses", member="manager1")), respond(content="لا")])
    assert summary(session) == "list_expenses → forbidden"


def test_outcome_is_stored_with_assistant_messages_only(session, member_ctx):
    turn(session, member_ctx, "x", [respond(content="تم")])
    user, assistant = session.scalars(select(Message).order_by(Message.id)).all()
    assert (user.ended_by, assistant.ended_by) == (None, "reply")


# ---------- linked clarification ----------

QUESTION = respond(call("ask_clarification", question="على شو دفعت الـ 50؟", missing_fields=["category"]))


def test_the_answer_reaches_the_model_with_the_question_and_the_request(session, member_ctx):
    first, _ = turn(session, member_ctx, "دفعت 50 لأبو أحمد", [QUESTION])
    assert first.ended_by == "clarification"
    assert session.scalars(select(Message).order_by(Message.id.desc())).first().ended_by == "clarification"

    second, llm = turn(session, member_ctx, "كانت حق خضرا", [
        respond(call("add_expense", amount=50, category=FOOD)), respond(content="تم")])
    assert sent(llm) == [{"role": "user", "content": "دفعت 50 لأبو أحمد"},
                         {"role": "assistant", "content": "على شو دفعت الـ 50؟ (ask_clarification → ok)"}]
    assert second.ended_by == "reply"
    assert session.scalars(select(Expense)).one().source_text == "دفعت 50 لأبو أحمد ← كانت حق خضرا"


def test_two_questions_in_a_row_keep_the_whole_chain(session, member_ctx):
    turn(session, member_ctx, "دفعت 50", [QUESTION])
    turn(session, member_ctx, "لأبو أحمد", [
        respond(call("ask_clarification", question="شو الفئة؟", missing_fields=["category"]))])
    turn(session, member_ctx, "كانت حق خضرا", [respond(call("add_expense", amount=50, category=FOOD)),
                                               respond(content="تم")])
    assert session.scalars(select(Expense)).one().source_text == "دفعت 50 ← لأبو أحمد ← كانت حق خضرا"


def test_source_text_is_plain_without_a_clarification(session, member_ctx):
    turn(session, member_ctx, "كم صرفنا؟", [respond(content="صفر")])
    turn(session, member_ctx, "دفعت 50 خبز", [respond(call("add_expense", amount=50, category=FOOD)),
                                              respond(content="تم")])
    assert session.scalars(select(Expense)).one().source_text == "دفعت 50 خبز"


def test_source_text_is_plain_when_memory_is_off(session, member_ctx):
    turn(session, member_ctx, "دفعت 50", [QUESTION])
    turn(session, member_ctx, "خبز", [respond(call("add_expense", amount=50, category=FOOD)), respond(content="تم")],
         TurnOptions(memory_messages=0))
    assert session.scalars(select(Expense)).one().source_text == "خبز"


def test_another_members_question_is_not_linked(session, member_ctx, other_member_ctx):
    turn(session, member_ctx, "دفعت 50", [QUESTION])
    turn(session, other_member_ctx, "خبز", [respond(call("add_expense", amount=50, category=FOOD)),
                                            respond(content="تم")])
    assert session.scalars(select(Expense)).one().source_text == "خبز"


def test_context_day_does_not_change_the_memory_window(session, member_ctx):
    """The window uses the real clock: AGENT_TODAY only fixes dates for the tools."""
    store(session, member_ctx, "user", "حديثة")
    _, llm = turn(session, replace(member_ctx, today=member_ctx.today - timedelta(days=40)), "جديد",
                  [respond(content="تم")])
    assert [m["content"] for m in sent(llm)] == ["حديثة"]
