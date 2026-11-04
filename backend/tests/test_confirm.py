"""Confirmation without the model (§6.5): the classifier, the fast path and the server-written preview."""
import io
import json
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlalchemy import select

from app import cli
from app.agent import confirm
from app.agent.confirm import classify, normalise, preview_text
from app.agent.loop import TurnOptions, run_turn
from app.db.models import Expense, LlmCall, Message, PendingAction, ToolLog, utcnow
from app.services import expenses, pending
from app.services.errors import ConflictError
from app.tools.schemas import TOOL_SCHEMAS

from api_support import join, make_api, register
from conftest import TODAY, FakeEmbedder, FakeVectorStore
from fake_llm import FakeLLM, call, respond
from test_cli import seeded

FOOD = "أكل وشرب"
FUEL = "بنزين ومواصلات"


@pytest.mark.parametrize("text, expected", [
    # yes
    ("اي", "yes"), ("إي", "yes"), ("أي", "yes"), ("ايه", "yes"), ("إيه", "yes"), ("نعم", "yes"), ("أكيد", "yes"),
    ("اكيد", "yes"), ("تمام", "yes"), ("موافق", "yes"), ("أكّد", "yes"), ("أَكِّد", "yes"), ("اوكي", "yes"),
    ("ok", "yes"), ("OK", "yes"), ("Yes", "yes"), ("yes!", "yes"), ("اي.", "yes"), ("اي،", "yes"), ("«اي»", "yes"),
    ("نَعَم", "yes"), ("أكـــيد", "yes"), ("اي امسحه", "yes"), ("اي اكيد تمام", "yes"), ("نعم عدلها يا ميزان", "yes"),
    # no
    ("لا", "no"), ("لأ", "no"), ("لاء", "no"), ("لا.", "no"), ("لا خليه", "no"), ("الغي", "no"), ("ألغي", "no"),
    ("إلغي", "no"), ("بلاش", "no"), ("خليه", "no"), ("no", "no"), ("NO!", "no"), ("لا تمسحه", "no"),
    ("لَا", "no"), ("لا لا بلاش", "no"),
    # mixed
    ("اي لا", None), ("لا اي", None), ("نعم بلاش", None), ("تمام الغي", None), ("yes no", None),
    # with a number
    ("اي 5000", None), ("لا 5000", None), ("نعم ٣", None), ("لا خليه ١٠٠", None), ("ok 2", None),
    # too long
    ("اي بس لما ترجع من الشغل", None), ("لا ما بدي ياه هلق بعدين", None), ("نعم امسحه هلق لو سمحت", None),
    # negated
    ("مو موافق", None), ("مش موافق", None), ("مو تمام", None), ("ما اكيد", None), ("غير موافق", None),
    # neither
    ("", None), ("   ", None), ("؟؟؟", None), ("👍", None), ("شو يعني", None), ("كم صرفنا اليوم؟", None), ("امسحه", None),
    ("لاحقا", None), ("ايش", None), ("نعمة", None),
])
def test_classify(text, expected):
    assert classify(text) == expected


def test_normalise():
    assert normalise("  أَكّد!!  إي، آه؟ ") == "اكد اي اه"
    assert normalise("لـــأ") == "لا"
    assert normalise("OK_Yes") == "ok yes"


def test_every_listed_word_survives_normalisation():
    """A word list in another form would silently never match."""
    for word in confirm.YES_WORDS | confirm.NO_WORDS | confirm.NEGATIONS:
        assert normalise(word) == word


# ---------- helpers ----------

def turn(session, ctx, text, responses=(), **kwargs):
    llm = FakeLLM(responses)
    result = run_turn(session, ctx, text, llm, model="fake", **kwargs)
    return result, llm


def ask_delete(session, ctx, expense_id):
    return turn(session, ctx, "امسحه", [respond(call("delete_expense", expense_id=expense_id))])


def ask_edit(session, ctx, expense_id, **changes):
    return turn(session, ctx, "عدّله", [respond(call("edit_expense", expense_id=expense_id, changes=changes))])


def add(session, ctx, amount=5000, category=FUEL, description="تعبئة"):
    return expenses.add_expense(session, ctx, amount=amount, category=category, description=description,
                                expense_date=TODAY)["id"]


def status(session, action_id):
    session.expire_all()
    return session.get(PendingAction, action_id).status


def last_log(session):
    return session.scalars(select(ToolLog).order_by(ToolLog.id.desc())).first()


# ---------- the answer runs without the model ----------

def test_yes_deletes_without_any_model_call(session, member_ctx):
    expense = add(session, member_ctx)
    first, _ = ask_delete(session, member_ctx, expense)
    llm = FakeLLM([])
    done = run_turn(session, member_ctx, "اي", llm, model="fake")

    assert llm.requests == [] and done.llm_steps == 0
    assert done.ended_by == "confirmed" and done.reply.startswith("تم ✅ حذفت: 5,000 ليرة قديمة")
    assert session.get(Expense, expense) is None
    assert status(session, first.pending_action["confirmation_id"]) == "confirmed"

    user, reply = session.scalars(select(Message).order_by(Message.id.desc()).limit(2)).all()[::-1]
    assert (user.role, user.content, reply.role, reply.content) == ("user", "اي", "assistant", done.reply)
    assert (reply.ended_by, reply.tool_summary) == ("confirmed", "confirm_action → ok")
    log = last_log(session)
    assert (log.tool_name, log.status, log.step, log.message_id) == ("confirm_action", "ok", 0, user.id)
    assert log.arguments == {"action_id": first.pending_action["confirmation_id"], "tool": "delete_expense"}
    assert len(session.scalars(select(LlmCall)).all()) == 1, "only the turn that asked"


def test_yes_runs_an_edit_and_updates_the_vector_store(session, member_ctx):
    embedder, vectors = FakeEmbedder(), FakeVectorStore()
    expense = add(session, member_ctx)
    ask_edit(session, member_ctx, expense, amount=6000)
    done, llm = turn(session, member_ctx, "أكيد", embedder=embedder, vector_store=vectors)
    assert llm.requests == [] and done.reply.startswith("تم ✅ عدّلت: 6,000 ليرة قديمة")
    assert session.get(Expense, expense).amount == 6000
    assert ("expenses", str(expense)) in vectors.items


def test_no_cancels_without_any_model_call(session, member_ctx):
    expense = add(session, member_ctx)
    first, _ = ask_delete(session, member_ctx, expense)
    done, llm = turn(session, member_ctx, "لا خليه")
    assert llm.requests == [] and done.llm_steps == 0
    assert (done.ended_by, done.reply) == ("cancelled", confirm.REPLY_CANCELLED)
    assert session.get(Expense, expense) is not None
    assert status(session, first.pending_action["confirmation_id"]) == "cancelled"
    assert (last_log(session).tool_name, last_log(session).status) == ("cancel_action", "ok")


def test_a_turn_after_the_answer_goes_to_the_model_with_it_in_memory(session, member_ctx):
    ask_delete(session, member_ctx, add(session, member_ctx))
    turn(session, member_ctx, "اي")
    _, llm = turn(session, member_ctx, "شكرا", [respond(content="العفو")])
    assert [m["content"] for m in llm.requests[0][-3:-1]][0] == "اي"
    assert "تم ✅" in llm.requests[0][-2]["content"] and "confirm_action → ok" in llm.requests[0][-2]["content"]


# ---------- clear messages for what went wrong ----------

def test_expired_gets_a_clear_message(session, member_ctx):
    expense = add(session, member_ctx)
    first, _ = ask_delete(session, member_ctx, expense)
    action_id = first.pending_action["confirmation_id"]
    session.get(PendingAction, action_id).expires_at = utcnow() - timedelta(seconds=1)
    session.commit()
    done, llm = turn(session, member_ctx, "اي")
    assert llm.requests == [] and (done.ended_by, done.reply) == ("confirm_error", confirm.REPLY_EXPIRED)
    assert status(session, action_id) == "expired" and session.get(Expense, expense) is not None
    assert (last_log(session).tool_name, last_log(session).status) == ("confirm_action", "conflict")


def test_no_after_expiry_says_expired_too(session, member_ctx):
    first, _ = ask_delete(session, member_ctx, add(session, member_ctx))
    session.get(PendingAction, first.pending_action["confirmation_id"]).expires_at = utcnow() - timedelta(seconds=1)
    session.commit()
    done, _ = turn(session, member_ctx, "لا")
    assert (done.ended_by, done.reply) == ("confirm_error", confirm.REPLY_EXPIRED)
    assert last_log(session).tool_name == "cancel_action"


def test_expense_gone_gets_a_clear_message(session, member_ctx):
    expense = add(session, member_ctx)
    ask_delete(session, member_ctx, expense)
    expenses.delete_expense(session, member_ctx, expense)
    done, llm = turn(session, member_ctx, "اي")
    assert llm.requests == [] and (done.ended_by, done.reply) == ("confirm_error", confirm.REPLY_NOT_FOUND)
    assert last_log(session).status == "not_found"


def test_already_executed_gets_a_clear_message(session, member_ctx, monkeypatch):
    """Another request wins the race after this one found the action pending."""
    ask_delete(session, member_ctx, add(session, member_ctx))

    def lost_the_race(*_args):
        raise ConflictError("action is already confirmed")

    monkeypatch.setattr(confirm, "execute_action", lost_the_race)
    done, _ = turn(session, member_ctx, "اي")
    assert (done.ended_by, done.reply) == ("confirm_error", confirm.REPLY_ALREADY_DONE)


def test_lost_permission_gets_a_clear_message(session, manager_ctx, member_ctx):
    expense = add(session, member_ctx)
    ask_delete(session, manager_ctx, expense)
    done, _ = turn(session, replace(manager_ctx, role="member"), "اي")
    assert (done.ended_by, done.reply) == ("confirm_error", confirm.REPLY_FORBIDDEN)
    assert session.get(Expense, expense) is not None


def test_a_service_refusal_gets_a_clear_message(session, member_ctx, family):
    family.usd_rate = None
    session.commit()
    expense = add(session, member_ctx)
    ask_edit(session, member_ctx, expense, currency="USD")
    done, _ = turn(session, member_ctx, "اي")
    assert (done.ended_by, done.reply) == ("confirm_error", confirm.REPLY_FAILED)
    assert session.get(Expense, expense).currency == "SYP_OLD"


# ---------- when an answer is not an answer ----------

def test_any_message_in_between_ends_the_confirmation(session, member_ctx):
    expense = add(session, member_ctx)
    first, _ = ask_delete(session, member_ctx, expense)
    turn(session, member_ctx, "كم صرفنا اليوم؟", [respond(content="شوي")])
    done, llm = turn(session, member_ctx, "اي", [respond(content="شو قصدك؟")])
    assert len(llm.requests) == 1 and done.ended_by == "reply"
    assert session.get(Expense, expense) is not None
    assert status(session, first.pending_action["confirmation_id"]) == "pending"


@pytest.mark.parametrize("text", ["اي بس لما ترجع من الشغل", "اي 5000", "شو يعني", "اي لا"])
def test_anything_else_is_an_ordinary_turn_and_the_action_stays(session, member_ctx, text):
    expense = add(session, member_ctx)
    first, _ = ask_delete(session, member_ctx, expense)
    done, llm = turn(session, member_ctx, text, [respond(content="رد بـ اي")])
    assert len(llm.requests) == 1 and done.ended_by == "reply"
    assert status(session, first.pending_action["confirmation_id"]) == "pending"


def test_yes_with_nothing_to_confirm_goes_to_the_model(session, member_ctx):
    done, llm = turn(session, member_ctx, "اي", [respond(content="على شو؟")])
    assert len(llm.requests) == 1 and done.ended_by == "reply"


def test_yes_after_a_clarification_goes_to_the_model(session, member_ctx):
    turn(session, member_ctx, "x", [respond(call("ask_clarification", question="بدك؟", missing_fields=["x"]))])
    _, llm = turn(session, member_ctx, "اي", [respond(content="تمام")])
    assert len(llm.requests) == 1


def test_yes_after_the_button_was_used_goes_to_the_model(session, member_ctx):
    expense = add(session, member_ctx)
    first, _ = ask_delete(session, member_ctx, expense)
    pending.execute_action(session, member_ctx, first.pending_action["confirmation_id"])
    done, llm = turn(session, member_ctx, "اي", [respond(content="تمام")])
    assert len(llm.requests) == 1 and done.ended_by == "reply"


def test_another_member_cannot_answer(session, member_ctx, other_member_ctx, manager_ctx):
    expense = add(session, member_ctx)
    first, _ = ask_delete(session, member_ctx, expense)
    for ctx in (other_member_ctx, manager_ctx):
        done, llm = turn(session, ctx, "اي", [respond(content="على شو؟")])
        assert len(llm.requests) == 1 and done.ended_by == "reply"
    assert session.get(Expense, expense) is not None
    assert status(session, first.pending_action["confirmation_id"]) == "pending"


def test_each_member_confirms_their_own_action(session, member_ctx, other_member_ctx):
    mine, theirs = add(session, member_ctx), add(session, other_member_ctx)
    ask_delete(session, member_ctx, mine)
    ask_delete(session, other_member_ctx, theirs)
    turn(session, other_member_ctx, "اي")
    assert session.get(Expense, theirs) is None and session.get(Expense, mine) is not None


# ---------- one pending action per member ----------

def test_a_new_action_cancels_the_older_one(session, member_ctx):
    first_expense, second_expense = add(session, member_ctx), add(session, member_ctx)
    first, _ = ask_delete(session, member_ctx, first_expense)
    second, _ = ask_delete(session, member_ctx, second_expense)
    assert status(session, first.pending_action["confirmation_id"]) == "cancelled"
    assert status(session, second.pending_action["confirmation_id"]) == "pending"
    turn(session, member_ctx, "اي")
    assert session.get(Expense, second_expense) is None and session.get(Expense, first_expense) is not None


def test_a_new_action_leaves_other_members_alone(session, member_ctx, other_member_ctx):
    theirs = pending.request_change(session, other_member_ctx, "delete_expense", add(session, other_member_ctx))
    pending.request_change(session, member_ctx, "delete_expense", add(session, member_ctx))
    assert status(session, theirs["confirmation_id"]) == "pending"


def test_a_refused_request_keeps_the_older_action(session, member_ctx, other_member_ctx):
    mine = pending.request_change(session, member_ctx, "delete_expense", add(session, member_ctx))
    theirs = add(session, other_member_ctx)
    result, _ = turn(session, member_ctx, "امسحه", [respond(call("delete_expense", expense_id=theirs)),
                                                    respond(content="ما فيني")])
    assert result.pending_action is None and status(session, mine["confirmation_id"]) == "pending"


# ---------- the model has no way to confirm ----------

def test_schemas_have_no_confirmation_id():
    assert "confirmation_id" not in json.dumps(TOOL_SCHEMAS)


def test_confirmation_id_from_the_model_is_invalid(session, member_ctx):
    expense = add(session, member_ctx)
    first, _ = ask_delete(session, member_ctx, expense)
    action_id = first.pending_action["confirmation_id"]
    done, _ = turn(session, member_ctx, "نفّذه لو سمحت", [
        respond(call("delete_expense", expense_id=expense, confirmation_id=action_id)), respond(content="ما فيني")])
    assert done.steps[0].status == "invalid_args" and done.ended_by == "reply"
    assert session.get(Expense, expense) is not None and status(session, action_id) == "pending"


# ---------- the preview ----------

def test_confirmation_required_ends_the_turn_without_another_model_call(session, member_ctx):
    expense = add(session, member_ctx, 1250000, FOOD, "لحمة")
    done, llm = ask_delete(session, member_ctx, expense)
    assert len(llm.requests) == 1 and done.llm_steps == 1 and done.ended_by == "confirmation"
    assert done.reply == ("بدي احذف هالمصروف:\n1,250,000 ليرة قديمة · أكل وشرب · لحمة · 2026-11-04\n"
                          "رد بـ «اي» للتأكيد أو «لا» للإلغاء، خلال 10 دقائق.")
    assert done.pending_action["expense"]["id"] == expense
    assert session.scalars(select(Message).order_by(Message.id.desc())).first().ended_by == "confirmation"


def test_preview_of_an_edit_has_before_and_after(session, member_ctx):
    done, _ = ask_edit(session, member_ctx, add(session, member_ctx), amount=6000, category="متفرقات", date="2026-11-01",
                       description="زيت")
    assert "قبل: 5,000 ليرة قديمة · بنزين ومواصلات · تعبئة · 2026-11-04" in done.reply
    assert "بعد: 6,000 ليرة قديمة · متفرقات · زيت · 2026-11-01" in done.reply
    assert done.reply.endswith("خلال 10 دقائق.")


def test_preview_names_each_currency_in_arabic(session, member_ctx):
    expense = expenses.add_expense(session, member_ctx, amount=12.5, category=FOOD, currency="USD")["id"]
    done, _ = ask_edit(session, member_ctx, expense, currency="SYP_NEW")
    assert "12.50 دولار" in done.reply and "12.50 ليرة جديدة" in done.reply


def test_preview_without_a_description():
    text = preview_text({"action": "delete_expense", "changes": None, "expense": {
        "amount": 5000, "currency": "SYP_OLD", "category": FOOD, "description": "", "expense_date": "2026-11-04"}})
    assert "5,000 ليرة قديمة · أكل وشرب · 2026-11-04" in text


def test_the_grounding_check_does_not_touch_server_text(session, member_ctx):
    done, _ = ask_delete(session, member_ctx, add(session, member_ctx, 7777))
    assert done.ungrounded == [] and not done.repaired
    assert session.scalars(select(ToolLog).where(ToolLog.tool_name == "grounding")).all() == []


def test_calls_after_the_confirmation_in_the_same_step_do_not_run(session, member_ctx):
    first, second = add(session, member_ctx), add(session, member_ctx)
    done, _ = turn(session, member_ctx, "امسحهن", [
        respond(call("delete_expense", expense_id=first), call("delete_expense", expense_id=second))])
    assert len(done.steps) == 1 and done.pending_action["expense"]["id"] == first
    assert len(session.scalars(select(PendingAction)).all()) == 1


def test_earlier_calls_in_the_step_still_ran(session, member_ctx):
    expense = add(session, member_ctx)
    done, _ = turn(session, member_ctx, "x", [
        respond(call("add_expense", amount=10, category=FOOD), call("delete_expense", expense_id=expense))])
    assert [s.status for s in done.steps] == ["ok", "confirmation_required"]
    assert len(session.scalars(select(Expense)).all()) == 2


def test_memory_off_still_confirms(session, member_ctx):
    """MEMORY_MESSAGES=0 is for the model's context; the answer is read from the last assistant message."""
    expense = add(session, member_ctx)
    options = TurnOptions(memory_messages=0)
    ask_delete(session, member_ctx, expense)
    turn(session, member_ctx, "اي", options=options)
    assert session.get(Expense, expense) is None


# ---------- through the API and the terminal ----------

def test_chat_api_confirms_without_the_model(tmp_path, monkeypatch):
    llm = FakeLLM([])
    with make_api(tmp_path, monkeypatch, llm=llm) as api:
        manager = register(api, "sami")
        member = join(api, manager, "rana")
        with api.app.state.session_factory() as db:
            expense = expenses.add_expense(db, member.ctx, amount=5000, category=FUEL)["id"]
        llm.responses = [respond(call("delete_expense", expense_id=expense))]
        asked = api.post("/chat", json={"message": "امسحه"}, headers=member.headers).json()
        assert asked["pending_action"]["expense"]["id"] == expense and "«اي»" in asked["reply"]
        assert len(llm.requests) == 1

        llm.responses = [respond(content="على شو؟")]
        other = api.post("/chat", json={"message": "اي"}, headers=manager.headers).json()
        assert other["reply"] == "على شو؟" and len(llm.requests) == 2, "another member's yes goes to the model"
        llm.requests.clear()

        done = api.post("/chat", json={"message": "اي"}, headers=member.headers).json()
        assert done["reply"].startswith("تم ✅") and "pending_action" not in done
        assert [(s["tool"], s["status"]) for s in done["steps"]] == [("confirm_action", "ok")]
        assert llm.requests == []
        assert api.get("/expenses", headers=member.headers).json()["expenses"] == []


def run_cli(tmp_path, responses, lines):
    inputs, out = iter([*lines, "exit"]), io.StringIO()
    llm = FakeLLM(responses)
    code = cli.main(["--user", "manager", "--today", TODAY.isoformat(), "--db", seeded(tmp_path)], client=llm,
                    input_fn=lambda _prompt: next(inputs), out=out, rag=(None, None))
    assert code == 0
    return out.getvalue(), llm


def test_terminal_remembers_and_confirms_with_yes(tmp_path):
    text, llm = run_cli(tmp_path, [
        respond(call("list_expenses", limit=1)), respond(content="هاد آخر مصروف"),
        respond(call("delete_expense", expense_id=1)),
    ], ["شو آخر مصروف؟", "امسحه", "اي"])
    assert len(llm.requests) == 3, "the last line never reached the model"
    assert llm.requests[2][1]["content"] == "شو آخر مصروف؟" and "list_expenses → #" in llm.requests[2][2]["content"]
    assert "رد بـ «اي» للتأكيد" in text and "[confirmation]" in text
    assert "ميزان: تم ✅ حذفت: " in text and "[confirmed]" in text


def test_terminal_cancels_with_no(tmp_path):
    text, llm = run_cli(tmp_path, [respond(call("delete_expense", expense_id=1))], ["امسح المصروف", "لا"])
    assert len(llm.requests) == 1 and "ميزان: تمام، ما غيّرت شي." in text and "[cancelled]" in text
