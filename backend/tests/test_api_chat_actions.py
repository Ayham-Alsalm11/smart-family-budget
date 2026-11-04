"""Pending actions and POST /chat with a scripted model: no Groq in these tests."""
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.db.models import Expense, LlmCall, Member, Message, PendingAction, utcnow
from app.llm.client import LLMRateLimitError

from api_support import assert_error, ask_to_change, join, make_api, register, spend
from fake_llm import FakeLLM, call, respond


@pytest.fixture
def llm():
    return FakeLLM([])


@pytest.fixture
def api(tmp_path, monkeypatch, llm):
    with make_api(tmp_path, monkeypatch, llm=llm) as client:
        yield client


def action_status(api, action_id):
    with api.app.state.session_factory() as session:
        return session.get(PendingAction, action_id).status


def expense_row(api, expense_id):
    with api.app.state.session_factory() as session:
        return session.get(Expense, expense_id)


def confirm(api, action, who):
    return api.post(f"/actions/{action}/confirm", headers=who.headers)


def cancel(api, action, who):
    return api.post(f"/actions/{action}/cancel", headers=who.headers)


# ---------- actions ----------

def test_confirm_runs_a_deletion(api):
    manager = register(api, "sami")
    expense = spend(api, manager)
    action = ask_to_change(api, manager, expense)
    response = confirm(api, action, manager)
    assert response.status_code == 200
    assert response.json()["status"] == "confirmed" and response.json()["action"] == "delete_expense"
    assert expense_row(api, expense) is None and action_status(api, action) == "confirmed"


def test_confirm_runs_an_edit(api):
    manager = register(api, "sami")
    expense = spend(api, manager, 1000)
    action = ask_to_change(api, manager, expense, {"amount": 6000})
    assert confirm(api, action, manager).json()["expense"]["amount"] == 6000
    assert expense_row(api, expense).amount == 6000


def test_confirm_by_another_member_is_refused(api):
    manager = register(api, "sami")
    member, other = join(api, manager, "rana"), join(api, manager, "omar")
    expense = spend(api, member)
    action = ask_to_change(api, member, expense)
    assert_error(confirm(api, action, other), 403, "forbidden")
    assert_error(confirm(api, action, manager), 403, "forbidden")
    assert_error(cancel(api, action, other), 403, "forbidden")
    assert expense_row(api, expense) is not None and action_status(api, action) == "pending"
    assert confirm(api, action, member).status_code == 200


def test_expired_action_becomes_expired(api):
    manager = register(api, "sami")
    expense = spend(api, manager)
    action = ask_to_change(api, manager, expense)
    with api.app.state.session_factory() as session:
        session.get(PendingAction, action).expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
    assert_error(confirm(api, action, manager), 409, "conflict")
    assert action_status(api, action) == "expired" and expense_row(api, expense) is not None
    assert_error(cancel(api, action, manager), 409, "conflict")


def test_second_confirm_is_a_conflict(api):
    manager = register(api, "sami")
    action = ask_to_change(api, manager, spend(api, manager, 1000), {"amount": 2000})
    assert confirm(api, action, manager).status_code == 200
    assert_error(confirm(api, action, manager), 409, "conflict")
    assert_error(cancel(api, action, manager), 409, "conflict")


def test_cancel(api):
    manager = register(api, "sami")
    expense = spend(api, manager)
    action = ask_to_change(api, manager, expense)
    response = cancel(api, action, manager)
    assert response.status_code == 200 and response.json()["status"] == "cancelled"
    assert expense_row(api, expense) is not None and action_status(api, action) == "cancelled"
    assert_error(confirm(api, action, manager), 409, "conflict")


def test_unknown_action(api):
    manager = register(api, "sami")
    assert_error(confirm(api, 9999, manager), 404, "not_found")


def test_permission_is_checked_again_at_confirmation(api):
    """A manager asks to delete a member's expense, then is demoted: the action no longer runs."""
    manager = register(api, "sami")
    second = join(api, manager, "rana", "manager")
    member = join(api, manager, "omar")
    expense = spend(api, member)
    action = ask_to_change(api, second, expense)
    api.put(f"/members/{second.id}/role", json={"role": "member"}, headers=manager.headers)
    assert_error(confirm(api, action, second), 403, "forbidden")
    assert expense_row(api, expense) is not None and action_status(api, action) == "pending"


def test_a_failed_execution_leaves_the_action_pending(api):
    manager = register(api, "sami")
    expense = spend(api, manager)
    action = ask_to_change(api, manager, expense)
    api.delete(f"/expenses/{expense}", headers=manager.headers)
    assert_error(confirm(api, action, manager), 404, "not_found")
    assert action_status(api, action) == "pending"


def test_confirmation_id_in_a_tool_call_stays_not_available(api, llm):
    """Confirmation in chat is step 6: the tools must not run an action yet."""
    manager = register(api, "sami")
    expense = spend(api, manager)
    action = ask_to_change(api, manager, expense)
    llm.responses = [respond(call("delete_expense", expense_id=expense, confirmation_id=action)),
                     respond(content="بعدين")]
    steps = api.post("/chat", json={"message": "نعم"}, headers=manager.headers).json()["steps"]
    assert steps[0]["status"] == "not_available"
    assert expense_row(api, expense) is not None and action_status(api, action) == "pending"


# ---------- chat ----------

def test_chat_returns_reply_steps_alerts_and_message_id(api, llm):
    manager = register(api, "sami")
    llm.responses = [respond(call("add_expense", amount=120000, category="أكل وشرب")),
                     respond(content="سجلت 120000.")]
    response = api.post("/chat", json={"message": "جبت أكل بـ 120 ألف"}, headers=manager.headers)
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "سجلت 120000."
    assert [(s["tool"], s["status"]) for s in body["steps"]] == [("add_expense", "ok")]
    assert set(body["steps"][0]) == {"step", "tool", "arguments", "status", "latency_ms"}
    assert body["alerts"][0]["status"] == "over" and "pending_action" not in body
    with api.app.state.session_factory() as session:
        stored = session.get(Message, body["message_id"])
        assert stored.member_id == manager.id and stored.content == "جبت أكل بـ 120 ألف"
        assert session.scalar(select(LlmCall.message_id)) == body["message_id"]


def test_chat_stores_cached_tokens(api, llm):
    manager = register(api, "sami")
    response = respond(content="أهلا")
    response.usage.cached_tokens = 1280
    llm.responses = [response]
    api.post("/chat", json={"message": "مرحبا"}, headers=manager.headers)
    with api.app.state.session_factory() as session:
        assert session.scalar(select(LlmCall.cached_tokens)) == 1280


def test_chat_returns_the_pending_action_and_the_button_runs_it(api, llm):
    manager = register(api, "sami")
    expense = spend(api, manager, 5000, "بنزين ومواصلات")
    llm.responses = [respond(call("delete_expense", expense_id=expense)), respond(content="أكيد بدك تمسحه؟")]
    body = api.post("/chat", json={"message": "امسح البنزين"}, headers=manager.headers).json()
    pending = body["pending_action"]
    assert pending["action"] == "delete_expense" and pending["expense"]["id"] == expense
    assert expense_row(api, expense) is not None, "nothing runs before the confirmation"
    assert confirm(api, pending["confirmation_id"], manager).status_code == 200
    assert expense_row(api, expense) is None


def test_chat_context_comes_from_the_token_only(api, llm):
    first, second = register(api, "sami"), register(api, "omar")
    llm.responses = [respond(call("add_expense", amount=1000, category="صحة")), respond(content="تم")]
    body = {"message": "دوا", "family_id": second.family_id, "member_id": second.id, "role": "manager"}
    assert api.post("/chat", json=body, headers=first.headers).status_code == 200
    with api.app.state.session_factory() as session:
        row = session.scalar(select(Expense))
        assert (row.family_id, row.member_id) == (first.family_id, first.id)
        message = session.scalar(select(Message))
        assert (message.family_id, message.member_id) == (first.family_id, first.id)


def test_chat_role_comes_from_the_database(api, llm):
    manager = register(api, "sami")
    member = join(api, manager, "rana")
    spend(api, manager)
    llm.responses = [respond(call("get_total", period="this_month", member="sami")), respond(content="هاد للمسؤول")]
    body = api.post("/chat", json={"message": "كم صرف سامي؟", "role": "manager"}, headers=member.headers).json()
    assert body["steps"][0]["status"] == "forbidden"


def test_chat_uses_the_configured_model_and_todays_setting(tmp_path, monkeypatch, llm):
    with make_api(tmp_path, monkeypatch, llm=llm, agent_model="small") as api:
        manager = register(api, "sami")
        llm.responses = [respond(content="أهلا")]
        api.post("/chat", json={"message": "مرحبا"}, headers=manager.headers)
        with api.app.state.session_factory() as session:
            assert session.scalar(select(LlmCall.model)) == api.app.state.settings.model_small
        assert "2026-11-04" in llm.requests[0][0]["content"]


@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "   "}, {"message": "x" * 1001}])
def test_chat_message_limits(api, llm, body):
    manager = register(api, "sami")
    assert api.post("/chat", json=body, headers=manager.headers).status_code == 422
    assert llm.requests == []


def test_chat_longest_message(api, llm):
    manager = register(api, "sami")
    llm.responses = [respond(content="طويلة")]
    assert api.post("/chat", json={"message": "x" * 1000}, headers=manager.headers).status_code == 200


def test_chat_without_a_groq_key(tmp_path, monkeypatch):
    with make_api(tmp_path, monkeypatch) as api:
        manager = register(api, "sami")
        error = assert_error(api.post("/chat", json={"message": "مرحبا"}, headers=manager.headers), 503, "not_available")
        assert "Groq" in error["message"]
        assert api.get("/expenses", headers=manager.headers).status_code == 200, "the rest of the API still works"
        with api.app.state.session_factory() as session:
            assert session.scalar(select(Message)) is None


def test_chat_survives_a_model_failure(api, llm):
    manager = register(api, "sami")
    llm.responses = [LLMRateLimitError("slow down", 429)]
    response = api.post("/chat", json={"message": "مرحبا"}, headers=manager.headers)
    assert response.status_code == 200 and response.json()["reply"]


def test_one_client_serves_all_requests(api, llm):
    first, second = register(api, "sami"), register(api, "omar")
    llm.responses = [respond(content="أهلا"), respond(content="مرحبا")]
    api.post("/chat", json={"message": "مرحبا"}, headers=first.headers)
    api.post("/chat", json={"message": "مرحبا"}, headers=second.headers)
    assert api.app.state.llm is llm and len(llm.requests) == 2


def test_app_builds_its_own_shared_client_from_the_key(tmp_path, monkeypatch):
    from pydantic import SecretStr

    from app.llm.client import GroqClient

    with make_api(tmp_path, monkeypatch, groq_api_key=SecretStr("gsk_test")) as api:
        assert isinstance(api.app.state.llm, GroqClient)


def test_messages_order_limit_and_default(api, llm):
    manager = register(api, "sami")
    with api.app.state.session_factory() as session:
        member = session.get(Member, manager.id)
        for n in range(60):
            session.add(Message(family_id=member.family_id, member_id=member.id, role="user", content=str(n)))
        session.commit()
    rows = api.get("/messages", headers=manager.headers).json()["messages"]
    assert [m["content"] for m in rows] == [str(n) for n in range(10, 60)], "the last 50, oldest first"
    assert len(api.get("/messages", params={"limit": 5}, headers=manager.headers).json()["messages"]) == 5
    assert_error(api.get("/messages", params={"limit": 0}, headers=manager.headers), 422, "invalid_args")
