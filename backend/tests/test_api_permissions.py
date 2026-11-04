"""The permission matrix of §5 and the isolation of families, through the API, with two families."""
from dataclasses import dataclass
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.db.models import Budget, Expense, Message

from api_support import Account, ask_to_change, assert_error, invite, join, make_api, register, spend
from conftest import TODAY

MONTH = TODAY.strftime("%Y-%m")


@dataclass
class Family:
    manager: Account
    member: Account
    other: Account


@pytest.fixture
def api(tmp_path, monkeypatch):
    with make_api(tmp_path, monkeypatch) as client:
        yield client


@pytest.fixture
def a(api) -> Family:
    manager = register(api, "manager_a")
    return Family(manager, join(api, manager, "member_a"), join(api, manager, "other_a"))


@pytest.fixture
def b(api) -> Family:
    manager = register(api, "manager_b")
    return Family(manager, join(api, manager, "member_b"), join(api, manager, "other_b"))


def expense_ids(response) -> list[int]:
    assert response.status_code == 200, response.text
    return [e["id"] for e in response.json()["expenses"]]


# ---------- seeing expenses ----------

def test_member_sees_only_own_expenses_manager_sees_all(api, a):
    mine, theirs, boss = spend(api, a.member), spend(api, a.other), spend(api, a.manager)
    assert expense_ids(api.get("/expenses", headers=a.member.headers)) == [mine]
    assert expense_ids(api.get("/expenses", headers=a.other.headers)) == [theirs]
    assert sorted(expense_ids(api.get("/expenses", headers=a.manager.headers))) == [mine, theirs, boss]


def test_member_filter(api, a):
    mine = spend(api, a.member)
    spend(api, a.other)
    response = api.get("/expenses", params={"member": a.other.id}, headers=a.member.headers)
    assert_error(response, 403, "forbidden")
    assert_error(api.get("/expenses", params={"member": "other_a"}, headers=a.member.headers), 403, "forbidden")
    assert_error(api.get("/expenses", params={"member": "nobody"}, headers=a.member.headers), 403, "forbidden")
    for own in ("me", a.member.id, "member_a"):
        assert expense_ids(api.get("/expenses", params={"member": own}, headers=a.member.headers)) == [mine]
    assert expense_ids(api.get("/expenses", params={"member": a.member.id}, headers=a.manager.headers)) == [mine]
    assert expense_ids(api.get("/expenses", params={"member": "member_a"}, headers=a.manager.headers)) == [mine]


def test_expense_filters_and_paging(api, a):
    food = spend(api, a.manager, 1000, "أكل وشرب", expense_date=TODAY)
    fuel = spend(api, a.manager, 2000, "بنزين ومواصلات", expense_date=TODAY - timedelta(days=1))
    old = spend(api, a.manager, 3000, "أكل وشرب", expense_date=TODAY - timedelta(days=40))
    get = lambda **params: expense_ids(api.get("/expenses", params=params, headers=a.manager.headers))
    assert get() == [food, fuel, old], "newest first"
    assert get(period="this_month") == [food, fuel]
    assert get(period=MONTH) == [food, fuel]
    assert get(category="أكل وشرب") == [food, old]
    assert get(start=(TODAY - timedelta(days=45)).isoformat(), end=(TODAY - timedelta(days=30)).isoformat()) == [old]
    assert get(limit=1, offset=1) == [fuel]
    assert get(limit=200) == [food, fuel, old]


@pytest.mark.parametrize("params, code", [
    ({"limit": 201}, "invalid_args"), ({"limit": 0}, "invalid_args"), ({"offset": -1}, "invalid_args"),
    ({"period": "someday"}, "invalid_args"), ({"category": "طيران"}, "invalid_args"),
    ({"start": "2026-11-01"}, "missing_fields"), ({"start": "not-a-date", "end": "2026-11-01"}, "invalid_args"),
    ({"start": "2026-11-02", "end": "2026-11-01"}, "invalid_args"),
])
def test_bad_expense_filters(api, a, params, code):
    assert_error(api.get("/expenses", params=params, headers=a.manager.headers), 422, code)


def test_a_huge_member_id_is_not_a_server_error(api, a):
    assert_error(api.get("/expenses", params={"member": "9" * 30}, headers=a.manager.headers), 404, "not_found")
    assert_error(api.delete(f"/expenses/{'9' * 30}", headers=a.manager.headers), 422, "invalid_args")


# ---------- changing expenses ----------

def test_member_edits_and_deletes_own_expense_without_a_pending_action(api, a):
    expense = spend(api, a.member, 1000)
    response = api.patch(f"/expenses/{expense}", json={"amount": 1500, "description": "خبز", "date": "2026-11-01"},
                         headers=a.member.headers)
    assert response.status_code == 200
    body = response.json()
    assert (body["amount"], body["description"], body["expense_date"]) == (1500, "خبز", "2026-11-01")
    assert api.delete(f"/expenses/{expense}", headers=a.member.headers).status_code == 200
    assert expense_ids(api.get("/expenses", headers=a.member.headers)) == []


def test_member_cannot_change_another_members_expense(api, a):
    boss, theirs = spend(api, a.manager, 1000), spend(api, a.other, 2000)
    for expense in (boss, theirs):
        assert_error(api.patch(f"/expenses/{expense}", json={"amount": 1}, headers=a.member.headers), 403, "forbidden")
        assert_error(api.delete(f"/expenses/{expense}", headers=a.member.headers), 403, "forbidden")
    assert sorted(expense_ids(api.get("/expenses", headers=a.manager.headers))) == sorted([boss, theirs])


def test_manager_changes_any_expense_of_the_family(api, a):
    expense = spend(api, a.member, 1000)
    assert api.patch(f"/expenses/{expense}", json={"category": "صحة"}, headers=a.manager.headers).json()["category"] == "صحة"
    assert api.delete(f"/expenses/{expense}", headers=a.manager.headers).status_code == 200


def test_expense_owner_and_family_cannot_be_changed_by_the_body(api, a, b):
    expense = spend(api, a.member, 1000)
    response = api.patch(f"/expenses/{expense}", json={"amount": 2000, "family_id": b.manager.family_id,
                                                       "member_id": b.manager.id}, headers=a.member.headers)
    assert response.status_code == 200
    with api.app.state.session_factory() as session:
        row = session.get(Expense, expense)
        assert (row.family_id, row.member_id) == (a.member.family_id, a.member.id)


@pytest.mark.parametrize("body, status", [({}, 422), ({"amount": -1}, 422), ({"amount": None}, 422),
                                          ({"category": "طيران"}, 422), ({"currency": "EUR"}, 422),
                                          ({"date": "soon"}, 422), ({"currency": "USD"}, 422)])
def test_bad_expense_changes(api, a, body, status):
    expense = spend(api, a.manager, 1000)
    assert api.patch(f"/expenses/{expense}", json=body, headers=a.manager.headers).status_code == status
    assert api.get("/expenses", headers=a.manager.headers).json()["expenses"][0]["amount"] == 1000


def test_direct_changes_update_the_vector_store(api, a):
    expense = spend(api, a.manager, 1000)
    api.patch(f"/expenses/{expense}", json={"description": "خبز"}, headers=a.manager.headers)
    assert "خبز" in api.store.items[("expenses", str(expense))]["document"]
    api.delete(f"/expenses/{expense}", headers=a.manager.headers)
    assert ("expenses", str(expense)) not in api.store.items


# ---------- family-level views ----------

def test_summary_is_family_level_for_both_roles(api, a):
    spend(api, a.manager, 80000, "أكل وشرب")
    spend(api, a.member, 30000, "أكل وشرب")
    spend(api, a.other, 5000, "صحة", expense_date=TODAY - timedelta(days=40))
    for who in (a.manager, a.member):
        body = api.get("/summary", params={"month": MONTH}, headers=who.headers).json()
        assert (body["income"], body["total"], body["remaining"]) == (500000, 110000, 390000)
        food = next(c for c in body["categories"] if c["category"] == "أكل وشرب")
        assert (food["spent"], food["limit"], food["percent_used"], food["status"]) == (110000, 100000, 110, "over")
        assert body["alerts"] == [{"category": "أكل وشرب", "status": "over", "spent": 110000, "limit": 100000,
                                   "percent_used": 110}]
    assert api.get("/summary", headers=a.member.headers).json()["month"] == MONTH, "the default is this month"


def test_summary_bad_month(api, a):
    assert_error(api.get("/summary", params={"month": "2026-13"}, headers=a.member.headers), 422, "invalid_args")


def test_budgets_read_for_all_with_their_source(api, a):
    put = lambda **body: api.put("/budgets", json=body, headers=a.manager.headers)
    assert put(category="صحة", amount=1234).status_code == 200
    assert put(category="ترفيه", amount=4321, month=MONTH).status_code == 200
    for who in (a.manager, a.member):
        body = api.get("/budgets", params={"month": MONTH}, headers=who.headers).json()
        rows = {r["category"]: r for r in body["budgets"]}
        assert len(rows) == 12 and body["currency"] == "SYP_OLD"
        assert (rows["صحة"]["limit"], rows["صحة"]["source"]) == (1234, "recurring")
        assert (rows["ترفيه"]["limit"], rows["ترفيه"]["source"]) == (4321, "month")
        assert (rows["أكل وشرب"]["limit"], rows["أكل وشرب"]["source"]) == (100000, "income_percent")
    other_month = api.get("/budgets", params={"month": "2026-12"}, headers=a.member.headers).json()
    assert next(r for r in other_month["budgets"] if r["category"] == "ترفيه")["source"] != "month"


def test_family_body_field_on_set_budget_is_ignored(api, a, b):
    response = api.put("/budgets", json={"category": "صحة", "amount": 10, "family_id": b.manager.family_id},
                       headers=a.manager.headers)
    assert response.status_code == 200
    with api.app.state.session_factory() as session:
        assert [row.family_id for row in session.scalars(select(Budget))] == [a.manager.family_id]


# ---------- manager-only actions ----------

def manager_only(api, target_id: int):
    yield lambda h: api.put("/budgets", json={"category": "صحة", "amount": 1}, headers=h)
    yield lambda h: api.put("/family", json={"monthly_income": 1}, headers=h)
    yield lambda h: api.get("/members", headers=h)
    yield lambda h: api.post("/invites", json={"role": "member"}, headers=h)
    yield lambda h: api.get("/invites", headers=h)
    yield lambda h: api.put(f"/members/{target_id}/role", json={"role": "manager"}, headers=h)
    yield lambda h: api.delete(f"/members/{target_id}", headers=h)


def test_member_is_refused_every_manager_action(api, a):
    for call in manager_only(api, a.other.id):
        assert_error(call(a.member.headers), 403, "forbidden")
    assert api.get("/me", headers=a.manager.headers).json()["family"]["monthly_income"] == 500000
    assert api.get("/me", headers=a.other.headers).status_code == 200, "the target was not stopped"


def test_manager_can_do_every_manager_action(api, a):
    for call in manager_only(api, a.other.id):
        assert call(a.manager.headers).status_code in (200, 201)


def test_every_private_path_needs_a_token(api, a):
    calls = [("get", "/me"), ("post", "/chat"), ("get", "/messages"), ("get", "/expenses"), ("patch", "/expenses/1"),
             ("delete", "/expenses/1"), ("post", "/actions/1/confirm"), ("post", "/actions/1/cancel"),
             ("get", "/summary"), ("get", "/budgets"), ("put", "/budgets"), ("get", "/members"), ("post", "/invites"),
             ("get", "/invites"), ("delete", "/invites/1"), ("delete", "/members/1"), ("put", "/members/1/role"),
             ("put", "/family")]
    for method, path in calls:
        response = api.request(method, path, json={"message": "x", "category": "صحة", "amount": 1, "role": "member"})
        assert_error(response, 401, "unauthorized")


# ---------- two families ----------

def test_another_familys_resources_answer_404(api, a, b):
    expense = spend(api, b.member)
    action = ask_to_change(api, b.member, expense)
    other_invite = invite(api, b.manager)
    for who in (a.manager, a.member):
        assert_error(api.patch(f"/expenses/{expense}", json={"amount": 1}, headers=who.headers), 404, "not_found")
        assert_error(api.delete(f"/expenses/{expense}", headers=who.headers), 404, "not_found")
        assert_error(api.post(f"/actions/{action}/confirm", headers=who.headers), 404, "not_found")
        assert_error(api.post(f"/actions/{action}/cancel", headers=who.headers), 404, "not_found")
    assert_error(api.delete(f"/invites/{other_invite['id']}", headers=a.manager.headers), 404, "not_found")
    assert_error(api.delete(f"/members/{b.member.id}", headers=a.manager.headers), 404, "not_found")
    assert_error(api.put(f"/members/{b.member.id}/role", json={"role": "manager"}, headers=a.manager.headers), 404,
                 "not_found")
    assert_error(api.get("/expenses", params={"member": b.member.id}, headers=a.manager.headers), 404, "not_found")
    assert_error(api.get("/expenses", params={"member": "member_b"}, headers=a.manager.headers), 404, "not_found")
    assert api.get("/me", headers=b.member.headers).status_code == 200


def test_404_is_the_same_as_for_a_resource_that_does_not_exist(api, a, b):
    expense = spend(api, b.manager)
    existing = api.delete(f"/expenses/{expense}", headers=a.manager.headers).json()["error"]
    missing = api.delete("/expenses/99999", headers=a.manager.headers).json()["error"]
    assert existing["code"] == missing["code"]


def test_lists_and_totals_never_cross_families(api, a, b):
    spend(api, a.manager, 1000)
    other = spend(api, b.manager, 9000)
    assert other not in expense_ids(api.get("/expenses", headers=a.manager.headers))
    assert api.get("/summary", headers=a.manager.headers).json()["total"] == 1000
    assert {m["username"] for m in api.get("/members", headers=a.manager.headers).json()["members"]} == {
        "manager_a", "member_a", "other_a"}
    invite(api, b.manager)
    assert api.get("/invites", headers=a.manager.headers).json()["invites"] == []


def test_budgets_and_family_changes_stay_in_the_family(api, a, b):
    api.put("/budgets", json={"category": "صحة", "amount": 1}, headers=a.manager.headers)
    api.put("/family", json={"monthly_income": 1}, headers=a.manager.headers)
    assert api.get("/me", headers=b.manager.headers).json()["family"]["monthly_income"] == 500000
    rows = api.get("/budgets", headers=b.manager.headers).json()["budgets"]
    assert next(r for r in rows if r["category"] == "صحة")["source"] == "income_percent"


def test_messages_are_private_even_inside_the_family(api, a, b):
    with api.app.state.session_factory() as session:
        for who, text in ((a.manager, "رسالة المسؤول"), (a.member, "رسالة الفرد"), (b.manager, "عائلة أخرى")):
            session.add(Message(family_id=who.family_id, member_id=who.id, role="user", content=text))
        session.commit()
    rows = api.get("/messages", headers=a.member.headers).json()["messages"]
    assert [m["content"] for m in rows] == ["رسالة الفرد"]


# ---------- the error shape ----------

def test_unknown_path_and_method_use_the_error_shape(api):
    assert_error(api.get("/no-such-path"), 404, "not_found")
    assert_error(api.put("/health"), 405, "method_not_allowed")


def test_unexpected_failure_uses_the_error_shape_and_hides_details(api, a, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("the secret detail")

    monkeypatch.setattr("app.services.reports.monthly_report", broken)
    response = api.get("/summary", headers=a.manager.headers)
    assert_error(response, 500, "error")
    assert "secret detail" not in response.text


def test_cors_allows_the_configured_origin_only(api):
    allowed = api.options("/me", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    refused = api.options("/me", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in refused.headers


def test_cors_origins_are_a_comma_separated_list(tmp_path, monkeypatch):
    with make_api(tmp_path, monkeypatch, cors_origins="http://a.example, http://b.example") as api:
        response = api.get("/health", headers={"Origin": "http://b.example"})
        assert response.headers["access-control-allow-origin"] == "http://b.example"

