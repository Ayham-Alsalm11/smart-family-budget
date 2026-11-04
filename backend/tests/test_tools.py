import json
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.db.categories import CATEGORY_NAMES
from app.db.models import Expense, PendingAction, utcnow
from app.services import expenses
from app.tools.dispatch import TOOLS, ToolDeps, dispatch
from app.tools.schemas import TOOL_NAMES, TOOL_SCHEMAS

FOOD = "أكل وشرب"


def run(session, ctx, name, embedder=None, store=None, **arguments):
    return dispatch(name, arguments, ctx, ToolDeps(session, "رسالة المستخدم", embedder, store))


def test_schemas():
    assert len(TOOL_SCHEMAS) == 11 and set(TOOL_NAMES) == set(TOOLS)
    text = json.dumps(TOOL_SCHEMAS, ensure_ascii=False)
    assert "family_id" not in text and "member_id" not in text
    assert "null" not in text and CATEGORY_NAMES[0] not in text, "no null, and no category enum (§7.1)"
    assert TOOL_SCHEMAS[0]["function"]["parameters"]["required"] == ["amount", "category"]
    for schema in TOOL_SCHEMAS:
        parameters = schema["function"]["parameters"]
        for name, prop in parameters["properties"].items():
            if name not in parameters.get("required", []):
                assert "omit" in prop.get("description", ""), f"{schema['function']['name']}.{name}"


def test_prompt_lists_each_category_once_with_its_keywords():
    from datetime import date
    from app.agent.prompts import CATEGORY_HINTS, build_system_prompt
    from app.db.categories import CATEGORIES
    prompt = build_system_prompt(date(2026, 11, 4), "SYP_OLD")
    for category in CATEGORIES:
        assert prompt.count(f"{category['name_ar']} (") == 1
        assert 0 < len(CATEGORY_HINTS[category["name_ar"]]) <= 4 or not category["keywords"]
        assert set(CATEGORY_HINTS[category["name_ar"]]) <= set(category["keywords"])


@pytest.mark.parametrize("name, arguments, code", [
    ("add_expense", {"amount": -5, "category": FOOD}, "invalid_args"),
    ("add_expense", {"amount": 5, "category": "طعام"}, "invalid_args"),
    ("add_expense", {"amount": 5, "category": "بنزين"}, "invalid_args"),
    ("add_expense", {"category": FOOD}, "missing_fields"),
    ("add_expense", {"amount": None, "category": FOOD}, "missing_fields"),
    ("add_expense", {"amount": 5, "category": FOOD, "family_id": 2}, "invalid_args"),
    ("add_expense", {"amount": 5, "category": FOOD, "date": "أمس"}, "invalid_args"),
    ("edit_expense", {"expense_id": "abc", "changes": {"amount": 5}}, "invalid_args"),
    ("edit_expense", {"expense_id": 1, "changes": {"member_id": 3}}, "invalid_args"),
    ("delete_expense", {}, "missing_fields"),
    ("list_expenses", {"limit": 0}, "invalid_args"),
    ("get_total", {}, "missing_fields"),
    ("get_total", {"period": "next_year"}, "invalid_args"),
    ("get_breakdown", {"period": "this_month", "by": "day"}, "invalid_args"),
    ("compare_months", {"month_a": "2026-13", "month_b": "2026-10"}, "invalid_args"),
    ("compare_months", {"month_a": "2026-10"}, "missing_fields"),
    ("search_expenses", {"query": ""}, "invalid_args"),
    ("check_budget", {"period": "this_year"}, "invalid_args"),
    ("set_budget", {"category": "ترفيهات", "amount": 5}, "invalid_args"),
    ("ask_clarification", {"question": "شو؟", "missing_fields": []}, "invalid_args"),
])
def test_invalid_arguments(session, manager_ctx, name, arguments, code):
    result = dispatch(name, arguments, manager_ctx, ToolDeps(session))
    assert (result.ok, result.code) == (False, code)
    assert session.scalars(select(Expense)).all() == []


def test_missing_fields_lists_the_fields(session, member_ctx):
    result = run(session, member_ctx, "add_expense", description="بنزين")
    assert result.error["missing_fields"] == ["amount", "category"]
    assert result.log_status == "invalid_args"


def test_add_expense_from_the_agent(session, member_ctx):
    expenses.add_expense(session, member_ctx, amount=79000, category=FOOD)
    result = run(session, member_ctx, "add_expense", amount=2000, category=FOOD, description="خبز")
    assert result.ok
    row = session.get(Expense, result.data["expense"]["id"])
    assert (row.created_by_agent, row.source_text) == (True, "رسالة المستخدم")
    assert result.data["alerts"][0]["status"] == "warning"
    assert json.loads(result.to_json())["data"]["expense"]["amount"] == 2000


def test_usd_without_rate_is_missing_fields(session):
    from conftest import ctx_for, make_family, make_member
    ctx = ctx_for(make_member(session, make_family(session, usd_rate=None), "x", "manager"))
    result = run(session, ctx, "add_expense", amount=20, category="فواتير", currency="USD")
    assert result.error["code"] == "missing_fields" and result.error["missing_fields"] == ["usd_rate"]


@pytest.mark.parametrize("name, extra", [("edit_expense", {"changes": {"amount": 6000}}), ("delete_expense", {})])
def test_edit_and_delete_need_confirmation(session, member_ctx, name, extra):
    mine = expenses.add_expense(session, member_ctx, amount=5000, category=FOOD)
    result = run(session, member_ctx, name, expense_id=mine["id"], **extra)
    assert result.code == "confirmation_required"
    assert result.error["expense"]["id"] == mine["id"]
    action = session.get(PendingAction, result.error["confirmation_id"])
    assert (action.tool_name, action.status, action.member_id) == (name, "pending", member_ctx.member_id)
    assert timedelta(minutes=9) < action.expires_at - utcnow() <= timedelta(minutes=10)
    assert expenses.get_expense(session, member_ctx, mine["id"])["amount"] == 5000, "nothing is changed yet"

    refused = run(session, member_ctx, name, expense_id=mine["id"], confirmation_id=action.id, **extra)
    assert refused.code == "invalid_args", "the model has no way to confirm"
    assert session.get(PendingAction, action.id).status == "pending"


def test_search_without_a_period_looks_at_the_last_3_months(session, manager_ctx, fake_embedder, fake_store):
    run(session, manager_ctx, "search_expenses", fake_embedder, fake_store, query="صيدلية")
    assert fake_store.queries[-1]["where"] == {"$and": [{"date_key": {"$gte": 20260901}},
                                                        {"date_key": {"$lte": 20261130}}]}


def test_search_period_in_the_schema():
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_expenses")
    description = schema["function"]["parameters"]["properties"]["period"]["description"]
    assert description.endswith("omit unless the user names a period (default: last 3 months)")


def test_edit_with_invalid_changes_creates_no_pending_action(session, member_ctx):
    mine = expenses.add_expense(session, member_ctx, amount=5000, category=FOOD)
    result = run(session, member_ctx, "edit_expense", expense_id=mine["id"], changes={})
    assert result.code == "invalid_args"
    assert session.scalars(select(PendingAction)).all() == []


def test_member_cannot_reach_another_members_expenses(session, member_ctx, other_member_ctx, fake_embedder,
                                                      fake_store):
    theirs = expenses.add_expense(session, other_member_ctx, amount=7000, category=FOOD, description="مطعم",
                                  embedder=fake_embedder, vector_store=fake_store)
    mine = expenses.add_expense(session, member_ctx, amount=1000, category=FOOD, description="خبز",
                                embedder=fake_embedder, vector_store=fake_store)

    listed = run(session, member_ctx, "list_expenses")
    assert [e["id"] for e in listed.data["expenses"]] == [mine["id"]]
    assert run(session, member_ctx, "list_expenses", member="member2").code == "forbidden"
    assert run(session, member_ctx, "get_total", period="this_month", member="member2").code == "forbidden"
    for name, extra in (("edit_expense", {"changes": {"amount": 1}}), ("delete_expense", {})):
        assert run(session, member_ctx, name, expense_id=theirs["id"], **extra).code == "forbidden"
    assert session.scalars(select(PendingAction)).all() == []

    found = run(session, member_ctx, "search_expenses", fake_embedder, fake_store, query="مطعم")
    assert [m["id"] for m in found.data["matches"]] == [mine["id"]]
    assert fake_store.queries[-1]["ctx"] == member_ctx


def test_search_without_rag_and_with_period(session, manager_ctx, fake_embedder, fake_store):
    assert run(session, manager_ctx, "search_expenses", query="دكتور").code == "not_available"
    run(session, manager_ctx, "search_expenses", fake_embedder, fake_store, query="دكتور", period="this_month")
    assert fake_store.queries[-1]["where"] == {"$and": [{"date_key": {"$gte": 20261101}},
                                                        {"date_key": {"$lte": 20261130}}]}


def test_check_budget_hides_untouched_categories(session, member_ctx):
    expenses.add_expense(session, member_ctx, amount=1000, category=FOOD)
    result = run(session, member_ctx, "check_budget")
    assert [r["category"] for r in result.data["categories"]] == [FOOD]
    assert result.data["not_shown"] == "11 categories with no spending"


def test_set_budget_member_forbidden_manager_ok(session, member_ctx, manager_ctx):
    assert run(session, member_ctx, "set_budget", category="ترفيه", amount=30000).code == "forbidden"
    assert run(session, manager_ctx, "set_budget", category="ترفيه", amount=30000).data["monthly_limit"] == 30000


def test_prompt_is_filled_and_names_currencies_in_arabic():
    from datetime import date
    from app.agent.prompts import build_system_prompt
    prompt = build_system_prompt(date(2026, 11, 4), "SYP_NEW", corrections=["a", "b", "c", "d"])
    assert "{" not in prompt and "}" not in prompt
    assert "Base currency SYP_NEW (ليرة جديدة)" in prompt and "2026-11-03" in prompt
    assert prompt.count("\n- ") == 3 + 11, "3 corrections at most, plus the 11 rules"
