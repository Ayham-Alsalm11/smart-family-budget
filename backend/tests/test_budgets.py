from decimal import Decimal

import pytest

from app.services import budgets, expenses
from app.services.errors import ForbiddenError, ValidationError

D = Decimal
FOOD = "أكل وشرب"


def food_row(session, ctx, period="this_month"):
    return budgets.check_budget(session, ctx, category=FOOD, period=period)["categories"][0]


def test_effective_limit_three_cases(session, manager_ctx):
    # 3. No budgets row: income × budget_percent (500,000 × 20 %).
    row = food_row(session, manager_ctx)
    assert (row["limit"], row["limit_source"]) == (D("100000.00"), "income_percent")

    # 2. A recurring row (month NULL) wins over the percentage.
    budgets.set_budget(session, manager_ctx, category=FOOD, amount=80000)
    row = food_row(session, manager_ctx)
    assert (row["limit"], row["limit_source"]) == (D("80000.00"), "recurring")

    # 1. A row for this month wins over the recurring row, and only for that month.
    budgets.set_budget(session, manager_ctx, category=FOOD, amount=60000, month="2026-11")
    row = food_row(session, manager_ctx)
    assert (row["limit"], row["limit_source"]) == (D("60000.00"), "month")
    row = food_row(session, manager_ctx, period="last_month")
    assert (row["limit"], row["limit_source"]) == (D("80000.00"), "recurring")


def test_set_budget_updates_instead_of_duplicating(session, manager_ctx):
    budgets.set_budget(session, manager_ctx, category=FOOD, amount=1000)
    budgets.set_budget(session, manager_ctx, category=FOOD, amount=2000)
    assert food_row(session, manager_ctx)["limit"] == D("2000.00")


def test_set_budget_is_for_managers(session, member_ctx):
    with pytest.raises(ForbiddenError):
        budgets.set_budget(session, member_ctx, category=FOOD, amount=1000)


def test_set_budget_in_another_currency(session, manager_ctx):
    stored = budgets.set_budget(session, manager_ctx, category=FOOD, amount=500, currency="SYP_NEW")
    assert (stored["monthly_limit"], stored["currency"]) == (D("50000.00"), "SYP_OLD")


def test_set_budget_rejects_bad_input(session, manager_ctx):
    with pytest.raises(ValidationError):
        budgets.set_budget(session, manager_ctx, category=FOOD, amount=1000, month="11-2026")
    with pytest.raises(ValidationError):
        budgets.set_budget(session, manager_ctx, category="أكل", amount=1000)


@pytest.mark.parametrize("spent, status, used", [
    (79000, "ok", D("79.0")), (80000, "warning", D("80.0")), (100000, "over", D("100.0")),
])
def test_status_thresholds(session, member_ctx, spent, status, used):
    expenses.add_expense(session, member_ctx, amount=spent, category=FOOD)
    row = food_row(session, member_ctx)
    assert (row["spent"], row["status"], row["percent_used"]) == (D(spent), status, used)
    assert row["remaining"] == D("100000.00") - D(spent)


def test_check_budget_periods(session, member_ctx):
    expenses.add_expense(session, member_ctx, amount=5000, category=FOOD, expense_date="2026-10-03")
    assert food_row(session, member_ctx, period="2026-10")["spent"] == D("5000.00")
    assert food_row(session, member_ctx, period="last_month")["spent"] == D("5000.00")
    assert food_row(session, member_ctx)["spent"] == D("0.00")
    assert len(budgets.check_budget(session, member_ctx)["categories"]) == 12
    with pytest.raises(ValidationError):
        budgets.check_budget(session, member_ctx, period="this_year")
