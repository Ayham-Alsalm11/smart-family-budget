"""Totals, breakdowns and month comparison (§7.1)."""
from decimal import Decimal

import pytest

from app.services import expenses, reports
from app.services.errors import ForbiddenError, MissingFieldsError

D = Decimal


def add(session, ctx, amount, currency=None, category="أكل وشرب", date="2026-11-02"):
    return expenses.add_expense(session, ctx, amount=amount, category=category, currency=currency, expense_date=date)


def test_total_with_mixed_currencies(session, manager_ctx, member_ctx):
    add(session, manager_ctx, 5000, "SYP_OLD")  # 5,000 old
    add(session, member_ctx, 50, "SYP_NEW")  # 5,000 old
    add(session, member_ctx, 1, "USD")  # 1 × 110 new = 11,000 old
    add(session, manager_ctx, 999, "SYP_OLD", date="2026-10-31")  # outside this month
    total = reports.get_total(session, manager_ctx, period="this_month")
    assert total["total"] == D("21000.00")
    assert total["count"] == 3
    assert total["currency"] == "SYP_OLD"


def test_total_filters(session, manager_ctx, member_ctx, member):
    add(session, manager_ctx, 1000)
    add(session, member_ctx, 2000, category="فواتير")
    assert reports.get_total(session, manager_ctx, period="this_month", category="فواتير")["total"] == D("2000.00")
    assert reports.get_total(session, manager_ctx, period="this_month", member=member.id)["total"] == D("2000.00")
    assert reports.get_total(session, member_ctx, period="this_month", member="me")["total"] == D("2000.00")
    # A member may see the family total, but not another member's total.
    assert reports.get_total(session, member_ctx, period="this_month")["total"] == D("3000.00")
    with pytest.raises(ForbiddenError):
        reports.get_total(session, member_ctx, period="this_month", member="manager1")
    with pytest.raises(MissingFieldsError):
        reports.get_total(session, manager_ctx, period=None)


@pytest.mark.parametrize("asked", ["بابا", "manager1", "nobody", 999])
def test_member_filter_is_refused_before_any_lookup(session, member_ctx, asked, monkeypatch):
    """A member gets forbidden for any other name, existing or not, so names in the family are not revealed."""
    from app.services import expenses as service
    looked_up = []
    original = service.select
    monkeypatch.setattr(service, "select", lambda *a, **k: looked_up.append(a) or original(*a, **k))
    with pytest.raises(ForbiddenError):
        reports.get_total(session, member_ctx, period="this_month", member=asked)
    assert looked_up == [], "no member query was run"


@pytest.mark.parametrize("own", ["me", "member1"])
def test_member_may_filter_on_themselves(session, member_ctx, member, own):
    add(session, member_ctx, 700)
    assert reports.get_total(session, member_ctx, period="this_month", member=own)["member_id"] == member.id
    assert reports.get_total(session, member_ctx, period="this_month", member=member.id)["total"] == D("700.00")


def test_breakdown_by_category(session, member_ctx):
    add(session, member_ctx, 3000)
    add(session, member_ctx, 1000, category="فواتير")
    result = reports.get_breakdown(session, member_ctx, period="this_month")
    assert [(r["category"], r["total"], r["share_percent"]) for r in result["rows"]] == [
        ("أكل وشرب", D("3000.00"), D("75.0")), ("فواتير", D("1000.00"), D("25.0"))]
    assert result["total"] == D("4000.00")


def test_breakdown_by_member_is_for_managers(session, manager_ctx, member_ctx, member):
    add(session, manager_ctx, 1000)
    add(session, member_ctx, 3000)
    rows = reports.get_breakdown(session, manager_ctx, period="this_month", by="member")["rows"]
    assert rows[0]["member_id"] == member.id and rows[0]["total"] == D("3000.00")
    with pytest.raises(ForbiddenError):
        reports.get_breakdown(session, member_ctx, period="this_month", by="member")


def test_compare_months_orders_chronologically(session, manager_ctx):
    add(session, manager_ctx, 2000, date="2026-10-10")
    add(session, manager_ctx, 3000, date="2026-11-01")
    for a, b in (("2026-11", "2026-10"), ("2026-10", "2026-11")):
        result = reports.compare_months(session, manager_ctx, month_a=a, month_b=b)
        assert [m["month"] for m in result["months"]] == ["2026-10", "2026-11"]
        assert [m["total"] for m in result["months"]] == [D("2000.00"), D("3000.00")]
        assert result["difference"] == D("1000.00")
        assert result["change_percent"] == D("50.0")


def test_compare_months_decrease_zero_and_category(session, member_ctx):
    add(session, member_ctx, 4000, date="2026-09-05")
    add(session, member_ctx, 1000, date="2026-10-05")
    add(session, member_ctx, 500, date="2026-10-06", category="فواتير")
    result = reports.compare_months(session, member_ctx, month_a="2026-09", month_b="2026-10", category="أكل وشرب")
    assert result["difference"] == D("-3000.00")
    assert result["change_percent"] == D("-75.0")
    empty_first = reports.compare_months(session, member_ctx, month_a="2026-08", month_b="2026-09")
    assert empty_first["change_percent"] is None
    with pytest.raises(MissingFieldsError):
        reports.compare_months(session, member_ctx, month_a="2026-09", month_b=None)


def test_monthly_report(session, member_ctx):
    add(session, member_ctx, 90000)
    report = reports.monthly_report(session, member_ctx)
    assert report["month"] == "2026-11"
    assert (report["income"], report["total"], report["remaining"]) == (D("500000.00"), D("90000.00"), D("410000.00"))
    food = next(c for c in report["categories"] if c["category"] == "أكل وشرب")
    assert (food["limit"], food["status"]) == (D("100000.00"), "warning")
    assert len(report["categories"]) == 12
