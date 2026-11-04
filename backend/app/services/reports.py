"""Totals, breakdowns, month comparison and the monthly report. Every sum is over `amount_base`."""
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.db.models import Category, Expense, Member
from app.services.currency import to_money
from app.services.errors import ForbiddenError, MissingFieldsError, ValidationError
from app.services.expenses import get_category, get_family, resolve_member_filter
from app.services.periods import Period, month_key, month_period, parse_month, resolve_period

PERCENT_STEP = Decimal("0.1")


def _percent(part: Decimal, whole: Decimal) -> Decimal | None:
    return None if whole == 0 else (part / whole * 100).quantize(PERCENT_STEP)


def sum_expenses(
    session: Session,
    family_id: int,
    period: Period,
    category_id: int | None = None,
    member_id: int | None = None,
) -> tuple[Decimal, int]:
    query = select(func.sum(Expense.amount_base), func.count(Expense.id)).where(
        Expense.family_id == family_id, Expense.expense_date.between(period.start, period.end),
    )
    if category_id is not None:
        query = query.where(Expense.category_id == category_id)
    if member_id is not None:
        query = query.where(Expense.member_id == member_id)
    total, count = session.execute(query).one()
    return to_money(total or 0), count


def get_total(
    session: Session,
    ctx: ToolContext,
    *,
    period: str | None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    category: str | None = None,
    member: int | str | None = None,
) -> dict:
    """Family total by default. A member filter is allowed for managers, or for "me" (§5)."""
    span = resolve_period(ctx, period, start_date, end_date)
    family = get_family(session, ctx)
    cat = get_category(session, category) if category else None
    member_id = resolve_member_filter(session, ctx, member)
    total, count = sum_expenses(session, ctx.family_id, span, cat.id if cat else None, member_id)
    return {
        "period": span.as_dict(),
        "category": cat.name_ar if cat else None,
        "member_id": member_id,
        "total": total,
        "count": count,
        "currency": family.base_currency,
    }


def get_breakdown(
    session: Session,
    ctx: ToolContext,
    *,
    period: str | None,
    by: str = "category",
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> dict:
    if by not in ("category", "member"):
        raise ValidationError(f"invalid by {by!r}: expected 'category' or 'member'")
    if by == "member" and not ctx.is_manager:
        raise ForbiddenError("the breakdown by member is for managers only")
    span = resolve_period(ctx, period, start_date, end_date)
    family = get_family(session, ctx)

    if by == "category":
        keys = (Category.name_ar,)
        query = select(Category.name_ar, func.sum(Expense.amount_base), func.count(Expense.id)).join(Category)
    else:
        keys = (Member.id, Member.name)
        query = select(Member.id, Member.name, func.sum(Expense.amount_base), func.count(Expense.id)).join(
            Member, Member.id == Expense.member_id)
    query = query.where(
        Expense.family_id == ctx.family_id, Expense.expense_date.between(span.start, span.end),
    ).group_by(*keys)

    rows = []
    for *key, total, count in session.execute(query):
        row = {"category": key[0]} if by == "category" else {"member_id": key[0], "member": key[1]}
        rows.append({**row, "total": to_money(total or 0), "count": count})
    grand_total = to_money(sum((r["total"] for r in rows), Decimal(0)))
    for row in rows:
        row["share_percent"] = _percent(row["total"], grand_total)
    rows.sort(key=lambda r: r["total"], reverse=True)
    return {
        "period": span.as_dict(),
        "by": by,
        "rows": rows,
        "total": grand_total,
        "currency": family.base_currency,
    }


def compare_months(
    session: Session,
    ctx: ToolContext,
    *,
    month_a: str | None,
    month_b: str | None,
    category: str | None = None,
) -> dict:
    """The two months are put in time order; the older one is the reference.

    difference = newer − older; change_percent = difference ÷ older × 100, or None when the older total is 0.
    """
    missing = [name for name, value in (("month_a", month_a), ("month_b", month_b)) if not value]
    if missing:
        raise MissingFieldsError(missing)
    older, newer = sorted([parse_month(month_a), parse_month(month_b)])
    family = get_family(session, ctx)
    cat = get_category(session, category) if category else None

    months = []
    for year, month in (older, newer):
        total, count = sum_expenses(session, ctx.family_id, month_period(year, month), cat.id if cat else None)
        months.append({"month": month_key(year, month), "total": total, "count": count})
    difference = months[1]["total"] - months[0]["total"]
    return {
        "months": months,
        "difference": difference,
        "change_percent": _percent(difference, months[0]["total"]),
        "category": cat.name_ar if cat else None,
        "currency": family.base_currency,
    }


def monthly_report(session: Session, ctx: ToolContext, month: str | None = None) -> dict:
    """Income, total, remaining and every category against its effective budget (for /summary)."""
    from app.services.budgets import budget_alerts, check_budget

    year, number = parse_month(month) if month else (ctx.today.year, ctx.today.month)
    key = month_key(year, number)
    family = get_family(session, ctx)
    budget = check_budget(session, ctx, period=key)
    total = to_money(sum((row["spent"] for row in budget["categories"]), Decimal(0)))
    return {
        "month": key,
        "income": family.monthly_income,
        "total": total,
        "remaining": family.monthly_income - total,
        "spent_percent": _percent(total, family.monthly_income),
        "categories": budget["categories"],
        "alerts": budget_alerts(budget["categories"]),
        "currency": family.base_currency,
    }
