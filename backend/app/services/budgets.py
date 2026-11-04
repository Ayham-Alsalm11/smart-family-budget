"""Budgets: the effective limit rule of §4, set_budget (managers) and check_budget.

Effective limit for category c in month m: the budgets row for (c, m); else the recurring row
(month IS NULL); else monthly_income × budget_percent / 100 (the GP1 rule).
"""
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.db.models import Budget, Category, Expense, Family
from app.services.currency import parse_amount, to_base, to_money, validate_currency
from app.services.errors import ForbiddenError, ValidationError
from app.services.expenses import get_category, get_family
from app.services.periods import month_key, month_period, parse_month, shift_month

WARNING_PERCENT = Decimal(80)
OVER_PERCENT = Decimal(100)
PERCENT_STEP = Decimal("0.1")


def effective_limit(session: Session, family: Family, category: Category, month: str) -> tuple[Decimal, str]:
    """Returns (limit in base currency, source) with source ∈ {month, recurring, income_percent}."""
    query = select(Budget.monthly_limit).where(Budget.family_id == family.id, Budget.category_id == category.id)
    limit = session.scalar(query.where(Budget.month == month))
    if limit is not None:
        return limit, "month"
    limit = session.scalar(query.where(Budget.month.is_(None)))
    if limit is not None:
        return limit, "recurring"
    return to_money(family.monthly_income * category.budget_percent / 100), "income_percent"


def list_budgets(session: Session, ctx: ToolContext, month: str | None = None) -> dict:
    """The effective limit of every category and where it comes from. Family level (§5)."""
    key = _month_of(ctx, month or "this_month")
    family = get_family(session, ctx)
    budgets = []
    for category in session.scalars(select(Category).order_by(Category.id)):
        limit, source = effective_limit(session, family, category, key)
        budgets.append({"category": category.name_ar, "limit": limit, "source": source})
    return {"month": key, "currency": family.base_currency, "budgets": budgets}


def set_budget(
    session: Session,
    ctx: ToolContext,
    *,
    category: str | None,
    amount: object,
    month: str | None = None,
    currency: str | None = None,
) -> dict:
    """Managers only. Without `month` the limit is the recurring default for the category."""
    if not ctx.is_manager:
        raise ForbiddenError("only a manager can set budgets")
    family = get_family(session, ctx)
    cat = get_category(session, category)
    value = parse_amount(amount, allow_zero=True)
    key = month_key(*parse_month(month)) if month else None
    limit, _rate = to_base(value, validate_currency(currency or family.base_currency), family)

    row = session.scalar(select(Budget).where(
        Budget.family_id == family.id, Budget.category_id == cat.id,
        Budget.month == key if key else Budget.month.is_(None),
    ))
    if row is None:
        row = Budget(family_id=family.id, category_id=cat.id, month=key, monthly_limit=limit)
        session.add(row)
    else:
        row.monthly_limit = limit
    session.commit()
    return {"category": cat.name_ar, "month": key, "monthly_limit": limit, "currency": family.base_currency}


def _month_of(ctx: ToolContext, period: str) -> str:
    if period == "this_month":
        return month_key(ctx.today.year, ctx.today.month)
    if period == "last_month":
        return month_key(*shift_month(ctx.today.year, ctx.today.month, -1))
    try:
        return month_key(*parse_month(period))
    except ValidationError:
        raise ValidationError(f"invalid period {period!r}: expected this_month, last_month or YYYY-MM") from None


def _status(spent: Decimal, limit: Decimal) -> tuple[Decimal | None, str]:
    if limit == 0:
        return None, "over" if spent > 0 else "ok"
    used = (spent / limit * 100).quantize(PERCENT_STEP)
    if used >= OVER_PERCENT:
        return used, "over"
    if used >= WARNING_PERCENT:
        return used, "warning"
    return used, "ok"


def budget_alerts(rows: list[dict]) -> list[dict]:
    """The rows of `check_budget` at `warning` or `over`."""
    return [{key: row[key] for key in ("category", "status", "spent", "limit", "percent_used")}
            for row in rows if row["status"] != "ok"]


def check_budget(
    session: Session,
    ctx: ToolContext,
    *,
    category: str | None = None,
    period: str = "this_month",
) -> dict:
    """Spent against the effective limit, per category (all categories when none is given). Family level (§5)."""
    key = _month_of(ctx, period or "this_month")
    family = get_family(session, ctx)
    categories = [get_category(session, category)] if category else list(
        session.scalars(select(Category).order_by(Category.id)))
    span = month_period(*parse_month(key))
    spent_by_category = dict(session.execute(
        select(Expense.category_id, func.sum(Expense.amount_base))
        .where(Expense.family_id == family.id, Expense.expense_date.between(span.start, span.end))
        .group_by(Expense.category_id)
    ).all())

    rows = []
    for cat in categories:
        spent = to_money(spent_by_category.get(cat.id) or 0)
        limit, source = effective_limit(session, family, cat, key)
        used, status = _status(spent, limit)
        rows.append({
            "category": cat.name_ar,
            "spent": spent,
            "limit": limit,
            "limit_source": source,
            "remaining": limit - spent,
            "percent_used": used,
            "status": status,
        })
    return {"month": key, "currency": family.base_currency, "categories": rows}
