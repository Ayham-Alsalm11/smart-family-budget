"""Expenses: add, get, list, update, delete, with the permission rules of §5.

Confirmation of edits and deletions (§6.5) is not handled here; it comes in step 6.
The embedder and the vector store are passed in (dependency injection); when embedding fails
the expense is still recorded, with `embedded_at` left empty for `reindex_missing`.
"""
import logging
from datetime import date
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.db.categories import CATEGORY_NAMES, SAVINGS_CATEGORY
from app.db.models import Category, Expense, Family, Member, SavingsGoal, utcnow
from app.services.currency import parse_amount, to_base, to_money, validate_currency
from app.services.errors import ForbiddenError, MissingFieldsError, NotFoundError, ValidationError
from app.services.periods import Period, parse_date, resolve_period

log = logging.getLogger(__name__)

EDITABLE_FIELDS = frozenset({"amount", "category", "currency", "date", "description"})
MAX_LIST_LIMIT = 100


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def upsert(self, collection: str, item_id: str, embedding: list[float], document: str,
               metadata: dict[str, Any]) -> None: ...

    def delete(self, collection: str, item_id: str) -> None: ...


# ---------- lookups shared with the other services ----------

def get_family(session: Session, ctx: ToolContext) -> Family:
    family = session.get(Family, ctx.family_id)
    if family is None:
        raise NotFoundError("family not found")
    return family


def get_category(session: Session, name: str | None) -> Category:
    """Exact match on one of the 12 Arabic names. No fuzzy matching, no fallback (P4)."""
    if name is None or name == "":
        raise MissingFieldsError(["category"])
    category = session.scalar(select(Category).where(Category.name_ar == name)) if isinstance(name, str) else None
    if category is None:
        raise ValidationError(f"unknown category {name!r}: expected one of {', '.join(CATEGORY_NAMES)}")
    return category


def resolve_member_filter(session: Session, ctx: ToolContext, member: int | str | None) -> int | None:
    """None → no filter; "me" → the caller; an id, username or name → that member of the family.

    Members may only filter on themselves (§5: totals by member are manager-only).
    """
    if member is None or member == "":
        return None
    if member == "me":
        member_id = ctx.member_id
    else:
        query = select(Member.id).where(Member.family_id == ctx.family_id)
        if isinstance(member, int) and not isinstance(member, bool):
            ids = list(session.scalars(query.where(Member.id == member)))
        elif isinstance(member, str):
            ids = list(session.scalars(query.where((Member.username == member) | (Member.name == member))))
        else:
            raise ValidationError(f"invalid member {member!r}")
        if not ids:
            raise NotFoundError(f"member {member!r} not found in this family")
        if len(ids) > 1:
            raise ValidationError(f"member {member!r} is ambiguous")
        member_id = ids[0]
    if not ctx.is_manager and member_id != ctx.member_id:
        raise ForbiddenError("members can only see their own expenses")
    return member_id


def expense_to_dict(expense: Expense) -> dict:
    return {
        "id": expense.id,
        "member_id": expense.member_id,
        "category": expense.category.name_ar,
        "amount": expense.amount,
        "currency": expense.currency,
        "amount_base": expense.amount_base,
        "rate_to_base": expense.rate_to_base,
        "description": expense.description,
        "expense_date": expense.expense_date.isoformat(),
        "goal_id": expense.goal_id,
        "created_by_agent": expense.created_by_agent,
        "embedded": expense.embedded_at is not None,
    }


def _load_expense(session: Session, ctx: ToolContext, expense_id: int) -> Expense:
    expense = session.get(Expense, expense_id)
    if expense is None or expense.family_id != ctx.family_id:
        raise NotFoundError(f"expense #{expense_id} not found")
    if not ctx.is_manager and expense.member_id != ctx.member_id:
        raise ForbiddenError("members can only see and change their own expenses")
    return expense


def _check_goal(session: Session, ctx: ToolContext, goal_id: int | None, category: Category) -> int | None:
    if goal_id is None:
        return None
    if category.name_ar != SAVINGS_CATEGORY:
        raise ValidationError(f"a savings goal can only be linked to the category {SAVINGS_CATEGORY!r}")
    goal = session.get(SavingsGoal, goal_id)
    if goal is None or goal.family_id != ctx.family_id:
        raise NotFoundError(f"savings goal #{goal_id} not found")
    return goal.id


# ---------- embeddings ----------

def _embedding_text(expense: Expense) -> str:
    return " ".join(part for part in (expense.description, expense.category.name_ar, expense.source_text) if part)


def _vector_metadata(expense: Expense) -> dict[str, Any]:
    return {
        "expense_id": expense.id,
        "family_id": expense.family_id,
        "member_id": expense.member_id,
        "category_id": expense.category_id,
        "category": expense.category.name_ar,
        "amount_base": float(expense.amount_base),
        "currency": expense.currency,
        "expense_date": expense.expense_date.isoformat(),
        "date_key": int(expense.expense_date.strftime("%Y%m%d")),
    }


def index_expense(session: Session, expense: Expense, embedder: Embedder, vector_store: VectorStore) -> bool:
    """Store the expense in the `expenses` collection. Never raises: a failure leaves `embedded_at` empty."""
    text = _embedding_text(expense)
    try:
        vector = embedder.embed(text)
        vector_store.upsert("expenses", str(expense.id), vector, text, _vector_metadata(expense))
    except Exception as exc:
        log.warning("embedding of expense #%s failed: %s", expense.id, exc)
        return False
    expense.embedded_at = utcnow()
    session.commit()
    return True


def reindex_missing(session: Session, embedder: Embedder, vector_store: VectorStore,
                    family_id: int | None = None) -> int:
    """Embed the expenses whose `embedded_at` is empty. Stops at the first failure. Returns how many were stored."""
    query = select(Expense).where(Expense.embedded_at.is_(None)).order_by(Expense.id)
    if family_id is not None:
        query = query.where(Expense.family_id == family_id)
    stored = 0
    for expense in session.scalars(query).all():
        if not index_expense(session, expense, embedder, vector_store):
            break
        stored += 1
    return stored


# ---------- operations ----------

def add_expense(
    session: Session,
    ctx: ToolContext,
    *,
    amount: object,
    category: str | None,
    currency: str | None = None,
    expense_date: date | str | None = None,
    description: str = "",
    source_text: str = "",
    goal_id: int | None = None,
    created_by_agent: bool = False,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
) -> dict:
    """Any member may add; the expense is always recorded under the caller (§5)."""
    missing = [name for name, value in (("amount", amount), ("category", category)) if value is None or value == ""]
    if missing:
        raise MissingFieldsError(missing)
    family = get_family(session, ctx)
    value = parse_amount(amount)
    cat = get_category(session, category)
    currency = validate_currency(currency or family.base_currency)
    amount_base, rate = to_base(value, currency, family)

    expense = Expense(
        family_id=ctx.family_id,
        member_id=ctx.member_id,
        category_id=cat.id,
        amount=value,
        currency=currency,
        amount_base=amount_base,
        rate_to_base=rate,
        description=description or "",
        source_text=source_text or "",
        expense_date=parse_date(expense_date) if expense_date else ctx.today,
        goal_id=_check_goal(session, ctx, goal_id, cat),
        created_by_agent=created_by_agent,
    )
    expense.category = cat
    session.add(expense)
    session.commit()

    if embedder is not None and vector_store is not None:
        index_expense(session, expense, embedder, vector_store)
    return expense_to_dict(expense)


def get_expense(session: Session, ctx: ToolContext, expense_id: int) -> dict:
    return expense_to_dict(_load_expense(session, ctx, expense_id))


def list_expenses(
    session: Session,
    ctx: ToolContext,
    *,
    period: str | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    category: str | None = None,
    member: int | str | None = None,
    limit: int = 10,
) -> dict:
    """Newest first. Members see only their own expenses."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIST_LIMIT:
        raise ValidationError(f"limit must be between 1 and {MAX_LIST_LIMIT}")
    member_id = resolve_member_filter(session, ctx, member)
    if not ctx.is_manager:
        member_id = ctx.member_id

    query = select(Expense).where(Expense.family_id == ctx.family_id)
    span: Period | None = None
    if period:
        span = resolve_period(ctx, period, start_date, end_date)
        query = query.where(Expense.expense_date.between(span.start, span.end))
    if category:
        query = query.where(Expense.category_id == get_category(session, category).id)
    if member_id is not None:
        query = query.where(Expense.member_id == member_id)
    rows = session.scalars(query.order_by(Expense.expense_date.desc(), Expense.id.desc()).limit(limit)).all()
    return {
        "period": span.as_dict() if span else None,
        "expenses": [expense_to_dict(e) for e in rows],
    }


def update_expense(
    session: Session,
    ctx: ToolContext,
    expense_id: int,
    changes: dict[str, Any],
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
) -> dict:
    """Only EDITABLE_FIELDS may change. Changing the amount keeps the stored rate; changing the currency uses today's."""
    if not changes:
        raise ValidationError("no changes given")
    unknown = sorted(set(changes) - EDITABLE_FIELDS)
    if unknown:
        raise ValidationError(f"these fields cannot be changed: {', '.join(unknown)}")
    expense = _load_expense(session, ctx, expense_id)
    family = get_family(session, ctx)

    if "category" in changes:
        expense.category = get_category(session, changes["category"])
        if expense.category.name_ar != SAVINGS_CATEGORY:
            expense.goal_id = None
    amount = parse_amount(changes["amount"]) if "amount" in changes else expense.amount
    if "currency" in changes:
        currency = validate_currency(changes["currency"])
        expense.amount_base, expense.rate_to_base = to_base(amount, currency, family)
        expense.currency = currency
    elif "amount" in changes:
        expense.amount_base = to_money(amount * expense.rate_to_base)
    expense.amount = amount
    if "date" in changes:
        expense.expense_date = parse_date(changes["date"])
    if "description" in changes:
        expense.description = changes["description"] or ""
    expense.embedded_at = None
    session.commit()

    if embedder is not None and vector_store is not None:
        index_expense(session, expense, embedder, vector_store)
    return expense_to_dict(expense)


def delete_expense(
    session: Session,
    ctx: ToolContext,
    expense_id: int,
    vector_store: VectorStore | None = None,
) -> dict:
    expense = _load_expense(session, ctx, expense_id)
    deleted = expense_to_dict(expense)
    session.delete(expense)
    session.commit()
    if vector_store is not None:
        try:
            vector_store.delete("expenses", str(expense_id))
        except Exception as exc:
            log.warning("removing expense #%s from the vector store failed: %s", expense_id, exc)
    return deleted
