"""The expenses table and direct edits (§8): the same services and permissions as the tools, without a pending
action, because the user acts on a specific row in the UI."""
from datetime import date as Date
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.api.deps import MAX_ID, ResourceId, SearchServices
from app.auth.dependencies import Context, DbSession
from app.services import expenses
from app.services.expenses import MAX_LIST_LIMIT

router = APIRouter(tags=["expenses"])


class ExpenseChanges(BaseModel):
    amount: Decimal | None = None
    category: str | None = None
    currency: str | None = None
    date: Date | None = None
    description: str | None = None


def _member_filter(member: str | None) -> int | str | None:
    """Digits mean a member id; other text is a username, a display name or "me"."""
    if member and member.isdecimal() and int(member) <= MAX_ID:
        return int(member)
    return member


@router.get("/expenses")
def list_expenses(
    session: DbSession,
    ctx: Context,
    period: str | None = None,
    start: Date | None = None,
    end: Date | None = None,
    category: str | None = None,
    member: Annotated[str | None, Query(max_length=50)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIST_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0, le=MAX_ID)] = 0,
):
    """`period` is a keyword or YYYY-MM; `start` with `end` gives a custom range. `member`: an id, username, display
    name or "me"; a member may only name themselves."""
    if period is None and (start or end):
        period = "custom"
    result = expenses.list_expenses(session, ctx, period=period, start_date=start, end_date=end, category=category,
                                    member=_member_filter(member), limit=limit, offset=offset)
    return {**result, "limit": limit, "offset": offset}


@router.patch("/expenses/{expense_id}")
def update_expense(expense_id: ResourceId, body: ExpenseChanges, session: DbSession, ctx: Context,
                   rag: SearchServices):
    return expenses.update_expense(session, ctx, expense_id, body.model_dump(exclude_unset=True),
                                   rag.embedder, rag.vector_store)


@router.delete("/expenses/{expense_id}")
def delete_expense(expense_id: ResourceId, session: DbSession, ctx: Context, rag: SearchServices):
    return expenses.delete_expense(session, ctx, expense_id, rag.vector_store)
