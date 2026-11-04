"""Budgets (§8): anyone reads the effective limits; only a manager sets one."""
from decimal import Decimal

from fastapi import APIRouter
from pydantic import BaseModel

from app.auth.dependencies import Context, DbSession
from app.services import budgets

router = APIRouter(tags=["budgets"])


class SetBudget(BaseModel):
    category: str
    amount: Decimal
    month: str | None = None  # omitted: the recurring limit of the category
    currency: str | None = None


@router.get("/budgets")
def list_budgets(session: DbSession, ctx: Context, month: str | None = None):
    return budgets.list_budgets(session, ctx, month)


@router.put("/budgets")
def set_budget(body: SetBudget, session: DbSession, ctx: Context):
    return budgets.set_budget(session, ctx, **body.model_dump())
