"""Argument models of the 11 core tools. The model's arguments are validated here before any service runs."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db.categories import CATEGORY_NAMES
from app.db.models import CURRENCIES

Category = Literal[tuple(CATEGORY_NAMES)]
Currency = Literal[tuple(CURRENCIES)]


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AddExpense(Arguments):
    amount: float = Field(gt=0)
    category: Category
    currency: Currency | None = None
    date: str | None = None
    description: str = ""
    goal_name: str | None = None


class Changes(Arguments):
    amount: float | None = Field(None, gt=0)
    category: Category | None = None
    currency: Currency | None = None
    date: str | None = None
    description: str | None = None


class EditExpense(Arguments):
    expense_id: int
    changes: Changes


class DeleteExpense(Arguments):
    expense_id: int


class ListExpenses(Arguments):
    period: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    category: Category | None = None
    member: str | int | None = None
    limit: int = Field(10, ge=1, le=100)


class GetTotal(Arguments):
    period: str
    start_date: str | None = None
    end_date: str | None = None
    category: Category | None = None
    member: str | int | None = None


class GetBreakdown(Arguments):
    period: str
    start_date: str | None = None
    end_date: str | None = None
    by: Literal["category", "member"] = "category"


class CompareMonths(Arguments):
    month_a: str
    month_b: str
    category: Category | None = None


class SearchExpenses(Arguments):
    query: str = Field(min_length=1)
    period: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int = Field(5, ge=1, le=20)


class CheckBudget(Arguments):
    category: Category | None = None
    period: str = "this_month"


class SetBudget(Arguments):
    category: Category
    amount: float = Field(ge=0)
    month: str | None = None
    currency: Currency | None = None


class AskClarification(Arguments):
    question: str = Field(min_length=1)
    missing_fields: list[str] = Field(min_length=1)  # free text: informational only
