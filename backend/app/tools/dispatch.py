"""Tool dispatch: validate the model's arguments, call the services, and wrap the outcome (§7).

Every tool returns {"ok": true, "data": …} or {"ok": false, "error": {"code": …, "message": …}}.
Error codes: invalid_args, missing_fields (+ the field list), forbidden, not_found,
confirmation_required (+ preview and confirmation_id), not_available, error.
"""
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import pydantic
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.services import budgets, expenses, pending, reports
from app.services.errors import ServiceError
from app.services.periods import resolve_period
from app.tools import arguments as A
from app.tools.schemas import TOOL_NAMES

log = logging.getLogger(__name__)

SEARCH_DEFAULT_PERIOD = "last_3_months"

# tool_log.status has no separate value for these codes (§4).
_LOG_STATUS = {"missing_fields": "invalid_args", "not_available": "error"}


@dataclass
class ToolDeps:
    session: Session
    source_text: str = ""  # the user's message, stored with expenses the agent adds
    embedder: Any = None
    vector_store: Any = None


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Any = None
    error: dict | None = None

    @property
    def code(self) -> str:
        return "ok" if self.ok else self.error["code"]

    @property
    def log_status(self) -> str:
        return _LOG_STATUS.get(self.code, self.code)

    def payload(self) -> dict:
        return {"ok": True, "data": self.data} if self.ok else {"ok": False, "error": self.error}

    def to_json(self) -> str:
        return dumps(self.payload())


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)


class ToolFailure(Exception):
    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.error = {"code": code, "message": message, **extra}


def _brief(expense: dict, with_member: bool = False) -> dict:
    row = {
        "id": expense["id"], "amount": expense["amount"], "currency": expense["currency"],
        "category": expense["category"], "date": expense["expense_date"], "description": expense["description"],
    }
    if expense["amount"] != expense["amount_base"]:
        row["amount_base"] = expense["amount_base"]
    if with_member:
        row["member_id"] = expense["member_id"]
    return row


# ---------- handlers ----------

def _add_expense(args: A.AddExpense, ctx: ToolContext, deps: ToolDeps) -> dict:
    goal_id = expenses.find_goal_id(deps.session, ctx, args.goal_name) if args.goal_name else None
    added = expenses.add_expense(
        deps.session, ctx, amount=args.amount, category=args.category, currency=args.currency,
        expense_date=args.date, description=args.description, source_text=deps.source_text, goal_id=goal_id,
        created_by_agent=True, embedder=deps.embedder, vector_store=deps.vector_store,
    )
    row = budgets.check_budget(deps.session, ctx, category=added["category"],
                               period=added["expense_date"][:7])["categories"][0]
    return {"expense": {**_brief(added), "amount_base": added["amount_base"]}, "alerts": budgets.budget_alerts([row])}


def _edit_expense(args: A.EditExpense, ctx: ToolContext, deps: ToolDeps) -> dict:
    preview = pending.request_change(deps.session, ctx, "edit_expense", args.expense_id,
                                     args.changes.model_dump(exclude_none=True))
    raise ToolFailure("confirmation_required", "Show the preview and ask the user to confirm.", **preview)


def _delete_expense(args: A.DeleteExpense, ctx: ToolContext, deps: ToolDeps) -> dict:
    preview = pending.request_change(deps.session, ctx, "delete_expense", args.expense_id)
    raise ToolFailure("confirmation_required", "Show the preview and ask the user to confirm.", **preview)


def _list_expenses(args: A.ListExpenses, ctx: ToolContext, deps: ToolDeps) -> dict:
    result = expenses.list_expenses(
        deps.session, ctx, period=args.period, start_date=args.start_date, end_date=args.end_date,
        category=args.category, member=args.member, limit=args.limit,
    )
    return {"period": result["period"],
            "expenses": [_brief(e, with_member=ctx.is_manager) for e in result["expenses"]]}


def _get_total(args: A.GetTotal, ctx: ToolContext, deps: ToolDeps) -> dict:
    return reports.get_total(deps.session, ctx, period=args.period, start_date=args.start_date,
                             end_date=args.end_date, category=args.category, member=args.member)


def _get_breakdown(args: A.GetBreakdown, ctx: ToolContext, deps: ToolDeps) -> dict:
    return reports.get_breakdown(deps.session, ctx, period=args.period, by=args.by,
                                 start_date=args.start_date, end_date=args.end_date)


def _compare_months(args: A.CompareMonths, ctx: ToolContext, deps: ToolDeps) -> dict:
    return reports.compare_months(deps.session, ctx, month_a=args.month_a, month_b=args.month_b,
                                  category=args.category)


def _search_expenses(args: A.SearchExpenses, ctx: ToolContext, deps: ToolDeps) -> dict:
    if deps.embedder is None or deps.vector_store is None:
        raise ToolFailure("not_available", "Search by meaning is not available now; use list_expenses.")
    span = resolve_period(ctx, args.period or SEARCH_DEFAULT_PERIOD, args.start_date, args.end_date)
    where = {"$and": [{"date_key": {"$gte": int(span.start.strftime("%Y%m%d"))}},
                      {"date_key": {"$lte": int(span.end.strftime("%Y%m%d"))}}]}
    try:
        vector = deps.embedder.embed(args.query)
        hits = deps.vector_store.query("expenses", ctx, vector, n_results=args.limit, where=where)
    except Exception as exc:
        log.warning("search_expenses failed: %s", exc)
        raise ToolFailure("not_available", "Search by meaning is not available now; use list_expenses.") from exc
    matches = []
    for hit in hits:
        try:
            expense = expenses.get_expense(deps.session, ctx, int(hit["metadata"]["expense_id"]))
        except ServiceError:
            continue  # deleted since it was indexed, or outside the caller's scope
        matches.append({**_brief(expense, with_member=ctx.is_manager), "distance": round(hit["distance"], 3)})
    return {"matches": matches}


def _check_budget(args: A.CheckBudget, ctx: ToolContext, deps: ToolDeps) -> dict:
    result = budgets.check_budget(deps.session, ctx, category=args.category, period=args.period)
    if args.category:
        return result
    # Without a category, leave out untouched categories that only have the default limit (saves tokens).
    shown = [r for r in result["categories"] if r["spent"] > 0 or r["limit_source"] != "income_percent"]
    hidden = len(result["categories"]) - len(shown)
    return {**result, "categories": shown, "not_shown": f"{hidden} categories with no spending"}


def _set_budget(args: A.SetBudget, ctx: ToolContext, deps: ToolDeps) -> dict:
    return budgets.set_budget(deps.session, ctx, category=args.category, amount=args.amount, month=args.month,
                              currency=args.currency)


def _ask_clarification(args: A.AskClarification, ctx: ToolContext, deps: ToolDeps) -> dict:
    return {"question": args.question, "missing_fields": args.missing_fields}


Handler = Callable[[Any, ToolContext, ToolDeps], dict]
TOOLS: dict[str, tuple[type[A.Arguments], Handler]] = {
    "add_expense": (A.AddExpense, _add_expense),
    "edit_expense": (A.EditExpense, _edit_expense),
    "delete_expense": (A.DeleteExpense, _delete_expense),
    "list_expenses": (A.ListExpenses, _list_expenses),
    "get_total": (A.GetTotal, _get_total),
    "get_breakdown": (A.GetBreakdown, _get_breakdown),
    "compare_months": (A.CompareMonths, _compare_months),
    "search_expenses": (A.SearchExpenses, _search_expenses),
    "check_budget": (A.CheckBudget, _check_budget),
    "set_budget": (A.SetBudget, _set_budget),
    "ask_clarification": (A.AskClarification, _ask_clarification),
}
assert list(TOOLS) == TOOL_NAMES, "every schema needs a handler, in the same order"


# ---------- dispatch ----------

def _validation_error(name: str, model: type[A.Arguments], exc: pydantic.ValidationError) -> dict:
    required = {field for field, info in model.model_fields.items() if info.is_required()}
    missing, problems = [], []
    for error in exc.errors():
        field = str(error["loc"][0]) if error["loc"] else ""
        if error["type"] == "missing" or (error.get("input") is None and field in required):
            missing.append(field)
        elif error["type"] == "literal_error" and field == "category":
            problems.append(f"category {error.get('input')!r} is not one of the 12 categories in the system prompt")
        else:
            location = ".".join(str(part) for part in error["loc"])
            problems.append(f"{location}: {error['msg']}"[:160])
    if missing and not problems:
        return {"code": "missing_fields", "message": f"{name} needs: {', '.join(missing)}", "missing_fields": missing}
    return {"code": "invalid_args", "message": "; ".join(problems + [f"{m}: required" for m in missing])}


def dispatch(name: str, arguments: dict | None, ctx: ToolContext, deps: ToolDeps,
             parse_error: str | None = None) -> ToolResult:
    if name not in TOOLS:
        return ToolResult(name, False, error={"code": "invalid_args",
                                              "message": f"unknown tool {name!r}; tools: {', '.join(TOOL_NAMES)}"})
    if arguments is None:
        return ToolResult(name, False, error={"code": "invalid_args", "message": parse_error or "no arguments"})
    model, handler = TOOLS[name]
    try:
        args = model.model_validate(arguments)
    except pydantic.ValidationError as exc:
        return ToolResult(name, False, error=_validation_error(name, model, exc))
    try:
        return ToolResult(name, True, data=handler(args, ctx, deps))
    except ToolFailure as exc:
        return ToolResult(name, False, error=exc.error)
    except ServiceError as exc:
        deps.session.rollback()
        return ToolResult(name, False, error=exc.to_dict())
    except Exception:
        deps.session.rollback()
        log.exception("tool %s failed", name)
        return ToolResult(name, False, error={"code": "error", "message": "internal error"})
