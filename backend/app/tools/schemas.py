"""JSON schemas of the 11 core tools (§7.1), as sent to the model.

Kept short because every call pays for them (§9.5). Groq rejects a call that breaks the schema
(HTTP 400 tool_use_failed), so the schemas are loose: no category enum (the 12 names are listed once
in the system prompt), no null, and optional arguments say "omit if unknown". The argument models in
arguments.py stay strict, and their errors go back to the model as invalid_args.
The family and the member never appear here: they come from ToolContext on the server (§6.2).
"""
from app.db.models import CURRENCIES

_OMIT = "omit if unknown"
_CATEGORY = {"type": "string", "description": "one of the 12 categories"}
_CATEGORY_OPT = {"type": "string", "description": f"category; {_OMIT}"}
_CURRENCY = {"type": "string", "enum": list(CURRENCIES), "description": _OMIT}
_DATE = {"type": "string", "description": f"YYYY-MM-DD; {_OMIT}"}
_PERIODS = "today|yesterday|this_week|last_week|this_month|last_month|last_3_months|this_year|YYYY-MM|custom"
_PERIOD = {"type": "string", "description": _PERIODS}
_PERIOD_OPT = {"type": "string", "description": f"{_PERIODS}; {_OMIT}"}
_SEARCH_PERIOD = {"type": "string", "description": f"{_PERIODS}; omit unless the user names a period (default: last 3 months)"}
_RANGE = {"start_date": {"type": "string", "description": "YYYY-MM-DD, only with custom; omit otherwise"},
          "end_date": {"type": "string", "description": "YYYY-MM-DD, only with custom; omit otherwise"}}
_MEMBER = {"type": "string", "description": "'me' or a name; omit for the whole family"}
_LIMIT = {"type": "integer", "description": _OMIT}


def _tool(name: str, description: str, properties: dict, required: tuple[str, ...] = ()) -> dict:
    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = list(required)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOL_SCHEMAS = [
    _tool("add_expense", "Record an expense.", {
        "amount": {"type": "number"}, "category": _CATEGORY, "currency": _CURRENCY, "date": _DATE,
        "description": {"type": "string", "description": f"2-4 words; {_OMIT}"},
        "goal_name": {"type": "string", "description": "savings goal, only for ادخار; omit otherwise"},
    }, ("amount", "category")),
    _tool("edit_expense", "Change an expense; needs confirmation.", {
        "expense_id": {"type": "integer"},
        "changes": {"type": "object", "description": "only the changed fields", "properties": {
            "amount": {"type": "number"}, "category": _CATEGORY, "currency": _CURRENCY,
            "date": {"type": "string"}, "description": {"type": "string"}}},
    }, ("expense_id", "changes")),
    _tool("delete_expense", "Delete an expense; needs confirmation.", {
        "expense_id": {"type": "integer"},
    }, ("expense_id",)),
    _tool("list_expenses", "List expenses with ids, newest first.", {
        "period": _PERIOD_OPT, **_RANGE, "category": _CATEGORY_OPT, "member": _MEMBER, "limit": _LIMIT,
    }),
    _tool("get_total", "Total spent.", {
        "period": _PERIOD, **_RANGE, "category": _CATEGORY_OPT, "member": _MEMBER,
    }, ("period",)),
    _tool("get_breakdown", "Totals and shares per category or member.", {
        "period": _PERIOD, **_RANGE,
        "by": {"type": "string", "enum": ["category", "member"], "description": _OMIT},
    }, ("period",)),
    _tool("compare_months", "Totals of two months.", {
        "month_a": {"type": "string", "description": "YYYY-MM"},
        "month_b": {"type": "string", "description": "YYYY-MM"},
        "category": _CATEGORY_OPT,
    }, ("month_a", "month_b")),
    _tool("search_expenses", "Find expenses by meaning or category word (shop, person, item), newest first.", {
        "query": {"type": "string"}, "period": _SEARCH_PERIOD, **_RANGE,
    }, ("query",)),
    _tool("check_budget", "Spent, limit and remaining per category in a month.", {
        "category": {"type": "string", "description": "category; omit for all"},
        "period": {"type": "string", "description": "this_month|last_month|YYYY-MM; omit for this month"},
    }),
    _tool("set_budget", "Set a category's monthly limit (managers).", {
        "category": _CATEGORY, "amount": {"type": "number"},
        "month": {"type": "string", "description": "YYYY-MM; omit for every month"},
        "currency": _CURRENCY,
    }, ("category", "amount")),
    _tool("ask_clarification", "Ask the user; ends the turn.", {
        "question": {"type": "string"},
        "missing_fields": {"type": "array", "items": {"type": "string"}, "description": "e.g. amount, category"},
    }, ("question", "missing_fields")),
]

TOOL_NAMES = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
