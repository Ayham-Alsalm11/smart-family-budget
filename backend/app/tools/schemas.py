"""JSON schemas of the 11 core tools (§7.1), as sent to the model.

Descriptions are kept very short because every call pays for them (§9.5). The family and the
member never appear here: they come from ToolContext on the server (§6.2).
"""
from app.db.categories import CATEGORY_NAMES
from app.db.models import CURRENCIES

MISSING_FIELDS = ("amount", "category", "currency", "date", "expense", "goal", "member", "other")

_CATEGORY = {"type": "string", "enum": CATEGORY_NAMES}
# Filters name a category without repeating the enum (the names are in the system prompt); the
# argument models still accept only the 12 names.
_CATEGORY_FILTER = {"type": "string", "description": "category name"}
_CURRENCY = {"type": "string", "enum": list(CURRENCIES)}
_DATE = {"type": "string", "description": "YYYY-MM-DD"}
_MONTH = {"type": "string", "description": "YYYY-MM"}
_PERIOD = {
    "type": "string",
    "description": "today|yesterday|this_week|last_week|this_month|last_month|last_3_months|this_year|YYYY-MM|"
                   "custom (+start_date, end_date)",
}
_RANGE = {"start_date": _DATE, "end_date": _DATE}
_MEMBER = {"type": "string", "description": "'me' or a member's name"}


def _tool(name: str, description: str, properties: dict, required: tuple[str, ...] = ()) -> dict:
    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = list(required)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOL_SCHEMAS = [
    _tool("add_expense", "Record an expense. Currency defaults to the base currency, date to today.", {
        "amount": {"type": "number"},
        "category": _CATEGORY,
        "currency": _CURRENCY,
        "date": _DATE,
        "description": {"type": "string", "description": "2-4 Arabic words"},
        "goal_name": {"type": "string", "description": "savings goal, only for ادخار"},
    }, ("amount", "category")),
    _tool("edit_expense", "Change an expense. Returns confirmation_required.", {
        "expense_id": {"type": "integer"},
        "changes": {"type": "object", "properties": {
            "amount": {"type": "number"}, "category": _CATEGORY, "currency": _CURRENCY,
            "date": _DATE, "description": {"type": "string"},
        }},
        "confirmation_id": {"type": "integer"},
    }, ("expense_id", "changes")),
    _tool("delete_expense", "Delete an expense. Returns confirmation_required.", {
        "expense_id": {"type": "integer"},
        "confirmation_id": {"type": "integer"},
    }, ("expense_id",)),
    _tool("list_expenses", "List expenses with ids, newest first.", {
        "period": _PERIOD, **_RANGE, "category": _CATEGORY_FILTER, "member": _MEMBER,
        "limit": {"type": "integer", "description": "default 10"},
    }),
    _tool("get_total", "Total spent in the base currency.", {
        "period": _PERIOD, **_RANGE, "category": _CATEGORY_FILTER, "member": _MEMBER,
    }, ("period",)),
    _tool("get_breakdown", "Totals and shares per category or per member.", {
        "period": _PERIOD, **_RANGE, "by": {"type": "string", "enum": ["category", "member"]},
    }, ("period",)),
    _tool("compare_months", "Compare the totals of two months.", {
        "month_a": _MONTH, "month_b": _MONTH, "category": _CATEGORY_FILTER,
    }, ("month_a", "month_b")),
    _tool("search_expenses", "Find expenses by meaning, e.g. a shop or a person.", {
        "query": {"type": "string"}, "period": _PERIOD, **_RANGE,
        "limit": {"type": "integer", "description": "default 5"},
    }, ("query",)),
    _tool("check_budget", "Spent, limit and remaining per category for a month.", {
        "category": _CATEGORY_FILTER,
        "period": {"type": "string", "description": "this_month|last_month|YYYY-MM"},
    }),
    _tool("set_budget", "Set a category's monthly limit (managers only).", {
        "category": _CATEGORY, "amount": {"type": "number"},
        "month": {"type": "string", "description": "YYYY-MM; omit for every month"},
        "currency": _CURRENCY,
    }, ("category", "amount")),
    _tool("ask_clarification", "Ask the user for missing or unclear details. Ends the turn.", {
        "question": {"type": "string", "description": "in Syrian Arabic"},
        "missing_fields": {"type": "array", "items": {"type": "string", "enum": list(MISSING_FIELDS)}},
    }, ("question", "missing_fields")),
]

TOOL_NAMES = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
