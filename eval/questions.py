"""The 10 numeric question types and their reference answers (§9.1, §9.2).

Each reference is computed with pandas from the neutral records file only: not through the agent's tools and not through
GP1's code, otherwise a system would be tested against itself. The reference is computed when the run starts, so it
always belongs to that day's database and is never written by hand.

Periods follow §7.4: a week starts on Saturday (WEEK_START), and "the last 3 months" are the current month and the two
before it. A month given by name or number is its latest occurrence that is not in the future.
"""
import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

TYPES = (
    "total_this_month", "category_this_month", "total_last_month", "category_last_month", "total_named_month",
    "difference_this_last", "budget_remaining", "top_category", "category_last_3_months", "total_this_week",
)
SATURDAY = 5
PILOT_SOURCE = "pilot"


def records_frame(records: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(records["expenses"], columns=["date", "category", "amount"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _month(today: date, back: int) -> tuple[int, int]:
    index = today.year * 12 + today.month - 1 - back
    return index // 12, index % 12 + 1


def _in_month(frame: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    return frame[(frame["date"].dt.year == year) & (frame["date"].dt.month == month)]


def _in_category(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    return frame[frame["category"] == category]


def _total(frame: pd.DataFrame) -> int | float:
    total = float(frame["amount"].sum())
    return int(total) if total.is_integer() else total


def _named_month(today: date, month: int) -> tuple[int, int]:
    return (today.year if month <= today.month else today.year - 1), month


def _total_this_month(frame, today, records, params):
    return {"value": _total(_in_month(frame, *_month(today, 0)))}


def _category_this_month(frame, today, records, params):
    return {"value": _total(_in_category(_in_month(frame, *_month(today, 0)), params["category"]))}


def _total_last_month(frame, today, records, params):
    return {"value": _total(_in_month(frame, *_month(today, 1)))}


def _category_last_month(frame, today, records, params):
    return {"value": _total(_in_category(_in_month(frame, *_month(today, 1)), params["category"]))}


def _total_named_month(frame, today, records, params):
    year, month = _named_month(today, params["month"])
    return {"value": _total(_in_month(frame, year, month)), "detail": f"{year}-{month:02d}"}


def _difference_this_last(frame, today, records, params):
    this_month = _total(_in_month(frame, *_month(today, 0)))
    last_month = _total(_in_month(frame, *_month(today, 1)))
    return {"value": abs(this_month - last_month), "detail": f"{this_month} - {last_month}"}


def _budget_remaining(frame, today, records, params):
    """The limit is income × budget_percent, in both systems: no budget rows exist (§9.2)."""
    percent = next(c["budget_percent"] for c in records["categories"] if c["name"] == params["category"])
    limit = records["family"]["monthly_income"] * percent / 100
    spent = _total(_in_category(_in_month(frame, *_month(today, 0)), params["category"]))
    value = limit - spent
    return {"value": int(value) if float(value).is_integer() else value, "detail": f"{limit:g} - {spent}"}


def _top_category(frame, today, records, params):
    totals = _in_month(frame, *_month(today, 0)).groupby("category")["amount"].sum()
    if totals.empty:
        return {"value": 0, "detail": ""}
    # A tie must not depend on the order of the rows.
    ordered = totals.sort_index().sort_values(ascending=False, kind="stable")
    return {"value": int(ordered.iloc[0]), "detail": ordered.index[0]}


def _category_last_3_months(frame, today, records, params):
    year, month = _month(today, 2)
    start = pd.Timestamp(year, month, 1)
    chosen = _in_category(frame[(frame["date"] >= start) & (frame["date"] <= pd.Timestamp(today))], params["category"])
    return {"value": _total(chosen), "detail": f"from {start.date()}"}


def _total_this_week(frame, today, records, params):
    start = today - timedelta(days=(today.weekday() - SATURDAY) % 7)
    chosen = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(today))]
    return {"value": _total(chosen), "detail": f"from {start}"}


REFERENCES: dict[str, Callable[..., dict[str, Any]]] = {
    "total_this_month": _total_this_month,
    "category_this_month": _category_this_month,
    "total_last_month": _total_last_month,
    "category_last_month": _category_last_month,
    "total_named_month": _total_named_month,
    "difference_this_last": _difference_this_last,
    "budget_remaining": _budget_remaining,
    "top_category": _top_category,
    "category_last_3_months": _category_last_3_months,
    "total_this_week": _total_this_week,
}
assert tuple(REFERENCES) == TYPES, "every type needs a reference"


def reference(question: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    """{"value": the number, "detail": how it was reached, when that helps a reader}."""
    today = date.fromisoformat(records["today"])
    return REFERENCES[question["type"]](records_frame(records), today, records, question.get("params") or {})


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Pilot questions are written by Claude Code, so they must never reach the student's final set (§9.1)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    for question in questions:
        if question["type"] not in TYPES:
            raise ValueError(f"{question['id']}: unknown type {question['type']!r}")
        if question.get("source") == PILOT_SOURCE and not Path(path).name.startswith("pilot_"):
            raise ValueError(f"{question['id']}: a pilot question in {Path(path).name}")
    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError(f"{path}: repeated ids")
    return questions
