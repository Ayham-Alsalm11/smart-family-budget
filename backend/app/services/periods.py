"""Period keywords → inclusive date ranges (§7.4), resolved from ToolContext.today.

Calendar periods cover the whole week, month or year, including days after today.
`last_3_months` = the current month and the two before it.
"""
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

from app.agent.context import ToolContext
from app.services.errors import MissingFieldsError, ValidationError
from app.settings import get_settings

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
KEYWORDS = (
    "today", "yesterday", "this_week", "last_week", "this_month", "last_month",
    "last_3_months", "this_year", "custom",
)
_MONTH = re.compile(r"^(\d{4})-(\d{2})$")


@dataclass(frozen=True)
class Period:
    start: date
    end: date  # inclusive

    def as_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


def parse_month(value: str) -> tuple[int, int]:
    """'YYYY-MM' → (year, month)."""
    match = _MONTH.match(value) if isinstance(value, str) else None
    if not match or not 1 <= int(match.group(2)) <= 12:
        raise ValidationError(f"invalid month {value!r}: expected YYYY-MM")
    return int(match.group(1)), int(match.group(2))


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + month - 1 + delta
    return index // 12, index % 12 + 1


def month_period(year: int, month: int) -> Period:
    return Period(date(year, month, 1), date(year, month, monthrange(year, month)[1]))


def parse_date(value: date | str, field: str = "date") -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValidationError(f"invalid {field} {value!r}: expected YYYY-MM-DD") from None


def resolve_period(
    ctx: ToolContext,
    period: str | None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    week_start: str | None = None,
) -> Period:
    if not period:
        raise MissingFieldsError(["period"])
    today = ctx.today
    week_start = week_start or get_settings().week_start

    if period == "today":
        return Period(today, today)
    if period == "yesterday":
        day = today - timedelta(days=1)
        return Period(day, day)
    if period in ("this_week", "last_week"):
        if week_start not in WEEKDAYS:
            raise ValidationError(f"invalid WEEK_START {week_start!r}")
        start = today - timedelta(days=(today.weekday() - WEEKDAYS.index(week_start)) % 7)
        if period == "last_week":
            start -= timedelta(days=7)
        return Period(start, start + timedelta(days=6))
    if period == "this_month":
        return month_period(today.year, today.month)
    if period == "last_month":
        return month_period(*shift_month(today.year, today.month, -1))
    if period == "last_3_months":
        first = month_period(*shift_month(today.year, today.month, -2))
        return Period(first.start, month_period(today.year, today.month).end)
    if period == "this_year":
        return Period(date(today.year, 1, 1), date(today.year, 12, 31))
    if period == "custom":
        missing = [name for name, value in (("start_date", start_date), ("end_date", end_date)) if not value]
        if missing:
            raise MissingFieldsError(missing)
        start, end = parse_date(start_date, "start_date"), parse_date(end_date, "end_date")
        if start > end:
            raise ValidationError("start_date is after end_date")
        return Period(start, end)
    if isinstance(period, str) and _MONTH.match(period):
        return month_period(*parse_month(period))
    raise ValidationError(f"unknown period {period!r}: expected one of {', '.join(KEYWORDS)} or YYYY-MM")
