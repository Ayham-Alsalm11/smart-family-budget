from datetime import date

import pytest

from app.agent.context import ToolContext
from app.services.errors import MissingFieldsError, ValidationError
from app.services.periods import parse_month, resolve_period

from conftest import TODAY


def ctx(today: date = TODAY) -> ToolContext:
    return ToolContext(family_id=1, member_id=1, role="manager", today=today)


def span(period, today=TODAY, week_start="saturday", **kwargs):
    p = resolve_period(ctx(today), period, week_start=week_start, **kwargs)
    return p.start, p.end


@pytest.mark.parametrize("period, expected", [
    ("today", (date(2026, 11, 4), date(2026, 11, 4))),
    ("yesterday", (date(2026, 11, 3), date(2026, 11, 3))),
    ("this_week", (date(2026, 10, 31), date(2026, 11, 6))),  # Saturday → Friday
    ("last_week", (date(2026, 10, 24), date(2026, 10, 30))),
    ("this_month", (date(2026, 11, 1), date(2026, 11, 30))),
    ("last_month", (date(2026, 10, 1), date(2026, 10, 31))),
    ("last_3_months", (date(2026, 9, 1), date(2026, 11, 30))),
    ("this_year", (date(2026, 1, 1), date(2026, 12, 31))),
    ("2026-02", (date(2026, 2, 1), date(2026, 2, 28))),
])
def test_keywords(period, expected):
    assert span(period) == expected


def test_custom_range():
    assert span("custom", start_date="2026-10-05", end_date=date(2026, 10, 20)) == (date(2026, 10, 5), date(2026, 10, 20))


def test_custom_needs_both_dates():
    with pytest.raises(MissingFieldsError) as error:
        span("custom", start_date="2026-10-05")
    assert error.value.fields == ["end_date"]


def test_custom_start_after_end():
    with pytest.raises(ValidationError):
        span("custom", start_date="2026-10-20", end_date="2026-10-05")


def test_missing_and_unknown_period():
    with pytest.raises(MissingFieldsError):
        span(None)
    with pytest.raises(ValidationError):
        span("next_month")
    with pytest.raises(ValidationError):
        span("custom", start_date="05/10/2026", end_date="2026-10-20")


@pytest.mark.parametrize("today, period, expected", [
    (date(2026, 1, 1), "yesterday", (date(2025, 12, 31), date(2025, 12, 31))),
    (date(2026, 1, 15), "last_month", (date(2025, 12, 1), date(2025, 12, 31))),
    (date(2026, 1, 15), "last_3_months", (date(2025, 11, 1), date(2026, 1, 31))),
    (date(2026, 3, 31), "last_month", (date(2026, 2, 1), date(2026, 2, 28))),
    (date(2028, 3, 1), "last_month", (date(2028, 2, 1), date(2028, 2, 29))),  # leap year
    (date(2026, 12, 31), "this_month", (date(2026, 12, 1), date(2026, 12, 31))),
    (date(2026, 1, 2), "this_week", (date(2025, 12, 27), date(2026, 1, 2))),  # week across the year
    (date(2026, 1, 2), "last_week", (date(2025, 12, 20), date(2025, 12, 26))),
])
def test_month_and_year_boundaries(today, period, expected):
    assert span(period, today=today) == expected


def test_week_start_setting():
    assert span("this_week", week_start="monday") == (date(2026, 11, 2), date(2026, 11, 8))
    assert span("this_week", week_start="sunday") == (date(2026, 11, 1), date(2026, 11, 7))
    saturday = date(2026, 10, 31)
    assert span("this_week", today=saturday) == (saturday, date(2026, 11, 6))


@pytest.mark.parametrize("value", ["2026-13", "2026-1", "26-01", "", None, "2026-00"])
def test_invalid_months(value):
    with pytest.raises(ValidationError):
        parse_month(value)
