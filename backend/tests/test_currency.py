"""Conversion to the family's base currency (§4, D10)."""
from decimal import Decimal

import pytest

from app.services import expenses
from app.services.currency import convert, parse_amount
from app.services.errors import MissingFieldsError, ValidationError

from conftest import ctx_for, make_family, make_member

D = Decimal


@pytest.mark.parametrize("amount, currency, base, usd_rate, expected_base, expected_rate", [
    ("5000", "SYP_OLD", "SYP_OLD", None, "5000.00", "1"),
    ("50", "SYP_NEW", "SYP_OLD", None, "5000.00", "100"),
    ("5000", "SYP_OLD", "SYP_NEW", None, "50.00", "0.01"),
    ("10", "USD", "SYP_NEW", "110", "1100.00", "110"),
    ("10", "USD", "SYP_OLD", "110", "110000.00", "11000"),
    ("1100", "SYP_NEW", "USD", "110", "10.00", "0.0090909091"),
    ("110000", "SYP_OLD", "USD", "110", "10.00", "0.0000909091"),
])
def test_conversion_and_rate(amount, currency, base, usd_rate, expected_base, expected_rate):
    amount_base, rate = convert(D(amount), currency, base, D(usd_rate) if usd_rate else None)
    assert amount_base == D(expected_base)
    assert rate == D(expected_rate)


def test_usd_without_rate_asks_for_the_rate():
    with pytest.raises(MissingFieldsError) as error:
        convert(D("10"), "USD", "SYP_OLD", None)
    assert error.value.fields == ["usd_rate"]
    with pytest.raises(MissingFieldsError):
        convert(D("1000"), "SYP_OLD", "USD", None)


def test_unknown_currency():
    with pytest.raises(ValidationError):
        convert(D("10"), "EUR", "SYP_OLD", None)


@pytest.mark.parametrize("value", [0, -5, "abc", True, float("nan"), "1e12"])
def test_invalid_amounts(value):
    with pytest.raises(ValidationError):
        parse_amount(value)


def test_amount_rounded_to_cents():
    assert parse_amount(12.345) == D("12.35")
    assert parse_amount("7000") == D("7000.00")


def test_rate_to_base_is_stored_and_not_rewritten(session):
    family = make_family(session, base_currency="SYP_OLD", usd_rate="110")
    ctx = ctx_for(make_member(session, family, "m", "manager"))
    added = expenses.add_expense(session, ctx, amount=10, category="فواتير", currency="USD")
    assert added["amount_base"] == D("110000.00")
    assert added["rate_to_base"] == D("11000")

    family.usd_rate = D("130")
    session.commit()
    assert expenses.get_expense(session, ctx, added["id"])["amount_base"] == D("110000.00")

    edited = expenses.update_expense(session, ctx, added["id"], {"amount": 20})
    assert edited["amount_base"] == D("220000.00"), "changing only the amount keeps the stored rate"
    edited = expenses.update_expense(session, ctx, added["id"], {"currency": "USD"})
    assert edited["rate_to_base"] == D("13000"), "changing the currency uses the current rate"


def test_usd_expense_without_family_rate(session):
    family = make_family(session, usd_rate=None)
    ctx = ctx_for(make_member(session, family, "m", "member"))
    with pytest.raises(MissingFieldsError) as error:
        expenses.add_expense(session, ctx, amount=10, category="فواتير", currency="USD")
    assert error.value.fields == ["usd_rate"]
