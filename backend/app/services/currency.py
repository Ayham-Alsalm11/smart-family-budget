"""Conversion to the family's base currency (§4, D10).

Old ↔ new Syrian pound is fixed (1 new = 100 old); USD goes through the family's `usd_rate`
(new SYP per 1 USD). The rate used is returned so that it can be stored in `rate_to_base`.
"""
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from app.db.models import CURRENCIES, Family
from app.services.errors import MissingFieldsError, ValidationError

CENT = Decimal("0.01")
RATE_STEP = Decimal("1e-10")
MAX_AMOUNT = Decimal("1e12")  # Numeric(14, 2)
MAX_RATE = Decimal("1e10")  # Numeric(20, 10)


def to_money(value: Decimal | int | float | str) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def parse_amount(value: object, field: str = "amount", allow_zero: bool = False) -> Decimal:
    if value is None or value == "":
        raise MissingFieldsError([field])
    if isinstance(value, bool):
        raise ValidationError(f"invalid {field} {value!r}")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ValidationError(f"invalid {field} {value!r}") from None
    if not amount.is_finite() or amount < 0 or (amount == 0 and not allow_zero) or amount >= MAX_AMOUNT:
        raise ValidationError(f"invalid {field} {value!r}: must be {'≥' if allow_zero else '>'} 0")
    return to_money(amount)


def parse_rate(value: object, field: str = "usd_rate") -> Decimal:
    if isinstance(value, bool):
        raise ValidationError(f"invalid {field} {value!r}", [field])
    try:
        rate = Decimal(str(value))
    except InvalidOperation:
        raise ValidationError(f"invalid {field} {value!r}", [field]) from None
    if not rate.is_finite() or rate <= 0 or rate >= MAX_RATE:
        raise ValidationError(f"invalid {field} {value!r}: must be > 0", [field])
    return rate.quantize(RATE_STEP, rounding=ROUND_HALF_UP)


def validate_currency(currency: str) -> str:
    if currency not in CURRENCIES:
        raise ValidationError(f"unknown currency {currency!r}: expected one of {', '.join(CURRENCIES)}")
    return currency


def _in_new_syp(currency: str, usd_rate: Decimal | None) -> Decimal:
    """Value of one unit of `currency` in new SYP."""
    if currency == "SYP_NEW":
        return Decimal(1)
    if currency == "SYP_OLD":
        return Decimal("0.01")
    if usd_rate is None:
        raise MissingFieldsError(["usd_rate"], "the family's USD rate is not set; a manager has to set it")
    return Decimal(usd_rate)


def convert(amount: Decimal, currency: str, base_currency: str, usd_rate: Decimal | None) -> tuple[Decimal, Decimal]:
    """Returns (amount in base currency, rate_to_base)."""
    validate_currency(currency)
    validate_currency(base_currency)
    if currency == base_currency:
        return to_money(amount), Decimal(1)
    rate = _in_new_syp(currency, usd_rate) / _in_new_syp(base_currency, usd_rate)
    return to_money(amount * rate), rate.quantize(RATE_STEP, rounding=ROUND_HALF_UP)


def to_base(amount: Decimal, currency: str, family: Family) -> tuple[Decimal, Decimal]:
    return convert(amount, currency, family.base_currency, family.usd_rate)
