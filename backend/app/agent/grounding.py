"""Grounding check (§6.9): numbers in the final reply that no tool result of the turn contains.

Arabic-Indic digits and thousands separators are normalised; 5,000 / ٥٠٠٠ / 5000.00 are the same number.
"""
import re
from decimal import Decimal, InvalidOperation

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٬٫", "01234567890123456789,.")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_GROUPED = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def extract_numbers(text: str) -> set[Decimal]:
    numbers: set[Decimal] = set()
    for token in _NUMBER.findall((text or "").translate(_DIGITS)):
        parts = [token.replace(",", "")] if _GROUPED.fullmatch(token) else token.split(",")
        for part in parts:
            try:
                numbers.add(Decimal(part))
            except InvalidOperation:
                continue
    return numbers


def ungrounded_numbers(reply: str, tool_results: list[str]) -> list[Decimal]:
    known: set[Decimal] = set()
    for text in tool_results:
        known |= extract_numbers(text)
    return sorted(n for n in extract_numbers(reply) if n not in known)
