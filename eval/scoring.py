"""The numeric scorer (§9.3): one scorer for GP1 and the agent, and independent of both.

It does not reuse the agent's grounding code on purpose: the scorer would then change whenever the system under test
does. Both record the same rules of writing numbers; they answer different questions.

- strict: a number in the reply equals the reference exactly.
- lenient: a number equals the reference after rounding it to the precision the number is written with, half up:
  "183 ألف" passes for 182,500, and "180 ألف" does not.
Normalised first: Arabic digits, "," and spaces between thousands, "." between thousands as in "182.500", "ألف/آلاف/تلاف/k",
"مليون", and the spoken forms "نص مليون", "ربع مليون", "مليون ونص", "مليون و200 ألف". Dates and list numbering are removed,
and percentages never match, because a reference here is an amount. Signs are ignored: a reply states the direction in
words ("تجاوزت بـ 5000"), so the reference is compared by its size.
"""
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٬٫", "01234567890123456789,.")
_GAP = "[    ]"  # a space, a no-break space, a narrow no-break space and a thin space
_THOUSAND = "(?:ألف|الف|آلاف|الاف|تلاف|تالاف)"
_MILLION = "(?:مليون|ملايين)"
_MULTIPLIERS = {"ألف": 1000, "الف": 1000, "آلاف": 1000, "الاف": 1000, "تلاف": 1000, "تالاف": 1000, "k": 1000, "K": 1000,
                "مليون": 1_000_000, "ملايين": 1_000_000}
_DATES = re.compile(r"\d{4}-\d{1,2}(?:-\d{1,2})?|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{4}|\d{4}/\d{1,2}(?:/\d{1,2})?")
_LIST_NUMBER = re.compile(r"(?m)^[ \t]*(?:[-*•][ \t]*)?\d{1,2}[.)][ \t]+")
_MILLION_AND_THOUSANDS = re.compile(rf"(?:(\d+)\s?)?{_MILLION}\s+و\s?(\d+)\s?{_THOUSAND}")
_SPOKEN = ((re.compile(rf"نص\s+{_MILLION}"), "0.5 مليون"), (re.compile(rf"ربع\s+{_MILLION}"), "0.25 مليون"),
           (re.compile(rf"(?<![\d.])\s*{_MILLION}\s+ونص"), " 1.5 مليون"))
_NUMBER = re.compile(
    rf"(?<![\d.,])(\d{{1,3}}(?:(?:,|{_GAP}|\.)\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)"
    rf"(?:{_GAP}?({_THOUSAND}|{_MILLION}|k(?![A-Za-z])|K(?![A-Za-z])))?"
    rf"(?:{_GAP}?([%٪]))?"
)
_DOTS_AS_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")


@dataclass(frozen=True)
class Number:
    value: Decimal
    step: Decimal  # the precision it is written with: 1 for "182500", 1000 for "183 ألف", 100000 for "1.2 مليون"
    percent: bool = False


def _expand_million_and_thousands(match: re.Match) -> str:
    return f"{int(match.group(1) or 1) * 1_000_000 + int(match.group(2)) * 1000}"


def _normalised(text: str) -> str:
    text = _DATES.sub(" ", (text or "").translate(_DIGITS))
    text = _LIST_NUMBER.sub("", text)
    text = _MILLION_AND_THOUSANDS.sub(_expand_million_and_thousands, text)
    for pattern, spoken in _SPOKEN:
        text = pattern.sub(spoken, text)
    return text


def extract_numbers(text: str) -> list[Number]:
    numbers = []
    for match in _NUMBER.finditer(_normalised(text)):
        digits, word, percent = match.groups()
        # "182.500" alone is Syrian writing for 182,500; "1.250 مليون" has a decimal point.
        if _DOTS_AS_THOUSANDS.fullmatch(digits) and not word:
            digits = digits.replace(".", "")
        clean = re.sub(rf",|{_GAP}", "", digits)
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        multiplier = Decimal(_MULTIPLIERS[word]) if word else Decimal(1)
        numbers.append(Number(Decimal(clean) * multiplier, Decimal(1).scaleb(-decimals) * multiplier, bool(percent)))
    return numbers


def _rounds_to(number: Number, reference: Decimal) -> bool:
    units = (reference / number.step).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return units * number.step == number.value


def score(reply: str, reference: Any) -> dict[str, Any]:
    """`numbers` are all those found in the reply, percentages included, so that a reader sees what the scorer saw."""
    target = abs(Decimal(str(reference)))
    numbers = extract_numbers(reply)
    amounts = [number for number in numbers if not number.percent]
    strict = any(number.value == target for number in amounts)
    return {
        "numbers": [format(number.value.normalize(), "f") for number in numbers],
        "has_number": bool(numbers),
        "strict": strict,
        "lenient": strict or any(_rounds_to(number, target) for number in amounts),
    }
