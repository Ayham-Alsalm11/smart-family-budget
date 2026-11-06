"""Grounding check (§6.9): numbers in the final reply that the turn's tool results do not support.

Reply side:
- Arabic-Indic digits, "٬"/"," and spaces between groups of three digits are normalised ("١٥ ٠٠٠" = 15000).
- "ألف/آلاف/تلاف/k" multiply by 1,000 and "مليون" by 1,000,000 ("15 ألف" = 15000, "26k" = 26000).
- List numbering at the start of a line ("1." "2)") is ignored.
- Each number keeps the precision it is shown with, so a rounded value is accepted: "183 ألف" for 182,500,
  "65%" for 65.4.
- A percentage must match a percentage in the results, or the ratio of two numbers of the same result
  object, within ±0.1 (or the shown precision).
- A year (2000–2100) is accepted only if a date in the results has it, so a wrong year is still caught.
Tool side: every number in the results, the parts of every date, and ratios inside each object.
The user's current message also counts: repeating a number they said is not inventing one.

`correct_years` runs first and fixes the one mistake that needs no repair request: a date whose month and day are in
the results with another year.
"""
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import permutations
from typing import Any

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٬٫", "01234567890123456789,.")
_LIST_NUMBER = re.compile(r"(?m)^[ \t]*(?:[-*•][ \t]*)?\d{1,2}[.)][ \t]+")
_MULTIPLIERS = {"ألف": 1000, "الف": 1000, "آلاف": 1000, "الاف": 1000, "تلاف": 1000, "k": 1000, "K": 1000,
                "مليون": 1_000_000, "ملايين": 1_000_000}
_NUMBER = re.compile(
    r"(?<![\d.])(\d{1,3}(?:[,   ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)"
    r"(?:[  ]?(ألف|الف|آلاف|الاف|تلاف|مليون|ملايين|k(?![A-Za-z])|K(?![A-Za-z])))?"
    r"(?:[  ]?([%٪]))?"
)
_DATE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")
_FULL_DATE = re.compile(r"(?<![0-9٠-٩۰-۹-])[0-9٠-٩۰-۹]{4}-[0-9٠-٩۰-۹]{2}-[0-9٠-٩۰-۹]{2}(?![0-9٠-٩۰-۹])")
_PLAIN = re.compile(r"\d+(?:\.\d+)?")
PERCENT_TOLERANCE = Decimal("0.1")


@dataclass(frozen=True)
class ShownNumber:
    value: Decimal
    step: Decimal  # precision as shown: 1 for "15000", 1000 for "26k", 0.1 for "61.7"
    percent: bool

    @property
    def is_year(self) -> bool:
        return not self.percent and self.step == 1 and 2000 <= self.value <= 2100


def shown_numbers(text: str) -> list[ShownNumber]:
    text = _LIST_NUMBER.sub("", (text or "").translate(_DIGITS))
    numbers = []
    for match in _NUMBER.finditer(text):
        digits, word, percent = match.groups()
        clean = re.sub(r"[,   ]", "", digits)
        try:
            value = Decimal(clean)
        except InvalidOperation:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        multiplier = Decimal(_MULTIPLIERS[word]) if word else Decimal(1)
        numbers.append(ShownNumber(value * multiplier, Decimal(1).scaleb(-decimals) * multiplier, bool(percent)))
    return numbers


def extract_numbers(text: str) -> set[Decimal]:
    return {n.value for n in shown_numbers(text)}


@dataclass
class Evidence:
    values: set[Decimal]
    percents: set[Decimal]
    years: set[int]


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return Decimal(str(value))


def _collect(node: Any, evidence: Evidence) -> None:
    if isinstance(node, dict):
        direct = []
        for key, value in node.items():
            number = _number(value)
            if number is not None:
                direct.append(number)
                if "percent" in key or "share" in key:
                    evidence.percents.add(number)
            else:
                _collect(value, evidence)
        evidence.values.update(direct)
        for a, b in permutations(direct, 2):
            if b != 0:
                evidence.percents.add(a / b * 100)
    elif isinstance(node, list):
        for item in node:
            _collect(item, evidence)
    elif isinstance(node, str):
        for match in _DATE.finditer(node):
            evidence.years.add(int(match.group(1)))
        evidence.values.update(Decimal(n) for n in _PLAIN.findall(node.translate(_DIGITS)))
    else:
        number = _number(node)
        if number is not None:
            evidence.values.add(number)


def collect_evidence(tool_results: list[str]) -> Evidence:
    evidence = Evidence(set(), set(), set())
    for text in tool_results:
        try:
            _collect(json.loads(text), evidence)
        except ValueError:
            _collect(text, evidence)
    return evidence


def _close(number: ShownNumber, candidates: set[Decimal], tolerance: Decimal) -> bool:
    """Signs are compared away: replies state the direction in words ("نقص 61.7%" for -61.7)."""
    return any(abs(abs(number.value) - abs(c)) <= tolerance for c in candidates)


def correct_years(reply: str, tool_results: list[str], user_message: str = "") -> tuple[str, list[dict[str, str]]]:
    """Put the results' year back into a reply's date (§6.9). Returns the reply and what changed.

    The model keeps writing 2024 for 2026 (S6), and each time it cost a repair request. A date stays as it is if the
    results or the user wrote it, and also if its month and day appear in the results under two years: over 12 months
    of history that is possible, and then only the model knows which one it meant.
    """
    found = {match.group().translate(_DIGITS) for text in tool_results for match in _FULL_DATE.finditer(text)}
    said = {match.group().translate(_DIGITS) for match in _FULL_DATE.finditer(user_message)}
    corrections = []

    def fix(match: re.Match) -> str:
        shown = match.group().translate(_DIGITS)
        same_day = [day for day in found if day[5:] == shown[5:]]
        if shown in found or shown in said or len(same_day) != 1:
            return match.group()
        corrections.append({"from": shown, "to": same_day[0]})
        return same_day[0]

    return _FULL_DATE.sub(fix, reply), corrections


def ungrounded_numbers(reply: str, tool_results: list[str], user_message: str = "") -> list[Decimal]:
    evidence = collect_evidence(tool_results)
    for number in shown_numbers(user_message):
        (evidence.percents if number.percent else evidence.values).add(number.value)
        if number.is_year:
            evidence.years.add(int(number.value))
    flagged = set()
    for number in shown_numbers(reply):
        half_step = number.step / 2
        if number.percent:
            ok = _close(number, evidence.percents, max(PERCENT_TOLERANCE, half_step))
        elif number.is_year:
            ok = int(number.value) in evidence.years
        else:
            ok = _close(number, evidence.values, half_step)
        if not ok:
            flagged.add(number.value)
    return sorted(flagged)
