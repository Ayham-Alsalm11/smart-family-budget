"""Grounding cases taken from the first token measurement (step 5a)."""
import json
from decimal import Decimal

from app.agent.grounding import shown_numbers, ungrounded_numbers

BUDGET = json.dumps({"ok": True, "data": {"month": "2026-11", "categories": [
    {"category": "أكل وشرب", "spent": 26000, "limit": 100000, "remaining": 74000, "percent_used": 26.0},
    {"category": "فواتير", "spent": 15000, "limit": 50000, "remaining": 35000, "percent_used": 30.0},
    {"category": "تعليم", "spent": 6500, "limit": 25000, "remaining": 18500},
]}}, ensure_ascii=False)
EXPENSES = json.dumps({"ok": True, "data": {"expenses": [
    {"id": 65, "amount": 15000, "category": "فواتير", "date": "2026-11-04"},
    {"id": 64, "amount": 20000, "category": "أكل وشرب", "date": "2026-11-04"},
]}}, ensure_ascii=False)
COMPARE = json.dumps({"ok": True, "data": {"months": [{"month": "2026-10", "total": 476000},
                                                      {"month": "2026-11", "total": 182500}],
                                           "difference": -293500, "change_percent": -61.7}})


def flagged(reply, *results):
    return ungrounded_numbers(reply, list(results))


def test_thousands_words_k_and_spaces():
    assert flagged("تم تسجيل فاتورة الكهرباء بمبلغ 15 ألف.", EXPENSES) == []
    assert flagged("أكل وشرب: 26k / 100k (74k متبقي)", BUDGET) == []
    assert flagged("15 000 ل. فواتير و ٢٠ ٠٠٠ أكل", EXPENSES) == []
    assert [n.value for n in shown_numbers("١٥ ألف و 1.5 مليون")] == [15000, 1500000]


def test_list_numbering_is_ignored():
    reply = "آخر مصاريف:\n1. 15,000 فواتير\n2. 20,000 أكل\n3) 15 000\n- 4. 20000"
    assert flagged(reply, EXPENSES) == []


def test_rounded_values_are_accepted():
    assert flagged("صرفنا تقريباً 183 ألف هالشهر، أقل بـ 62%", COMPARE) == []
    assert flagged("الفرق 293.5 ألف", COMPARE) == []
    assert flagged("صرفنا 185 ألف", COMPARE) == [Decimal(185000)], "2,500 away is more than rounding"


def test_percentages_from_results_or_from_two_numbers_of_one_row():
    assert flagged("الأكل 26% والفواتير 30%", BUDGET) == []
    assert flagged("التعليم 26% من الحد", BUDGET) == [], "6,500 / 25,000 in the same row"
    assert flagged("وفّر 30–40% من الفئات", BUDGET) == [Decimal(40)], "the invented 40 % is still caught"


def test_years_only_from_dates_in_results():
    assert flagged("آخر مصروف (15000 ل.س، تاريخ 2024‑04‑04)", EXPENSES) == [Decimal(2024)]
    assert flagged("آخر مصروف بتاريخ 2026-11-04", EXPENSES) == []


def test_invented_numbers_are_still_flagged():
    assert flagged("صرفت ١٢٬٣٤٥ ليرة", EXPENSES) == [Decimal(12345)]
    assert flagged("بتوفر 5000 بالشهر", BUDGET) == [Decimal(5000)]
