"""Grounding cases taken from the first token measurement of step 5a, and the year correction (§6.9)."""
import json
from decimal import Decimal

from app.agent.grounding import correct_years, shown_numbers, ungrounded_numbers

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


# ---------- correct_years ----------

LISTED = json.dumps({"expenses": [{"id": 3, "date": "2026-11-04"}, {"id": 2, "date": "2026-10-02"}]})


def test_a_wrong_year_takes_the_year_of_the_results():
    reply, corrected = correct_years("دفعت 5000 بتاريخ 2024-11-04", [LISTED])
    assert reply == "دفعت 5000 بتاريخ 2026-11-04"
    assert corrected == [{"from": "2024-11-04", "to": "2026-11-04"}]


def test_arabic_digits_in_a_wrong_date():
    reply, corrected = correct_years("بتاريخ ٢٠٢٤-١١-٠٤", [LISTED])
    assert reply == "بتاريخ 2026-11-04" and corrected[0]["from"] == "2024-11-04"


def test_a_right_date_and_an_unknown_day_stay():
    text = "2026-11-04 و 2024-03-09"
    assert correct_years(text, [LISTED]) == (text, [])


def test_a_date_the_user_wrote_stays():
    text = "ما في شي بتاريخ 2024-11-04"
    assert correct_years(text, [LISTED], "شو صرفنا بـ 2024-11-04؟") == (text, [])


def test_the_same_day_under_two_years_stays():
    both = json.dumps({"expenses": [{"date": "2026-11-04"}, {"date": "2025-11-04"}]})
    assert correct_years("2024-11-04", [both]) == ("2024-11-04", [])
