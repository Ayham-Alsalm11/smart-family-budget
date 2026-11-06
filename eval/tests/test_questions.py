"""Reference answers on small records written by hand (§9.2), so each expected number can be checked by eye."""
import json

import pytest

from eval import DATA_DIR
from eval.questions import TYPES, load_questions, reference

FOOD = "أكل وشرب"
FUEL = "بنزين ومواصلات"
RENT = "إيجار"
RECORDS = {
    "today": "2026-11-06",  # a Friday: this week started on Saturday 2026-10-31, in the month before
    "family": {"monthly_income": 500_000},
    "categories": [{"name": FOOD, "budget_percent": 20}, {"name": FUEL, "budget_percent": 15}],
    "expenses": [
        {"date": "2026-11-01", "category": RENT, "amount": 125_000},
        {"date": "2026-11-02", "category": FOOD, "amount": 20_000},
        {"date": "2026-11-06", "category": FUEL, "amount": 5_000},
        {"date": "2026-10-31", "category": FOOD, "amount": 7_000},
        {"date": "2026-10-30", "category": FUEL, "amount": 10_000},
        {"date": "2026-10-01", "category": RENT, "amount": 125_000},
        {"date": "2026-09-15", "category": FOOD, "amount": 3_000},
        {"date": "2026-08-31", "category": FOOD, "amount": 1_000},  # just before the last 3 months
        {"date": "2026-07-10", "category": FOOD, "amount": 40_000},
        {"date": "2025-12-10", "category": FOOD, "amount": 9_000},
    ],
}


def value(kind, **params):
    return reference({"type": kind, "params": params}, RECORDS)["value"]


def test_this_month():
    assert value("total_this_month") == 150_000
    assert value("category_this_month", category=FOOD) == 20_000
    assert value("category_this_month", category="ملابس") == 0


def test_last_month():
    assert value("total_last_month") == 142_000
    assert value("category_last_month", category=FUEL) == 10_000


def test_a_named_month_is_its_latest_past_occurrence():
    assert value("total_named_month", month=7) == 40_000
    assert value("total_named_month", month=12) == 9_000  # December 2026 is in the future
    assert reference({"type": "total_named_month", "params": {"month": 12}}, RECORDS)["detail"] == "2025-12"


def test_difference_has_no_sign():
    assert value("difference_this_last") == 8_000


def test_budget_remaining_is_income_times_percent_minus_spent():
    assert value("budget_remaining", category=FOOD) == 80_000
    assert value("budget_remaining", category=FUEL) == 70_000


def test_top_category():
    found = reference({"type": "top_category", "params": {}}, RECORDS)
    assert (found["value"], found["detail"]) == (125_000, RENT)


def test_a_tie_in_the_top_category_does_not_depend_on_row_order():
    tied = {**RECORDS, "expenses": [{"date": "2026-11-02", "category": FUEL, "amount": 5_000},
                                    {"date": "2026-11-03", "category": FOOD, "amount": 5_000}]}
    assert reference({"type": "top_category", "params": {}}, tied)["detail"] == FOOD


def test_last_3_months_are_this_month_and_the_two_before():
    assert value("category_last_3_months", category=FOOD) == 30_000


def test_this_week_starts_on_saturday_and_crosses_the_month():
    assert value("total_this_week") == 157_000


def test_ten_pilot_questions_of_ten_types():
    questions = load_questions(DATA_DIR / "pilot_numeric_questions.jsonl")
    assert sorted(q["type"] for q in questions) == sorted(TYPES)
    assert {q["source"] for q in questions} == {"pilot"}
    assert all("reference" not in q for q in questions)


def test_a_pilot_question_cannot_enter_the_final_set(tmp_path):
    final = tmp_path / "numeric_questions.jsonl"
    final.write_text(json.dumps({"id": "P01", "type": TYPES[0], "question": "?", "source": "pilot"}), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot"):
        load_questions(final)
