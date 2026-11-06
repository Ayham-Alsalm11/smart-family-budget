"""The shared numeric scorer on the kinds of replies both systems write (§9.3)."""
import pytest

from eval.scoring import extract_numbers, score


def values(text):
    return [number.value for number in extract_numbers(text)]


@pytest.mark.parametrize("text, expected", [
    ("صرفتوا 182500 ليرة", 182500),
    ("المجموع 182,500 ل.س", 182500),
    ("١٨٢٬٥٠٠ ل.س", 182500),
    ("١٨٢٥٠٠", 182500),
    ("182 500 ليرة", 182500),
    ("182 500", 182500),
    ("182 500", 182500),
    ("182.500 ليرة", 182500),
    ("**1,250,000** ل.س", 1250000),
    ("بـ125,000 ليرة", 125000),
    ("15 ألف", 15000),
    ("١٥ الف", 15000),
    ("خمسين؟ لا، 50 تلاف", 50000),
    ("3 آلاف", 3000),
    ("26k", 26000),
    ("2 مليون", 2000000),
    ("1.2 مليون", 1200000),
    ("نص مليون", 500000),
    ("ربع مليون", 250000),
    ("مليون ونص", 1500000),
    ("مليون و200 ألف", 1200000),
    ("2 مليون و 300 ألف", 2300000),
])
def test_normalisation(text, expected):
    assert expected in values(text)


def test_a_decimal_point_before_a_multiplier_stays_a_decimal_point():
    assert values("1.250 مليون") == [1250000]


def test_dates_and_list_numbers_are_not_numbers():
    assert values("بتاريخ 2026-11-04 دفعتوا 5,000") == [5000]
    assert values("ملخص الشهر (11/2026)") == []
    assert values("1. فواتير: 7,000\n2. إيجار: 125,000") == [7000, 125000]


def test_strict_and_lenient_on_an_exact_number():
    assert score("صرفتوا 182,500 ليرة", 182500) == {"numbers": ["182500"], "has_number": True, "strict": True,
                                                    "lenient": True}


def test_a_rounded_number_is_lenient_only():
    found = score("حوالي 183 ألف", 182500)
    assert (found["strict"], found["lenient"]) == (False, True)


def test_rounding_goes_to_the_shown_precision_and_no_further():
    assert not score("180 ألف", 182500)["lenient"]
    assert not score("182 ألف", 182500)["lenient"]  # half up gives 183
    assert score("1.2 مليون", 1234567)["lenient"]
    assert not score("1.3 مليون", 1234567)["lenient"]


def test_a_plain_number_is_taken_at_its_own_precision():
    assert not score("183000", 182500)["lenient"]


def test_any_number_in_the_reply_may_match():
    assert score("دخلكم 500,000 وصرفتوا 182,500", 182500)["strict"]


def test_a_reply_without_numbers():
    assert score("ما عندي معلومات عن هالشهر", 182500) == {"numbers": [], "has_number": False, "strict": False,
                                                         "lenient": False}


def test_a_percentage_is_a_number_but_never_the_amount():
    found = score("صرفتوا 65% من الدخل", 65)
    assert found["has_number"] and not found["strict"] and not found["lenient"]


def test_a_negative_reference_is_compared_by_size():
    assert score("تجاوزتوا الحد بـ 5,000 ليرة", -5000)["strict"]
