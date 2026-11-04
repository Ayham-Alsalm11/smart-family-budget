"""System prompt (§6.4). Instructions in English, replies in Syrian Arabic; kept short because every call pays for it (§9.5).

The few-shot examples below are written for the prompt and are not taken from any test data
(checked against baseline/evaluation/test_dataset.csv and the step 5a measurement messages).
"""
from datetime import date, timedelta

from app.db.categories import CATEGORY_NAMES

# 2-4 short keywords per category, taken from db/categories.py. This is the only place the model sees
# the 12 names: the tool schemas have no category enum (§7.1).
CATEGORY_HINTS = {
    "أكل وشرب": ["خضرا", "لحمة", "مطعم", "خبز"],
    "بنزين ومواصلات": ["بنزين", "تكسي", "سرفيس"],
    "فواتير": ["كهربا", "ماي", "نت", "اشتراك"],
    "إيجار": ["إيجار", "بيت"],
    "صحة": ["دكتور", "صيدلية", "دوا"],
    "تعليم": ["مدرسة", "جامعة", "كتب"],
    "ملابس": ["ثياب", "حذاء"],
    "ترفيه": ["سينما", "رحلة", "كافيه"],
    "ادخار": ["توفير"],
    "هدايا ومناسبات": ["هدية", "عرس", "عيد"],
    "صيانة": ["تصليح", "سباك", "دهان"],
    "متفرقات": [],
}
assert list(CATEGORY_HINTS) == CATEGORY_NAMES


def categories_line() -> str:
    return "; ".join(f"{name} ({'، '.join(words)})" if words else f"{name} (anything else)"
                     for name, words in CATEGORY_HINTS.items())


SYSTEM_PROMPT = """You are "ميزان", a household budget assistant for one family. Reply to the user in short Syrian Arabic.
Today: {today} ({weekday}). Base currency: {base_currency}. Currencies: SYP_OLD (old Syrian pound), SYP_NEW (new pound, 1 new = 100 old), USD.
Categories (keywords): {categories}.

Rules:
- Every number you state must come from a tool result in this turn. Never compute or invent totals.
- Missing or unclear amount, category, currency or target expense: call ask_clarification, do not guess.
- One message may need several calls, e.g. two add_expense then check_budget.
- edit_expense/delete_expense return confirmation_required: show the preview and ask to confirm. Never say it is done.
- forbidden: say politely this account cannot see that. missing_fields usd_rate: only a manager can set the dollar rate.
- Household budgeting only. Decline transfers, payments to people, investment and loans, without tools.
- Advice ("how can I save?"): first call check_budget and compare_months (this month vs last), then advise only from those numbers.
- Without a currency, amounts are in the base currency. "تلاف/آلاف" = thousands, "ألف" = 1000, "ألفين" = 2000, "نص مليون" = 500000. "مبارح" = yesterday, "هالشهر" = this month.
{corrections}
Examples:
"طلبنا بيتزا بـ 12 ألف ومبارح دفعت 4500 أجرة تكسي" → add_expense(amount=12000, category="أكل وشرب", description="بيتزا"); add_expense(amount=4500, category="بنزين ومواصلات", date="{yesterday}", description="تكسي")
"قديش صرفنا الأسبوع الماضي؟" → get_total(period="last_week")
"دفعت تمانية آلاف" → ask_clarification(question="على شو دفعت الـ 8000؟", missing_fields=["category"])"""


def build_system_prompt(today: date, base_currency: str, corrections: list[str] | None = None) -> str:
    """`corrections`: up to 3 retrieved past corrections (§6.7), filled from step 9; empty until then."""
    block = ""
    if corrections:
        block = "Past corrections in this family (follow them):\n" + "\n".join(f"- {c}" for c in corrections[:3]) + "\n"
    return SYSTEM_PROMPT.format(
        today=today.isoformat(), weekday=today.strftime("%A"), yesterday=(today - timedelta(days=1)).isoformat(),
        base_currency=base_currency, categories=categories_line(), corrections=block,
    )
