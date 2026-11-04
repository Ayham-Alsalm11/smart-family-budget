"""Development seed: the 12 categories and a demo family with 3 months of expenses.

    cd backend
    .venv\\Scripts\\python -m app.db.seed

Passwords come from SEED_MANAGER_PASSWORD and SEED_MEMBER_PASSWORD (backend/.env or the environment).
"""
import random
import sys
from datetime import date, timedelta
from decimal import Decimal

import bcrypt
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.categories import CATEGORIES
from app.db.models import Category, Expense, Family, Member
from app.db.session import init_db, make_engine, make_session_factory
from app.settings import get_settings

DEMO_FAMILY = "عائلة السالم"
DEMO_INCOME = Decimal("500000")  # old SYP, as in GP1
DEMO_RENT = Decimal("125000")  # 25 % of the income, paid on the 1st of each month
RNG_SEED = 42
EXPENSE_COUNT = 60

# Amount range in old SYP and relative frequency for the randomly drawn expenses.
DEMO_SPENDING = {
    "أكل وشرب": (2_000, 25_000, 22),
    "بنزين ومواصلات": (3_000, 20_000, 10),
    "فواتير": (5_000, 30_000, 5),
    "صحة": (5_000, 40_000, 3),
    "تعليم": (5_000, 30_000, 2),
    "ملابس": (10_000, 60_000, 2),
    "ترفيه": (3_000, 20_000, 4),
    "هدايا ومناسبات": (5_000, 25_000, 2),
    "صيانة": (5_000, 30_000, 2),
    "متفرقات": (1_000, 10_000, 3),
}


# Short Syrian descriptions (2-5 words), so that search by meaning has something to find.
DEMO_DESCRIPTIONS = {
    "أكل وشرب": ["خضرا من سوق الهال", "لحمة من عند اللحام", "ربطة خبز وحليب", "غدا بمطعم شاورما",
                 "أغراض من البقالية", "فواكه للبيت", "سندويشات فلافل للعيلة"],
    "بنزين ومواصلات": ["تعبئة بنزين للسيارة", "تكسي عالشغل", "سرفيس للجامعة", "تغيير زيت السيارة",
                       "بنشرجي صلّح الدولاب"],
    "فواتير": ["فاتورة الكهربا", "فاتورة المي", "اشتراك النت الشهري", "تعبئة رصيد موبايل",
               "فاتورة الهاتف الأرضي"],
    "صحة": ["كشفية عند دكتور الأسنان", "أدوية من الصيدلية", "تحاليل دم بالمخبر", "مراجعة دكتور الأطفال",
            "صورة أشعة للركبة"],
    "تعليم": ["قسط المدرسة للولاد", "كتب وقرطاسية", "دورة إنكليزي", "رسوم الجامعة"],
    "ملابس": ["جاكيت شتوي للولد", "حذاء رياضة", "فستان للعيد", "ثياب مدرسة"],
    "ترفيه": ["سينما مع الولاد", "قعدة بالكافيه", "رحلة عالبحر", "اشتراك بالنادي"],
    "هدايا ومناسبات": ["هدية عرس ابن الجيران", "عيدية للولاد", "عزيمة عشا للأهل"],
    "صيانة": ["تصليح الغسالة", "سباك صلّح الحنفية", "دهان غرفة النوم", "تصليح البراد"],
    "متفرقات": ["أغراض منوعة للبيت", "شنطة جديدة", "حلاقة للولد", "مصروف متفرق"],
}
RENT_DESCRIPTION = "دفعة أجار البيت"


class SeedError(Exception):
    pass


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def seed_categories(session: Session) -> int:
    """Insert the categories that are missing. Returns how many were added."""
    existing = set(session.scalars(select(Category.name_ar)))
    added = 0
    for c in CATEGORIES:
        if c["name_ar"] in existing:
            continue
        session.add(Category(
            name_ar=c["name_ar"], name_en=c["name_en"], budget_percent=Decimal(c["budget_percent"]),
            keywords=list(c["keywords"]), icon=c["icon"],
        ))
        added += 1
    session.commit()
    return added


def _first_of_month_back(today: date, months: int) -> date:
    index = today.year * 12 + today.month - 1 - months
    return date(index // 12, index % 12 + 1, 1)


def seed_demo_family(session: Session, manager_password: str, member_password: str, today: date) -> Family:
    """The GP1 family with a manager, a member and EXPENSE_COUNT expenses over the current and previous 2 months."""
    if session.scalar(select(func.count()).select_from(Family).where(Family.name == DEMO_FAMILY)):
        raise SeedError(f"the demo family '{DEMO_FAMILY}' already exists")

    family = Family(name=DEMO_FAMILY, monthly_income=DEMO_INCOME, base_currency="SYP_OLD", usd_rate=None)
    session.add(family)
    session.flush()
    manager = Member(family_id=family.id, name="أبو أحمد", username="manager",
                     password_hash=hash_password(manager_password), role="manager")
    member = Member(family_id=family.id, name="أم أحمد", username="member",
                    password_hash=hash_password(member_password), role="member")
    session.add_all([manager, member])
    session.flush()

    categories = {c.name_ar: c for c in session.scalars(select(Category))}
    rng = random.Random(RNG_SEED)
    start = _first_of_month_back(today, 2)
    span_days = (today - start).days

    def add(category: str, amount: Decimal, when: date, description: str, who: Member) -> None:
        session.add(Expense(
            family_id=family.id, member_id=who.id, category_id=categories[category].id,
            amount=amount, currency="SYP_OLD", amount_base=amount, rate_to_base=Decimal(1),
            description=description, expense_date=when,
        ))

    rent_months = [_first_of_month_back(today, back) for back in (2, 1, 0)]
    for first in rent_months:
        add("إيجار", DEMO_RENT, first, RENT_DESCRIPTION, manager)

    names = list(DEMO_SPENDING)
    weights = [DEMO_SPENDING[n][2] for n in names]
    for _ in range(EXPENSE_COUNT - len(rent_months)):
        category = rng.choices(names, weights)[0]
        low, high, _weight = DEMO_SPENDING[category]
        amount = Decimal(rng.randrange(low, high + 1, 500))
        description = rng.choice(DEMO_DESCRIPTIONS[category])
        add(category, amount, start + timedelta(days=rng.randint(0, span_days)), description,
            rng.choice([manager, member]))

    session.commit()
    return family


def main() -> int:
    settings = get_settings()
    missing = [name for name, value in (
        ("SEED_MANAGER_PASSWORD", settings.seed_manager_password),
        ("SEED_MEMBER_PASSWORD", settings.seed_member_password),
    ) if not (value and value.get_secret_value())]
    if missing:
        print(f"Seed stopped: set {', '.join(missing)} in backend/.env or the environment.", file=sys.stderr)
        return 1

    engine = make_engine(settings.database_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        added = seed_categories(session)
        try:
            family = seed_demo_family(
                session,
                settings.seed_manager_password.get_secret_value(),
                settings.seed_member_password.get_secret_value(),
                settings.today(),
            )
        except SeedError as exc:
            print(f"Seed stopped: {exc}.", file=sys.stderr)
            return 1
        count = session.scalar(select(func.count()).select_from(Expense).where(Expense.family_id == family.id))
    engine.dispose()
    print(f"Categories added: {added}. Family #{family.id} with 2 members and {count} expenses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
