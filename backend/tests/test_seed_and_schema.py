"""The database schema and the development seed (§4)."""
from decimal import Decimal

import bcrypt
import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from app.db import seed
from app.db.categories import CATEGORIES
from app.db.models import Category, Expense, Family, Member
from app.db.session import init_db, make_engine, make_session_factory
from app.settings import Settings

from conftest import TODAY, make_family

TABLES = {"families", "members", "invites", "categories", "expenses", "budgets", "savings_goals",
          "messages", "pending_actions", "corrections", "tool_log", "llm_calls"}


def test_tables_and_indexes(session):
    inspector = inspect(session.get_bind())
    assert set(inspector.get_table_names()) == TABLES
    indexes = {tuple(i["column_names"]) for i in inspector.get_indexes("expenses")}
    assert ("family_id", "expense_date") in indexes and ("member_id",) in indexes


def test_categories_match_gp1(session):
    stored = {c.name_ar: c for c in session.scalars(select(Category))}
    assert len(stored) == 12
    assert sum(c["budget_percent"] for c in CATEGORIES) == 100
    assert stored["أكل وشرب"].budget_percent == Decimal(20) and "خضرا" in stored["أكل وشرب"].keywords
    assert seed.seed_categories(session) == 0, "seeding categories twice adds nothing"


@pytest.mark.parametrize("column, value", [("currency", "EUR"), ("amount", Decimal(0))])
def test_check_constraints(session, column, value):
    family = make_family(session)
    member = Member(family_id=family.id, name="m", username="m", password_hash="x", role="member")
    session.add(member)
    session.commit()
    values = {"family_id": family.id, "member_id": member.id, "category_id": 1, "amount": Decimal(5),
              "currency": "SYP_OLD", "amount_base": Decimal(5), "rate_to_base": Decimal(1), "expense_date": TODAY}
    session.add(Expense(**{**values, column: value}))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.add(Member(family_id=family.id, name="x", username="x", password_hash="x", role="owner"))
    with pytest.raises(IntegrityError):
        session.commit()


def seed_into(tmp_path, name):
    engine = make_engine(f"sqlite:///{(tmp_path / name).as_posix()}")
    init_db(engine)
    db = make_session_factory(engine)()
    seed.seed_categories(db)
    family = seed.seed_demo_family(db, "manager-pass", "member-pass", TODAY)
    return engine, db, family


def test_demo_family(tmp_path):
    engine, db, family = seed_into(tmp_path, "a.db")
    assert (family.name, family.base_currency, family.monthly_income, family.usd_rate) == (
        "عائلة السالم", "SYP_OLD", Decimal("500000.00"), None)
    members = {m.role: m for m in db.scalars(select(Member))}
    assert bcrypt.checkpw(b"manager-pass", members["manager"].password_hash.encode())
    assert bcrypt.checkpw(b"member-pass", members["member"].password_hash.encode())
    rows = db.scalars(select(Expense)).all()
    assert len(rows) == 60
    assert {r.currency for r in rows} == {"SYP_OLD"}
    assert min(r.expense_date for r in rows).isoformat() >= "2026-09-01"
    assert max(r.expense_date for r in rows) <= TODAY
    with pytest.raises(seed.SeedError):
        seed.seed_demo_family(db, "a", "b", TODAY)
    db.close()
    engine.dispose()


def test_demo_family_is_deterministic(tmp_path):
    def amounts(name):
        engine, db, _ = seed_into(tmp_path, name)
        result = [(r.expense_date, r.amount, r.category_id, r.member_id) for r in db.scalars(select(Expense))]
        db.close()
        engine.dispose()
        return result

    assert amounts("a.db") == amounts("b.db")


def test_seed_stops_without_passwords(tmp_path, monkeypatch, capsys):
    db_file = tmp_path / "seed.db"
    settings = Settings(_env_file=None, database_url=f"sqlite:///{db_file.as_posix()}",
                        seed_manager_password=None, seed_member_password=None)
    monkeypatch.setattr(seed, "get_settings", lambda: settings)
    assert seed.main() == 1
    assert "SEED_MANAGER_PASSWORD" in capsys.readouterr().err
    assert not db_file.exists()


def test_demo_descriptions_are_short_phrases(tmp_path):
    assert set(seed.DEMO_DESCRIPTIONS) == set(seed.DEMO_SPENDING)
    for phrases in seed.DEMO_DESCRIPTIONS.values():
        assert all(2 <= len(p.split()) <= 5 for p in phrases)
    engine, db, _ = seed_into(tmp_path, "d.db")
    descriptions = {r.description for r in db.scalars(select(Expense))}
    assert descriptions & set(seed.DEMO_DESCRIPTIONS["صحة"]), "health expenses have descriptive text"
    assert len(descriptions) >= 20
    db.close()
    engine.dispose()
