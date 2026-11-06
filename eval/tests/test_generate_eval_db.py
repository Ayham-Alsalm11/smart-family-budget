"""The evaluation records and the GP2 database built from them (§9.2). GP1's side needs Ollama, so it runs in the real
generation, where the two counts are compared."""
from datetime import date

import pytest
from sqlalchemy import func, select

from eval.generate_eval_db import OUTPUTS, build_gp2, generate, generate_records, write_records

from app.db.models import Budget, Expense, Member
from app.db.session import make_engine, make_session_factory

TODAY = date(2026, 11, 6)


class FakeEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]


def test_the_same_seed_and_day_give_the_same_file(tmp_path):
    first = write_records(generate_records(TODAY), tmp_path / "a.json")
    second = write_records(generate_records(TODAY), tmp_path / "b.json")
    third = write_records(generate_records(TODAY, 7), tmp_path / "c.json")
    assert first == second != third


def test_twelve_months_up_to_today_in_one_currency():
    records = generate_records(TODAY)
    days = [item["date"] for item in records["expenses"]]
    assert days == sorted(days) and max(days) <= TODAY.isoformat()
    assert {day[:7] for day in days} == {"2025-12", *(f"2026-{month:02d}" for month in range(1, 12))}
    assert {item["currency"] for item in records["expenses"]} == {"SYP_OLD"}
    assert {item["member"] for item in records["expenses"]} == {"manager", "member"}


def test_rent_every_month_and_a_month_near_the_income():
    records = generate_records(TODAY)
    rents = [item for item in records["expenses"] if item["category"] == "إيجار"]
    assert len(rents) == 12 and {item["date"][8:] for item in rents} == {"01"}
    october = sum(item["amount"] for item in records["expenses"] if item["date"].startswith("2026-10"))
    assert 300_000 < october < 800_000


def test_the_gp2_database_holds_the_records(tmp_path):
    records = generate_records(TODAY)
    assert build_gp2(records, tmp_path, FakeEmbedder()) == {"expenses": len(records["expenses"]),
                                                            "vectors": len(records["expenses"])}
    engine = make_engine(f"sqlite:///{(tmp_path / 'gp2.db').as_posix()}")
    with make_session_factory(engine)() as session:
        total = session.scalar(select(func.sum(Expense.amount_base)))
        assert float(total) == sum(item["amount"] for item in records["expenses"])
        assert session.scalar(select(func.count()).select_from(Member)) == 2
        assert session.scalar(select(func.count()).select_from(Budget)) == 0
    engine.dispose()


def test_an_earlier_database_is_kept_without_overwrite(tmp_path):
    (tmp_path / OUTPUTS[0]).write_text("kept", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate(TODAY, 42, tmp_path, False, FakeEmbedder())
    assert (tmp_path / OUTPUTS[0]).read_text(encoding="utf-8") == "kept"
