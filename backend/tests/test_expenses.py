from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from app.db.models import Expense
from app.rag.vector_store import scope_filter
from app.services import expenses
from app.services.errors import ForbiddenError, MissingFieldsError, NotFoundError, ValidationError

from conftest import TODAY, FailingEmbedder, ctx_for, make_family, make_member


def count(session) -> int:
    return session.scalar(select(func.count()).select_from(Expense))


def add(session, ctx, amount=5000, category="بنزين ومواصلات", **kwargs):
    return expenses.add_expense(session, ctx, amount=amount, category=category, **kwargs)


# ---------- adding ----------

def test_add_defaults(session, member_ctx, member):
    added = add(session, member_ctx, description="بنزين")
    assert added["member_id"] == member.id
    assert added["currency"] == "SYP_OLD"
    assert added["amount"] == Decimal("5000.00")
    assert added["expense_date"] == TODAY.isoformat()
    assert added["embedded"] is False


@pytest.mark.parametrize("category", ["بنزين", "Fuel & Transport", "متفرقاتx", " فواتير", 7])
def test_invalid_category_is_rejected(session, member_ctx, category):
    with pytest.raises(ValidationError):
        add(session, member_ctx, category=category)
    assert count(session) == 0


@pytest.mark.parametrize("field", ["amount", "category"])
@pytest.mark.parametrize("value", [None, ""])
def test_missing_fields(session, member_ctx, field, value):
    kwargs = {"amount": 5000, "category": "فواتير", field: value}
    with pytest.raises(MissingFieldsError) as error:
        expenses.add_expense(session, member_ctx, **kwargs)
    assert error.value.fields == [field]
    assert count(session) == 0


def test_explicit_date_and_invalid_date(session, member_ctx):
    assert add(session, member_ctx, expense_date="2026-10-15")["expense_date"] == "2026-10-15"
    with pytest.raises(ValidationError):
        add(session, member_ctx, expense_date="15-10-2026")


def test_goal_only_for_savings(session, member_ctx):
    with pytest.raises(ValidationError):
        add(session, member_ctx, category="أكل وشرب", goal_id=1)


# ---------- permissions (§5) ----------

def test_member_cannot_see_or_change_another_members_expense(session, member_ctx, other_member_ctx):
    theirs = add(session, other_member_ctx)
    with pytest.raises(ForbiddenError):
        expenses.get_expense(session, member_ctx, theirs["id"])
    with pytest.raises(ForbiddenError):
        expenses.update_expense(session, member_ctx, theirs["id"], {"amount": 1})
    with pytest.raises(ForbiddenError):
        expenses.delete_expense(session, member_ctx, theirs["id"])
    assert expenses.get_expense(session, other_member_ctx, theirs["id"])["amount"] == Decimal("5000.00")


def test_member_can_change_own_expense(session, member_ctx):
    mine = add(session, member_ctx)
    edited = expenses.update_expense(session, member_ctx, mine["id"],
                                     {"amount": 7000, "category": "أكل وشرب", "date": "2026-11-01",
                                      "description": "غدا"})
    assert (edited["amount"], edited["category"], edited["expense_date"], edited["description"]) == (
        Decimal("7000.00"), "أكل وشرب", "2026-11-01", "غدا")
    expenses.delete_expense(session, member_ctx, mine["id"])
    assert count(session) == 0


def test_manager_can_change_members_expense(session, manager_ctx, member_ctx):
    theirs = add(session, member_ctx)
    assert expenses.get_expense(session, manager_ctx, theirs["id"])["id"] == theirs["id"]
    assert expenses.update_expense(session, manager_ctx, theirs["id"], {"amount": 1})["amount"] == Decimal("1.00")
    expenses.delete_expense(session, manager_ctx, theirs["id"])
    assert count(session) == 0


def test_other_family_is_not_found(session, member_ctx):
    stranger = make_member(session, make_family(session), "stranger", "manager")
    mine = add(session, member_ctx)
    with pytest.raises(NotFoundError):
        expenses.get_expense(session, ctx_for(stranger), mine["id"])
    with pytest.raises(NotFoundError):
        expenses.delete_expense(session, ctx_for(stranger), mine["id"])


def test_update_whitelist(session, manager_ctx):
    mine = add(session, manager_ctx)
    for changes in ({"family_id": 2}, {"member_id": 5}, {"amount_base": 1}, {}):
        with pytest.raises(ValidationError):
            expenses.update_expense(session, manager_ctx, mine["id"], changes)


def test_list_scoping(session, manager_ctx, member_ctx, other_member_ctx, other_member):
    add(session, member_ctx, amount=1000)
    add(session, other_member_ctx, amount=2000)
    add(session, manager_ctx, amount=3000)

    own = expenses.list_expenses(session, member_ctx)["expenses"]
    assert [e["amount"] for e in own] == [Decimal("1000.00")]
    assert len(expenses.list_expenses(session, member_ctx, member="me")["expenses"]) == 1
    with pytest.raises(ForbiddenError):
        expenses.list_expenses(session, member_ctx, member=other_member.id)
    with pytest.raises(ForbiddenError):
        expenses.list_expenses(session, member_ctx, member="member2")

    assert len(expenses.list_expenses(session, manager_ctx)["expenses"]) == 3
    filtered = expenses.list_expenses(session, manager_ctx, member="member2")["expenses"]
    assert [e["amount"] for e in filtered] == [Decimal("2000.00")]


def test_list_order_period_category_and_limit(session, manager_ctx):
    add(session, manager_ctx, amount=1, expense_date="2026-10-30")
    add(session, manager_ctx, amount=2, expense_date="2026-11-02", category="فواتير")
    add(session, manager_ctx, amount=3, expense_date="2026-11-03")
    listed = expenses.list_expenses(session, manager_ctx, period="this_month")
    assert [e["amount"] for e in listed["expenses"]] == [Decimal("3.00"), Decimal("2.00")]
    assert listed["period"] == {"start": "2026-11-01", "end": "2026-11-30"}
    assert len(expenses.list_expenses(session, manager_ctx, category="فواتير")["expenses"]) == 1
    assert len(expenses.list_expenses(session, manager_ctx, limit=1)["expenses"]) == 1
    with pytest.raises(ValidationError):
        expenses.list_expenses(session, manager_ctx, limit=0)


# ---------- embeddings (A3) ----------

def test_failed_embedding_does_not_fail_the_expense(session, member_ctx, fake_store):
    added = add(session, member_ctx, embedder=FailingEmbedder(), vector_store=fake_store)
    stored = session.get(Expense, added["id"])
    assert stored is not None
    assert stored.embedded_at is None
    assert added["embedded"] is False
    assert fake_store.items == {}


def test_successful_embedding_has_typed_metadata(session, member_ctx, member, family, fake_embedder, fake_store):
    added = add(session, member_ctx, description="بنزين", source_text="دفعت 5000 بنزين",
                embedder=fake_embedder, vector_store=fake_store)
    assert added["embedded"] is True
    assert session.get(Expense, added["id"]).embedded_at.tzinfo is not None
    metadata = fake_store.items[("expenses", str(added["id"]))]["metadata"]
    assert metadata["family_id"] == family.id and isinstance(metadata["family_id"], int)
    assert metadata["member_id"] == member.id and isinstance(metadata["member_id"], int)
    assert metadata["date_key"] == 20261104


def test_reindex_missing(session, member_ctx, fake_embedder, fake_store):
    first = add(session, member_ctx, embedder=FailingEmbedder(), vector_store=fake_store)
    second = add(session, member_ctx, amount=10)
    assert expenses.reindex_missing(session, FailingEmbedder(), fake_store) == 0
    assert expenses.reindex_missing(session, fake_embedder, fake_store) == 2
    assert {key[1] for key in fake_store.items} == {str(first["id"]), str(second["id"])}
    assert expenses.reindex_missing(session, fake_embedder, fake_store) == 0


def test_edit_reindexes_and_delete_removes_vector(session, member_ctx, fake_embedder, fake_store):
    added = add(session, member_ctx, embedder=fake_embedder, vector_store=fake_store)
    expenses.update_expense(session, member_ctx, added["id"], {"date": date(2026, 11, 1)},
                            embedder=fake_embedder, vector_store=fake_store)
    assert fake_store.items[("expenses", str(added["id"]))]["metadata"]["date_key"] == 20261101
    expenses.delete_expense(session, member_ctx, added["id"], vector_store=fake_store)
    assert fake_store.items == {}


def test_vector_scope_filter(manager_ctx, member_ctx):
    assert scope_filter(manager_ctx) == {"family_id": manager_ctx.family_id}
    assert scope_filter(member_ctx) == {"$and": [{"family_id": member_ctx.family_id},
                                                 {"member_id": member_ctx.member_id}]}
