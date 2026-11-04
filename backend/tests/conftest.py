"""Shared fixtures: a temporary SQLite database per test, a family with members, and fake RAG services."""
import hashlib
from datetime import date
from decimal import Decimal

import pytest

from app.agent.context import ToolContext
from app.db.models import Family, Member
from app.db.seed import seed_categories
from app.db.session import init_db, make_engine, make_session_factory
from app.rag.embeddings import EmbeddingError

TODAY = date(2026, 11, 4)  # a Wednesday


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(engine)
    with make_session_factory(engine)() as db:
        seed_categories(db)
        yield db
    engine.dispose()


def make_family(session, base_currency="SYP_OLD", income="500000", usd_rate="110") -> Family:
    family = Family(name="عائلة اختبار", monthly_income=Decimal(income), base_currency=base_currency,
                    usd_rate=Decimal(usd_rate) if usd_rate is not None else None)
    session.add(family)
    session.commit()
    return family


def make_member(session, family, username, role) -> Member:
    member = Member(family_id=family.id, name=username, username=username, password_hash="x", role=role)
    session.add(member)
    session.commit()
    return member


def ctx_for(member: Member, today: date = TODAY) -> ToolContext:
    return ToolContext(family_id=member.family_id, member_id=member.id, role=member.role, today=today)


@pytest.fixture
def family(session) -> Family:
    return make_family(session)


@pytest.fixture
def manager(session, family) -> Member:
    return make_member(session, family, "manager1", "manager")


@pytest.fixture
def member(session, family) -> Member:
    return make_member(session, family, "member1", "member")


@pytest.fixture
def other_member(session, family) -> Member:
    return make_member(session, family, "member2", "member")


@pytest.fixture
def manager_ctx(manager) -> ToolContext:
    return ctx_for(manager)


@pytest.fixture
def member_ctx(member) -> ToolContext:
    return ctx_for(member)


@pytest.fixture
def other_member_ctx(other_member) -> ToolContext:
    return ctx_for(other_member)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255 for b in digest[:8]]


class FailingEmbedder:
    def embed(self, text: str) -> list[float]:
        raise EmbeddingError("Ollama is down")


class FakeVectorStore:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict] = {}
        self.queries: list[dict] = []

    def upsert(self, collection, item_id, embedding, document, metadata) -> None:
        self.items[(collection, item_id)] = {"embedding": embedding, "document": document, "metadata": metadata}

    def delete(self, collection, item_id) -> None:
        self.items.pop((collection, item_id), None)

    def query(self, collection, ctx, embedding, n_results=5, where=None) -> list[dict]:
        """Returns every stored item, ignoring the scope, so tests can check that the tools still filter."""
        self.queries.append({"collection": collection, "ctx": ctx, "where": where})
        return [{"id": key[1], "document": item["document"], "metadata": item["metadata"], "distance": 0.1}
                for key, item in self.items.items() if key[0] == collection][:n_results]

    def heartbeat(self) -> int:
        return 1


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()
