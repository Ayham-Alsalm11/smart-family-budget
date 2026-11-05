"""`python -m app.rag.reindex` on an in-process ChromaDB (§4)."""
import chromadb
import pytest

from app.db.models import Expense
from app.rag import reindex as module
from app.rag.reindex import reindex
from app.rag.vector_store import COLLECTIONS, ChromaVectorStore
from app.services.expenses import add_expense

from conftest import FailingEmbedder


@pytest.fixture
def store() -> ChromaVectorStore:
    """The real store on in-process ChromaDB. Those clients share their data, so each test starts by clearing it."""
    client = chromadb.EphemeralClient()
    for collection in client.list_collections():
        client.delete_collection(collection.name)
    return ChromaVectorStore("", 0, client=client)


def count(store: ChromaVectorStore, collection: str = "expenses") -> int:
    return store.collection(collection).count()


def stale_vector(store: ChromaVectorStore, item_id: str, family_id: int = 1) -> None:
    store.upsert("expenses", item_id, [0.5] * 8, "مصروف من قاعدة قديمة", {"expense_id": int(item_id), "family_id": family_id})


def test_reset_removes_old_vectors_and_embeds_every_expense(session, manager_ctx, fake_embedder, store):
    first = add_expense(session, manager_ctx, amount=1000, category="صحة", description="دوا")
    second = add_expense(session, manager_ctx, amount=2000, category="صحة", description="كشفية")
    store.ensure_collections()
    stale_vector(store, "999")
    stale_vector(store, str(first["id"]), family_id=7)

    result = reindex(session, fake_embedder, store, reset=True)

    assert result == {"removed": 2, "expenses": 2, "stored": 2, "missing": 0}
    data = store.collection("expenses").get(include=["metadatas", "documents"])
    assert sorted(data["ids"]) == sorted([str(first["id"]), str(second["id"])])
    assert {m["family_id"] for m in data["metadatas"]} == {manager_ctx.family_id}, "no vector of the old family stayed"
    assert all(row.embedded_at is not None for row in session.query(Expense))


def test_reset_clears_both_collections_and_keeps_cosine(session, fake_embedder, store):
    store.ensure_collections()
    store.upsert("corrections", "1", [0.5] * 8, "تصحيح قديم", {"family_id": 1})
    assert reindex(session, fake_embedder, store, reset=True)["removed"] == 1
    assert count(store, "corrections") == 0
    assert store.ensure_collections() == {name: "cosine" for name in COLLECTIONS}


def test_reset_works_on_a_server_without_collections(session, fake_embedder, store):
    assert reindex(session, fake_embedder, store, reset=True)["removed"] == 0


def test_without_reset_only_missing_expenses_are_embedded(session, manager_ctx, fake_embedder, store):
    add_expense(session, manager_ctx, amount=1000, category="صحة", embedder=fake_embedder, vector_store=store)
    add_expense(session, manager_ctx, amount=2000, category="صحة")
    stale_vector(store, "999")
    fake_embedder.calls.clear()

    result = reindex(session, fake_embedder, store)

    assert result == {"removed": 0, "expenses": 2, "stored": 1, "missing": 0}
    assert len(fake_embedder.calls) == 1 and count(store) == 3, "other vectors are left alone"


def test_failure_is_reported_as_missing(session, manager_ctx, store):
    add_expense(session, manager_ctx, amount=1000, category="صحة")
    assert reindex(session, FailingEmbedder(), store, reset=True) == {
        "removed": 0, "expenses": 1, "stored": 0, "missing": 1}


class DownEmbedder:
    def __init__(self, *_args) -> None:
        pass

    def warmup(self) -> dict:
        return {"ok": False, "seconds": 0.0, "error": "Ollama is down"}


class UntouchableStore:
    def __init__(self, *_args) -> None:
        pass

    def reset_collections(self) -> int:
        raise AssertionError("nothing may be wiped when Ollama is down")


def test_nothing_is_wiped_when_ollama_is_down(monkeypatch, capsys):
    monkeypatch.setattr(module, "OllamaEmbedder", DownEmbedder)
    monkeypatch.setattr(module, "ChromaVectorStore", UntouchableStore)
    assert module.main(["--reset"]) == 1
    assert "Ollama" in capsys.readouterr().err
