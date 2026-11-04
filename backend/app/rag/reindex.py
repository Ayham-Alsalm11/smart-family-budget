"""Bring ChromaDB in line with the development database.

    cd backend
    .venv\\Scripts\\python -m app.rag.reindex          # embed the expenses that have no vector yet
    .venv\\Scripts\\python -m app.rag.reindex --reset  # wipe the server's collections and embed every expense again

Run it with --reset after every re-creation of backend/data/fba.db: the ChromaDB server keeps its data apart from
the database, so old vectors would stay under ids that now mean other expenses. Needs ChromaDB and Ollama running.
"""
import argparse
import sys

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.models import Expense
from app.db.session import init_db, make_engine, make_session_factory
from app.rag.embeddings import OllamaEmbedder
from app.rag.vector_store import ChromaVectorStore
from app.services.expenses import reindex_missing
from app.settings import get_settings


def reindex(session: Session, embedder, store: ChromaVectorStore, reset: bool = False) -> dict:
    """Returns how many vectors were removed, how many expenses exist and how many are stored now."""
    removed = 0
    if reset:
        removed = store.reset_collections()
        session.execute(update(Expense).values(embedded_at=None))
        session.commit()
    store.ensure_collections()
    stored = reindex_missing(session, embedder, store)
    expenses = session.scalar(select(func.count()).select_from(Expense))
    missing = session.scalar(select(func.count()).select_from(Expense).where(Expense.embedded_at.is_(None)))
    return {"removed": removed, "expenses": expenses, "stored": stored, "missing": missing}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.rag.reindex", description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="wipe the ChromaDB collections first")
    args = parser.parse_args(argv)

    settings = get_settings()
    store = ChromaVectorStore(settings.chroma_host, settings.chroma_port)
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    warmup = embedder.warmup()
    if not warmup["ok"]:
        print(f"Stopped before changing anything: Ollama is not ready ({warmup['error']}).", file=sys.stderr)
        return 1

    engine = make_engine(settings.database_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        result = reindex(session, embedder, store, reset=args.reset)
    engine.dispose()
    print(f"Vectors removed: {result['removed']}. Expenses: {result['expenses']}. Embedded now: {result['stored']}. "
          f"Still missing: {result['missing']}.")
    return 1 if result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
