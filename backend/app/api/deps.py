"""Shared pieces of the routers: id types, and the search services for the current request.

Route functions have no return annotation on purpose: with `-> dict`, FastAPI would send Decimal money values as
strings instead of numbers.
"""
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Path, Request

MAX_ID = 2 ** 63 - 1  # SQLite's INTEGER; a larger number would fail in the driver instead of returning 404
ResourceId = Annotated[int, Path(ge=1, le=MAX_ID)]


@dataclass(frozen=True)
class Rag:
    embedder: Any = None
    vector_store: Any = None


def get_rag(request: Request) -> Rag:
    """Checked on every call (A4). When Ollama or ChromaDB is down the expense is still recorded, and
    `search_expenses` answers not_available."""
    state = request.app.state
    if not state.embedder.status()["ok"]:
        return Rag()
    if state.vector_store is None:
        try:
            state.vector_store = state.vector_store_factory(state.settings)
        except Exception:
            return Rag()
    return Rag(state.embedder, state.vector_store)


SearchServices = Annotated[Rag, Depends(get_rag)]
