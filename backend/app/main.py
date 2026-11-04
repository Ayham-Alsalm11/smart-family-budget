"""FastAPI application (§8).

At start-up the schema and the 12 categories are created, and Ollama is warmed up in the background; a failed
warm-up never stops the application. One Groq client serves the whole application, so that its rate limiter sees
every request.
"""
import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine, text

from app.api import actions, auth, budgets, chat, expenses, family, summary
from app.api.errors import install_error_handlers
from app.db.seed import seed_categories
from app.db.session import init_db, make_engine, make_session_factory
from app.llm.client import GroqClient
from app.rag.embeddings import OllamaEmbedder
from app.rag.vector_store import ChromaVectorStore
from app.settings import Settings, get_settings

log = logging.getLogger(__name__)


def _default_vector_store(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(settings.chroma_host, settings.chroma_port)


def _database_status(engine: Engine) -> dict:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def _chroma_status(factory: Callable[[Settings], Any], settings: Settings) -> dict:
    try:
        factory(settings).heartbeat()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def create_app(
    settings: Settings | None = None,
    embedder: OllamaEmbedder | None = None,
    vector_store_factory: Callable[[Settings], Any] = _default_vector_store,
    llm_client: Any = None,
) -> FastAPI:
    """`embedder`, `vector_store_factory` and `llm_client` can be injected by tests and scripts."""
    settings = settings or get_settings()
    embedder = embedder or OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)

    async def warm_up(app: FastAPI) -> None:
        app.state.ollama_warmup = await asyncio.to_thread(embedder.warmup)
        log.info("Ollama warm-up: %s", app.state.ollama_warmup)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = make_engine(settings.database_url)
        init_db(app.state.engine)
        app.state.session_factory = make_session_factory(app.state.engine)
        with app.state.session_factory() as session:
            seed_categories(session)
        own_client = None
        if llm_client is None and settings.groq_key_present:
            own_client = GroqClient.from_settings(settings)
        app.state.llm = llm_client or own_client
        app.state.ollama_warmup = {"ok": None, "status": "running"}
        task = asyncio.create_task(warm_up(app))
        yield
        task.cancel()
        if own_client is not None:
            own_client.close()
        app.state.engine.dispose()

    app = FastAPI(title="Family Budget Agent", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.vector_store = None
    app.state.vector_store_factory = vector_store_factory
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["*"],
                       allow_headers=["*"])
    install_error_handlers(app)
    for router in (auth.router, auth.me_router, chat.router, expenses.router, actions.router, summary.router,
                   budgets.router, family.router):
        app.include_router(router)

    @app.get("/health")
    def health() -> dict:
        database = _database_status(app.state.engine)
        chroma = _chroma_status(vector_store_factory, settings)
        ollama = {**embedder.status(), "warmup": app.state.ollama_warmup}
        healthy = database["ok"] and chroma["ok"] and ollama["ok"]
        return {
            "status": "ok" if healthy else "degraded",
            "database": database,
            "chroma": chroma,
            "ollama": ollama,
            "groq_api_key_present": settings.groq_key_present,
        }

    return app


app = create_app()
