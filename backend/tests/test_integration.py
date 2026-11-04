"""Tests against the real local services. Not run by default: `pytest -m integration`."""
import pytest

from app.rag.embeddings import OllamaEmbedder
from app.rag.vector_store import ChromaVectorStore
from app.settings import get_settings

pytestmark = pytest.mark.integration


def test_ollama_bge_m3_has_1024_dimensions():
    settings = get_settings()
    vector = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model).embed("دفعت 5000 بنزين")
    assert len(vector) == 1024


def test_chroma_server_collections_use_cosine():
    settings = get_settings()
    store = ChromaVectorStore(settings.chroma_host, settings.chroma_port)
    assert store.heartbeat() > 0
    assert store.ensure_collections() == {"expenses": "cosine", "corrections": "cosine"}
