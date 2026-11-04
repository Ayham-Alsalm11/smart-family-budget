from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from app.main import create_app
from app.settings import Settings

SECRET = "gsk_test_value_never_shown"


class StubEmbedder:
    def warmup(self) -> dict:
        return {"ok": True, "seconds": 0.0, "dimensions": 1024}

    def status(self) -> dict:
        return {"ok": True, "model": "bge-m3", "model_available": True}


def unreachable(_settings):
    raise ConnectionError("chroma is down")


def make_client(tmp_path, groq_api_key):
    url = f"sqlite:///{(tmp_path / 'health.db').as_posix()}"
    settings = Settings(_env_file=None, database_url=url, groq_api_key=groq_api_key)
    return TestClient(create_app(settings, embedder=StubEmbedder(), vector_store_factory=unreachable)), url


def test_health_reports_each_service_and_never_the_key(tmp_path):
    client, url = make_client(tmp_path, SECRET)
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["database"]["ok"] is True
    assert body["chroma"]["ok"] is False
    assert body["ollama"]["ok"] is True
    assert body["groq_api_key_present"] is True
    assert body["status"] == "degraded"
    assert SECRET not in response.text
    assert len(inspect(create_engine(url)).get_table_names()) == 12, "create_all runs at start-up"


def test_health_without_key(tmp_path):
    client, _ = make_client(tmp_path, None)
    with client:
        assert client.get("/health").json()["groq_api_key_present"] is False
