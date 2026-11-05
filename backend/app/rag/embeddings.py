"""Ollama embeddings client (§1, A3 in §3.2): BGE-M3, 1024 dimensions.

The timeout is 120 s with 2 extra attempts, because the first request after a cold start
takes about 30 s (A3). `warmup()` sends that first request at start-up.
"""
import logging
import time

import httpx

log = logging.getLogger(__name__)


class EmbeddingError(Exception):
    pass


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        retries: int = 2,
        retry_delay: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.retries = retries
        self.retry_delay = retry_delay
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def embed(self, text: str) -> list[float]:
        last_error: Exception | None = None
        for attempt in range(1 + self.retries):
            try:
                response = self._client.post("/api/embed", json={"model": self.model, "input": text})
                response.raise_for_status()
                vectors = response.json().get("embeddings") or []
                if not vectors or not vectors[0]:
                    raise EmbeddingError("Ollama returned no embedding")
                return vectors[0]
            except (httpx.HTTPError, ValueError, EmbeddingError) as exc:
                last_error = exc
                log.warning("embedding attempt %d failed: %s", attempt + 1, exc)
                if attempt < self.retries:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise EmbeddingError(f"embedding failed after {1 + self.retries} attempts: {last_error}") from last_error

    def warmup(self) -> dict:
        """One embedding request so that the model is loaded. Never raises."""
        started = time.perf_counter()
        try:
            dimensions = len(self.embed("warm-up"))
        except EmbeddingError as exc:
            return {"ok": False, "seconds": round(time.perf_counter() - started, 1), "error": str(exc)}
        return {"ok": True, "seconds": round(time.perf_counter() - started, 1), "dimensions": dimensions}

    def status(self) -> dict:
        """Is Ollama reachable now, and is the model pulled? Checked on every call (A4)."""
        try:
            response = self._client.get("/api/tags", timeout=3.0)
            response.raise_for_status()
            names = {m.get("name") for m in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        available = self.model in names or f"{self.model}:latest" in names
        return {"ok": available, "model": self.model, "model_available": available}
