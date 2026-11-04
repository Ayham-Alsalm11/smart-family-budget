"""ChromaDB in server mode (HttpClient), collections `expenses` and `corrections` with cosine distance.

Metadata keeps real types (family_id and member_id are integers). Every query is scoped by
`scope_filter(ctx)`: the family, plus the member for members (§5, P8).
"""
from typing import Any

import chromadb

from app.agent.context import ToolContext

COLLECTIONS = ("expenses", "corrections")
COSINE = {"hnsw:space": "cosine"}


class VectorStoreError(Exception):
    pass


def scope_filter(ctx: ToolContext) -> dict[str, Any]:
    if ctx.is_manager:
        return {"family_id": ctx.family_id}
    return {"$and": [{"family_id": ctx.family_id}, {"member_id": ctx.member_id}]}


def collection_space(collection: Any) -> str:
    """The distance function of a collection, from its legacy metadata or its configuration."""
    space = (collection.metadata or {}).get("hnsw:space")
    if space:
        return space
    configuration = getattr(collection, "configuration", None) or {}
    hnsw = configuration.get("hnsw") or {}
    return hnsw.get("space") or "l2"


class ChromaVectorStore:
    def __init__(self, host: str, port: int, client: Any | None = None) -> None:
        self._client = client if client is not None else chromadb.HttpClient(host=host, port=port)

    def heartbeat(self) -> int:
        return self._client.heartbeat()

    def collection(self, name: str) -> Any:
        if name not in COLLECTIONS:
            raise ValueError(f"unknown collection {name!r}")
        return self._client.get_or_create_collection(name=name, metadata=COSINE)

    def ensure_collections(self) -> dict[str, str]:
        """Create both collections if needed and check that they use cosine distance."""
        spaces = {name: collection_space(self.collection(name)) for name in COLLECTIONS}
        wrong = {name: space for name, space in spaces.items() if space != "cosine"}
        if wrong:
            raise VectorStoreError(f"collections without cosine distance: {wrong}")
        return spaces

    def upsert(self, collection: str, item_id: str, embedding: list[float], document: str,
               metadata: dict[str, Any]) -> None:
        self.collection(collection).upsert(
            ids=[item_id], embeddings=[embedding], documents=[document], metadatas=[metadata])

    def delete(self, collection: str, item_id: str) -> None:
        self.collection(collection).delete(ids=[item_id])

    def query(self, collection: str, ctx: ToolContext, embedding: list[float], n_results: int = 5,
              where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        scope = scope_filter(ctx)
        if where:
            conditions = scope["$and"] if "$and" in scope else [scope]
            scope = {"$and": [*conditions, where]}
        result = self.collection(collection).query(query_embeddings=[embedding], n_results=n_results, where=scope)
        return [
            {"id": item_id, "document": document, "metadata": metadata, "distance": distance}
            for item_id, document, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]
