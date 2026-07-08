"""Thin wrapper around ChromaDB collection access, shared by the legacy
query CLI and the agent's tools. Centralizing this means every caller gets
the same error handling instead of each reimplementing get_collection +
try/except around a missing collection."""

from dataclasses import dataclass

import chromadb

LOGS_COLLECTION = "prod_logs"
CODE_COLLECTION = "my_repo"


class CollectionNotFoundError(Exception):
    """Raised when a Chroma collection hasn't been created yet (ingestion
    for that side of the pipeline has never been run)."""


@dataclass(frozen=True)
class Match:
    id: str
    metadata: dict
    distance: float


class ChromaStore:
    """Opens one persistent Chroma client and hands out query helpers for
    the two collections this project uses."""

    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)

    def _get_collection(self, name: str):
        try:
            return self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(
                f"Collection '{name}' not found in Chroma -- has ingestion been run yet?"
            ) from exc

    def query(
        self,
        collection_name: str,
        embedding: list,
        top_k: int,
        where: dict | None = None,
    ) -> list:
        """Nearest-neighbor similarity search."""
        collection = self._get_collection(collection_name)
        result = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where or None,
            include=["metadatas", "distances"],
        )
        return [
            Match(id=id_, metadata=meta, distance=dist)
            for id_, meta, dist in zip(result["ids"][0], result["metadatas"][0], result["distances"][0])
        ]

    def get_by_metadata(self, collection_name: str, where: dict) -> list:
        """Exact metadata filter, no embedding or similarity search involved
        -- used when the caller already knows exactly what it wants (e.g.
        every chunk belonging to one specific file)."""
        collection = self._get_collection(collection_name)
        result = collection.get(where=where, include=["metadatas"])
        return [Match(id=id_, metadata=meta, distance=0.0) for id_, meta in zip(result["ids"], result["metadatas"])]
