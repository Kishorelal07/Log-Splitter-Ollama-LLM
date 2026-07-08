"""search_logs tool: semantic search over the prod_logs Chroma collection,
with optional status/level metadata filters."""

from dataclasses import dataclass

from agent.tools.base import BaseTool
from chroma_store import LOGS_COLLECTION, ChromaStore
from ollama_utils import embed_query


@dataclass(frozen=True)
class LogFilters:
    status: str | None = None
    level: str | None = None

    def as_where(self) -> dict:
        where = {}
        if self.status:
            where["status"] = self.status
        if self.level:
            where["level"] = self.level
        return where


class SearchLogsTool(BaseTool):
    name = "search_logs"
    description = (
        "Searches THIS project's indexed production log data. Use ONLY when "
        "the question is about actual runtime behavior of this system: "
        "production failures, error patterns, stack traces that occurred, "
        "debugging a real incident, or project-specific application errors "
        "(e.g. 'why is PAN verification failing', 'show today's failed "
        "logs'). Do NOT use for definitions, general programming concepts, "
        "HTTP status code explanations, language or framework concepts, or "
        "anything answerable from general knowledge. Supports optional "
        "filters by outcome status or log level."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language description of what to search for"},
            "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
            "status": {
                "type": "string",
                "enum": ["fail", "success", "other"],
                "description": "Optional: only return logs with this outcome",
            },
            "level": {"type": "string", "description": "Optional: only return logs at this level, e.g. ERROR, INFO"},
        },
        "required": ["query"],
    }

    def __init__(self, store: ChromaStore, embed_model: str, ollama_host: str, max_distance: float | None):
        self._store = store
        self._embed_model = embed_model
        self._ollama_host = ollama_host
        self._max_distance = max_distance

    def execute(self, query: str, top_k: int = 5, status: str | None = None, level: str | None = None) -> list:
        filters = LogFilters(status=status, level=level)
        embedding = embed_query(query, self._embed_model, self._ollama_host)
        matches = self._store.query(LOGS_COLLECTION, embedding, top_k, where=filters.as_where() or None)

        if self._max_distance is not None:
            matches = [m for m in matches if m.distance <= self._max_distance]

        return [
            {
                "id": m.id,
                "distance": round(m.distance, 4),
                "level": m.metadata.get("level"),
                "status": m.metadata.get("status"),
                "component": m.metadata.get("component"),
                "count": m.metadata.get("count"),
                "first_seen": m.metadata.get("first_seen"),
                "last_seen": m.metadata.get("last_seen"),
                "message": m.metadata.get("raw"),
            }
            for m in matches
        ]
