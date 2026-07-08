"""search_code tool: semantic search over the my_repo Chroma collection.
Returns file/line/name citations and a similarity score only -- never code
text. If the model needs to actually see the code, it must follow up with
a read_file call."""

from agent.tools.base import BaseTool
from chroma_store import CODE_COLLECTION, ChromaStore
from ollama_utils import embed_query


class SearchCodeTool(BaseTool):
    name = "search_code"
    description = (
        "Semantic search over indexed functions, methods, and classes in "
        "the codebase. Returns WHERE relevant code lives (file path, line "
        "range, qualified name) with a similarity score -- it does not "
        "return the code's contents. Call read_file afterwards to see the "
        "actual code at a returned location."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language description of the code you're looking for"},
            "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, store: ChromaStore, embed_model: str, ollama_host: str, max_distance: float | None):
        self._store = store
        self._embed_model = embed_model
        self._ollama_host = ollama_host
        self._max_distance = max_distance

    def execute(self, query: str, top_k: int = 5) -> list:
        embedding = embed_query(query, self._embed_model, self._ollama_host)
        matches = self._store.query(CODE_COLLECTION, embedding, top_k)

        if self._max_distance is not None:
            matches = [m for m in matches if m.distance <= self._max_distance]

        return [
            {
                "distance": round(m.distance, 4),
                "name": m.metadata.get("name"),
                "kind": m.metadata.get("kind"),
                "language": m.metadata.get("language"),
                "file": m.metadata.get("file"),
                "start_line": m.metadata.get("start_line"),
                "end_line": m.metadata.get("end_line"),
            }
            for m in matches
        ]
