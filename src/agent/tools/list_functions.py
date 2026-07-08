"""list_functions tool: lists every indexed function/method/class in a
file, straight from Chroma's stored metadata -- no embedding or similarity
search involved, just an exact metadata filter."""

from agent.tools.base import BaseTool
from chroma_store import CODE_COLLECTION, ChromaStore


class ListFunctionsTool(BaseTool):
    name = "list_functions"
    description = (
        "Lists every function, method, and class the code-indexing pipeline "
        "found in a specific file, with their line ranges. Use this to get "
        "a table of contents for a file before deciding what to read_file. "
        "The path must match exactly what search_code returned."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the repo root, exactly as returned by search_code",
            },
        },
        "required": ["path"],
    }

    def __init__(self, store: ChromaStore):
        self._store = store

    def execute(self, path: str) -> list:
        matches = self._store.get_by_metadata(CODE_COLLECTION, where={"file": path})
        items = [
            {
                "name": m.metadata.get("name"),
                "kind": m.metadata.get("kind"),
                "language": m.metadata.get("language"),
                "start_line": m.metadata.get("start_line"),
                "end_line": m.metadata.get("end_line"),
            }
            for m in matches
        ]
        items.sort(key=lambda item: item["start_line"] or 0)
        return items
