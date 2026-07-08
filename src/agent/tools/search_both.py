"""search_both tool: convenience wrapper that runs search_logs and
search_code with the same query and merges the results, tagged by source.
Composes the other two tool instances directly rather than reimplementing
their logic."""

from agent.tools.base import BaseTool
from agent.tools.search_code import SearchCodeTool
from agent.tools.search_logs import SearchLogsTool


class SearchBothTool(BaseTool):
    name = "search_both"
    description = (
        "Searches logs and code together in one call. Use ONLY when the "
        "question is already established to be project-specific AND it's "
        "genuinely ambiguous whether the answer lives in logs or in code "
        "(e.g. 'what's going wrong with PAN verification' could mean the "
        "failing logs or the verification code itself). Prefer the more "
        "specific search_logs or search_code whenever the intent is clear, "
        "and never use this for general-knowledge questions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "top_k_logs": {"type": "integer", "description": "Number of log results", "default": 5},
            "top_k_code": {"type": "integer", "description": "Number of code results", "default": 3},
        },
        "required": ["query"],
    }

    def __init__(self, search_logs_tool: SearchLogsTool, search_code_tool: SearchCodeTool):
        self._search_logs = search_logs_tool
        self._search_code = search_code_tool

    def execute(self, query: str, top_k_logs: int = 5, top_k_code: int = 3) -> dict:
        return {
            "logs": self._search_logs.execute(query=query, top_k=top_k_logs),
            "code": self._search_code.execute(query=query, top_k=top_k_code),
        }
