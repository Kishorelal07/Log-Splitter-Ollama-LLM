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
        "Searches logs and code together in one call. Use this only when "
        "the question could plausibly be answered by either -- for example "
        "'what's going wrong with PAN verification' could mean 'show me the "
        "failing logs' or 'show me the verification code'. Prefer the more "
        "specific search_logs or search_code when the intent is clear."
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
