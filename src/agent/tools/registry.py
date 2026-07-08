"""Holds every registered tool and hands out both the Ollama-facing schema
list and lookup-by-name for the agent loop. Adding a new tool anywhere else
in the codebase never requires touching this file -- just pass an extra
BaseTool instance into ToolRegistry(...) at construction time."""

from agent.tools.base import BaseTool


class ToolRegistry:
    def __init__(self, tools: list):
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list:
        return [tool.to_ollama_schema() for tool in self._tools.values()]

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list:
        return list(self._tools.keys())
