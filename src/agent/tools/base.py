"""Base class every tool implements, plus the JSON-serializable result type
returned by tool execution."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None

    def to_json_dict(self) -> dict:
        """What actually gets serialized back to the LLM as the tool's
        response message. Only JSON-serializable primitives/dicts/lists
        should ever end up in `data`."""
        if self.success:
            return {"success": True, "data": self.data}
        return {"success": False, "error": self.error}


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema for the tool's arguments

    def to_ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Runs the tool and returns JSON-serializable data. Raise a normal
        exception on failure -- safe_execute() converts it into a ToolResult
        so one bad tool call can't crash the whole agent loop."""

    def safe_execute(self, **kwargs) -> ToolResult:
        try:
            data = self.execute(**kwargs)
            return ToolResult(tool_name=self.name, success=True, data=data)
        except Exception as exc:
            logger.exception("Tool '%s' failed with args %s", self.name, kwargs)
            return ToolResult(tool_name=self.name, success=False, error=str(exc))
