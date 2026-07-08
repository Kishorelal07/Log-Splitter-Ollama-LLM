"""Wraps Ollama's tool-calling chat API, with a fallback parser for cases
where Ollama doesn't populate message.tool_calls but the model still emits
its native <tool_call>{...}</tool_call> convention inline in message.content
(this is Qwen's own chat-template convention for tool calls, so it's a
reasonable thing to look for rather than an arbitrary guess)."""

import json
import logging
import re
from dataclasses import dataclass

from ollama_utils import chat_with_tools

logger = logging.getLogger(__name__)

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class AgentResponse:
    content: str
    tool_calls: list

    @property
    def is_final_answer(self) -> bool:
        return not self.tool_calls


def _parse_native_tool_calls(message: dict) -> list:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Could not parse tool call arguments as JSON: %r", arguments)
                arguments = {}
        calls.append(ParsedToolCall(name=function.get("name", ""), arguments=arguments))
    return calls


def _parse_fallback_tool_calls(content: str) -> list:
    calls = []
    for raw in _TOOL_CALL_TAG_RE.findall(content):
        try:
            parsed = json.loads(raw)
            calls.append(ParsedToolCall(name=parsed.get("name", ""), arguments=parsed.get("arguments", {})))
        except json.JSONDecodeError:
            logger.warning("Found a <tool_call> tag but couldn't parse its JSON: %r", raw)
    return calls


def call_llm(messages: list, tool_schemas: list, model: str, host: str) -> AgentResponse:
    message = chat_with_tools(messages, tool_schemas, model, host)
    content = message.get("content", "") or ""

    tool_calls = _parse_native_tool_calls(message)
    if not tool_calls:
        tool_calls = _parse_fallback_tool_calls(content)

    return AgentResponse(content=content, tool_calls=tool_calls)
