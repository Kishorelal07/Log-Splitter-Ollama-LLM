"""The agent's tool-calling execution loop: send the conversation + tool
definitions to the model, execute whatever tools it asks for, feed the
results back, and repeat until it returns a final answer or the iteration
cap is hit."""

import json
import logging

from agent.errors import MaxIterationsExceededError
from agent.llm_client import call_llm
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an assistant with tools for searching production logs and a "
    "codebase. Decide which tools (if any) are relevant to the user's "
    "message -- for a greeting or general question, use no tools and just "
    "reply. For questions about errors or behavior, use search_logs. For "
    "questions about where code lives or how it works, use search_code, "
    "then read_file to see the actual code before answering. Use "
    "list_functions to see what's in a file before reading all of it. Use "
    "search_both only when the question is genuinely ambiguous between logs "
    "and code. Always cite the file path and line numbers you actually read "
    "when answering about code. Never invent code or log content that "
    "wasn't returned by a tool -- if the tools found nothing relevant, say "
    "so plainly instead of guessing."
)


class AgentLoop:
    def __init__(self, registry: ToolRegistry, chat_model: str, ollama_host: str, max_iterations: int = 6):
        self._registry = registry
        self._chat_model = chat_model
        self._ollama_host = ollama_host
        self._max_iterations = max_iterations

    def run(self, user_message: str, history_messages: list):
        """Runs one full turn, including any number of tool calls. Returns
        (final_answer_text, transcript) where transcript is every message
        exchanged this turn (system prompt included) -- useful for logging
        or debugging, not persisted into ConversationHistory."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_message})

        tool_schemas = self._registry.schemas()

        for iteration in range(1, self._max_iterations + 1):
            logger.info("Agent iteration %d/%d", iteration, self._max_iterations)
            response = call_llm(messages, tool_schemas, self._chat_model, self._ollama_host)

            if response.is_final_answer:
                messages.append({"role": "assistant", "content": response.content})
                return response.content, messages

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {"function": {"name": c.name, "arguments": c.arguments}} for c in response.tool_calls
                    ],
                }
            )

            for call in response.tool_calls:
                logger.info("Tool call: %s(%s)", call.name, call.arguments)
                tool = self._registry.get(call.name)
                if tool is None:
                    result_payload = {"success": False, "error": f"Unknown tool '{call.name}'"}
                else:
                    result_payload = tool.safe_execute(**call.arguments).to_json_dict()

                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": json.dumps(result_payload, default=str),
                    }
                )

        raise MaxIterationsExceededError(
            f"Agent did not produce a final answer within {self._max_iterations} tool-call rounds"
        )
