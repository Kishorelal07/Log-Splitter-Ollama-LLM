"""The agent's tool-calling execution loop: send the conversation + tool
definitions to the model, execute whatever tools it asks for, feed the
results back, and repeat until it returns a final answer or the iteration
cap is hit.

Also enforces tool-call discipline that a system prompt alone can't
guarantee against a small local model: exact and near-duplicate calls are
intercepted before they reach Chroma/disk, and failed or empty results get
an explicit hint steering the model back toward answering from general
knowledge instead of chaining more tools."""

import difflib
import json
import logging

from agent.errors import MaxIterationsExceededError
from agent.llm_client import call_llm
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Two tool calls on the same tool are treated as a repeat if their query
# text is at least this similar (difflib ratio, 0-1). Tuned to catch
# rephrasings of the same search ("pan verification failing" vs "why does
# pan verification fail") without conflating genuinely different queries.
_NEAR_DUPLICATE_THRESHOLD = 0.88

SYSTEM_PROMPT = (
    "You are an assistant with tools for searching THIS project's production "
    "logs and THIS project's codebase. Before doing anything else, classify "
    "the user's message into one of two categories.\n\n"
    "1. GENERAL KNOWLEDGE -- programming concepts, language or framework "
    "features, HTTP status codes, standard exceptions, or general software "
    "engineering questions you already know the answer to. Examples: "
    "'what is HTTP 500', 'what is Spring Boot', 'explain NullPointerException', "
    "'difference between GET and POST', 'what is a vector database'. "
    "For these, DO NOT call any tools -- answer directly. Calling a tool for "
    "something you can already answer wastes time and produces irrelevant "
    "results.\n\n"
    "2. PROJECT-SPECIFIC -- questions that require information only found in "
    "THIS project's logs or THIS project's source code. Examples: 'why is "
    "PAN verification failing', 'show today's failed logs', 'where is "
    "addDataToGrid implemented', 'why is our API returning 500 today', 'find "
    "the code causing this stack trace'. For these, use the appropriate "
    "tool(s): search_logs for runtime/failure questions, search_code then "
    "read_file for locating and viewing real code, list_functions for a "
    "file's table of contents, search_both only when it's genuinely "
    "ambiguous between logs and code.\n\n"
    "Rules for using tools:\n"
    "- Never call a tool just because it exists -- only call one when the "
    "question genuinely requires project-specific data you don't already "
    "have.\n"
    "- Reason about what you already know before retrieving anything.\n"
    "- Use the minimum number of tool calls needed. Do not repeat a search "
    "with a near-identical query, and do not call the same tool again "
    "without a concrete new reason -- repeating an unchanged query will not "
    "produce new information.\n"
    "- Stop as soon as you have enough information to answer. More tool "
    "calls are not automatically better.\n"
    "- If a tool call fails or returns nothing relevant, do not blindly try "
    "another tool. First ask: can this be answered from general knowledge "
    "instead? If yes, answer directly. Only try a different tool if there "
    "is a concrete reason to believe it will find something the failed one "
    "couldn't.\n"
    "- Always cite the file path and line numbers you actually read when "
    "answering about code.\n"
    "- Never invent code or log content that wasn't returned by a tool -- "
    "if the tools found nothing relevant and you also don't know the answer "
    "generally, say so plainly instead of guessing."
)


def _call_signature(name: str, arguments: dict) -> str:
    """Canonical string for exact-repeat detection, shaped per tool since
    each one's meaningful arguments differ."""
    if name in ("search_logs", "search_code", "search_both"):
        query = str(arguments.get("query", "")).strip().lower()
        extra = f"|{arguments.get('status', '')}|{arguments.get('level', '')}" if name == "search_logs" else ""
        return f"{name}:{query}{extra}"
    if name == "read_file":
        return f"{name}:{arguments.get('path', '')}:{arguments.get('start_line')}:{arguments.get('end_line')}"
    if name == "list_functions":
        return f"{name}:{arguments.get('path', '')}"
    return f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"


def _is_near_duplicate_query(tool_name: str, query: str, seen_queries_by_tool: dict) -> bool:
    for prior in seen_queries_by_tool.get(tool_name, []):
        if difflib.SequenceMatcher(None, prior, query).ratio() >= _NEAR_DUPLICATE_THRESHOLD:
            return True
    return False


def _is_empty_result(data) -> bool:
    if data is None:
        return True
    if isinstance(data, list):
        return len(data) == 0
    if isinstance(data, dict):
        return all(_is_empty_result(v) for v in data.values())
    return False


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

        # Per-turn only -- a later, genuinely new question shouldn't be
        # penalized for wording similar to an earlier one.
        seen_signatures = set()
        seen_queries_by_tool = {}

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
                signature = _call_signature(call.name, call.arguments)
                query_text = str(call.arguments.get("query", "")).strip().lower()
                is_repeat = signature in seen_signatures or (
                    bool(query_text) and _is_near_duplicate_query(call.name, query_text, seen_queries_by_tool)
                )

                if is_repeat:
                    logger.info("Blocked repeat/near-duplicate tool call: %s(%s)", call.name, call.arguments)
                    result_payload = {
                        "success": False,
                        "error": (
                            "This tool was already called this turn with the same or a very "
                            "similar query -- repeating it will not produce new information. "
                            "Use the results you already have to answer, try a genuinely "
                            "different tool or query only if you have a concrete reason to, "
                            "or answer from general knowledge if that's sufficient."
                        ),
                    }
                else:
                    logger.info("Tool call: %s(%s)", call.name, call.arguments)
                    tool = self._registry.get(call.name)
                    if tool is None:
                        result_payload = {"success": False, "error": f"Unknown tool '{call.name}'"}
                    else:
                        result = tool.safe_execute(**call.arguments)
                        result_payload = result.to_json_dict()
                        if not result.success:
                            result_payload["hint"] = (
                                "This tool call failed. If the question can still be answered "
                                "from general knowledge, answer directly instead of trying "
                                "another tool."
                            )
                        elif _is_empty_result(result.data):
                            result_payload["hint"] = (
                                "No relevant results were found. If the question can be "
                                "answered from general knowledge, answer directly instead of "
                                "calling more tools."
                            )

                    seen_signatures.add(signature)
                    if query_text:
                        seen_queries_by_tool.setdefault(call.name, []).append(query_text)

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
