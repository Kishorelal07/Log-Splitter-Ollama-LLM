"""Agent CLI entrypoint: a tool-calling chat agent over the log/code Chroma
collections. Unlike query.py (which always searches both collections up
front for every question), this lets the model decide which tools -- if
any -- a given question actually needs, and lets it chain multiple tool
calls in one turn (e.g. search_code -> read_file).

Usage:
    One-shot:
        python src/agent_cli.py "why did PAN verification fail?"
        python src/agent_cli.py "where is Aadhaar validation implemented?" --repo-path "C:\\path\\to\\project"

    Chat mode (omit the question -> interactive loop):
        python src/agent_cli.py --repo-path "C:\\path\\to\\project"
"""

import argparse
import logging
import sys

from agent.config import AgentConfig
from agent.errors import AgentError
from agent.history import ConversationHistory
from agent.loop import AgentLoop
from agent.tools.list_functions import ListFunctionsTool
from agent.tools.read_file import ReadFileTool
from agent.tools.registry import ToolRegistry
from agent.tools.search_both import SearchBothTool
from agent.tools.search_code import SearchCodeTool
from agent.tools.search_logs import SearchLogsTool
from chroma_store import ChromaStore
from ollama_utils import DEFAULT_OLLAMA_HOST, check_ollama


def build_registry(config: AgentConfig, store: ChromaStore) -> ToolRegistry:
    search_logs = SearchLogsTool(store, config.embed_model, config.ollama_host, config.max_distance)
    search_code = SearchCodeTool(store, config.embed_model, config.ollama_host, config.max_distance)
    tools = [
        search_logs,
        search_code,
        SearchBothTool(search_logs, search_code),
        ReadFileTool(config.repo_path),
        ListFunctionsTool(store),
    ]
    return ToolRegistry(tools)


def run_turn(loop: AgentLoop, history: ConversationHistory, question: str) -> None:
    try:
        answer, _transcript = loop.run(question, history.as_messages())
    except AgentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return
    print(answer)
    history.record_turn(question, answer)


def main():
    parser = argparse.ArgumentParser(description="Tool-calling agent over the log/code Chroma collections.")
    parser.add_argument("question", nargs="?", default=None, help="Omit to enter interactive chat mode")
    parser.add_argument("--persist-dir", default="chroma_store")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--chat-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument(
        "--repo-path", default=None, help="Root of the indexed codebase, needed for read_file/list_functions"
    )
    parser.add_argument("--max-distance", type=float, default=None, help="Drop tool matches with distance above this")
    parser.add_argument(
        "--max-iterations", type=int, default=6, help="Cap on tool-call rounds per turn before giving up"
    )
    parser.add_argument("--verbose", action="store_true", help="Log each tool call as it happens")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not check_ollama(args.ollama_host):
        print(f"ERROR: could not reach Ollama at {args.ollama_host}.", file=sys.stderr)
        sys.exit(1)

    config = AgentConfig(
        ollama_host=args.ollama_host,
        embed_model=args.embed_model,
        chat_model=args.chat_model,
        persist_dir=args.persist_dir,
        repo_path=args.repo_path,
        max_iterations=args.max_iterations,
        max_distance=args.max_distance,
    )
    store = ChromaStore(config.persist_dir)
    registry = build_registry(config, store)
    loop = AgentLoop(registry, config.chat_model, config.ollama_host, config.max_iterations)
    history = ConversationHistory()

    if args.question:
        run_turn(loop, history, args.question)
        return

    print(f"Agent chat mode -- tools: {', '.join(registry.names())}")
    print(f"repo-path: {config.repo_path or '(not set -- read_file/list_functions will fail)'}")
    print("Type a question and press Enter. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        run_turn(loop, history, question)
        print()


if __name__ == "__main__":
    main()
