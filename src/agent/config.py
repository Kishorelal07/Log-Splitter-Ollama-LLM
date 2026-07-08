"""Central configuration for the agent, built once from CLI args in
agent_cli.py and passed down to every tool so nothing reaches into argparse
or module-level globals directly."""

from dataclasses import dataclass

from ollama_utils import DEFAULT_OLLAMA_HOST


@dataclass(frozen=True)
class AgentConfig:
    ollama_host: str = DEFAULT_OLLAMA_HOST
    embed_model: str = "nomic-embed-text"
    chat_model: str = "qwen2.5:7b-instruct"
    persist_dir: str = "chroma_store"
    repo_path: str | None = None
    max_iterations: int = 6
    max_distance: float | None = None
