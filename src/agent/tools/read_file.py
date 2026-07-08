"""read_file tool: reads a file fresh from disk, optionally a line range.
This is where a code answer's actual content comes from -- search_code only
ever returns coordinates, never text."""

import os

from agent.errors import UnsafePathError
from agent.tools.base import BaseTool


def resolve_safe_path(repo_path: str, relative_path: str) -> str:
    """Resolves relative_path against repo_path and guarantees the result
    stays inside repo_path, rejecting '..' escapes, absolute-path overrides,
    or a different drive entirely."""
    repo_root = os.path.realpath(repo_path)
    candidate = os.path.realpath(os.path.join(repo_root, relative_path))
    try:
        common = os.path.commonpath([repo_root, candidate])
    except ValueError:
        raise UnsafePathError(f"'{relative_path}' resolves outside the repo root") from None
    if common != repo_root:
        raise UnsafePathError(f"'{relative_path}' resolves outside the repo root")
    return candidate


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Reads the current contents of a file from the indexed codebase, "
        "fresh from disk. Optionally restrict to a line range (1-indexed, "
        "inclusive) -- use this after search_code or list_functions to see "
        "the actual code at a specific location, or omit the range to read "
        "an entire file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the repo root"},
            "start_line": {"type": "integer", "description": "Optional 1-indexed start line (inclusive)"},
            "end_line": {"type": "integer", "description": "Optional 1-indexed end line (inclusive)"},
        },
        "required": ["path"],
    }

    def __init__(self, repo_path: str | None, max_lines: int = 400):
        self._repo_path = repo_path
        self._max_lines = max_lines

    def execute(self, path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
        if not self._repo_path:
            raise ValueError("No --repo-path was configured for this agent session")

        abs_path = resolve_safe_path(self._repo_path, path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"'{path}' does not exist under the configured repo root")

        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()

        total_lines = len(lines)
        start = max(1, start_line or 1)
        requested_end = end_line or total_lines
        end = min(total_lines, requested_end)

        truncated = False
        if end - start + 1 > self._max_lines:
            end = start + self._max_lines - 1
            truncated = True

        content = "\n".join(lines[start - 1 : end])

        return {
            "path": path,
            "start_line": start,
            "end_line": end,
            "total_lines": total_lines,
            "truncated": truncated,
            "content": content,
        }
