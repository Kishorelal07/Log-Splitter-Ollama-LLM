"""Exception types for the agent package."""


class AgentError(Exception):
    """Base class for all agent-specific errors."""


class MaxIterationsExceededError(AgentError):
    """Raised when the agent loop hits its tool-call round cap without the
    model returning a final answer."""


class UnsafePathError(AgentError):
    """Raised when a file-access tool is asked to read outside the
    configured repo root (path traversal guard)."""
