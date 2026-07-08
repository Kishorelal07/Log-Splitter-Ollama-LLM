"""Conversation history for the agent's chat mode.

Only the user's question and the model's *final* answer are retained across
turns -- the intermediate tool-call / tool-result messages generated while
answering a given turn are discarded once that turn finishes. This keeps
prompt size roughly constant per turn in a long chat session instead of
re-sending every retrieved log/code snippet on every subsequent question."""

from dataclasses import dataclass, field


@dataclass
class ConversationHistory:
    _messages: list = field(default_factory=list)

    def as_messages(self) -> list:
        return list(self._messages)

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        self._messages.append({"role": "user", "content": user_text})
        self._messages.append({"role": "assistant", "content": assistant_text})

    def clear(self) -> None:
        self._messages.clear()
