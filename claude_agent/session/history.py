from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types.messages import Message, ToolUse
from datetime import datetime


class MessageHistory:
    """Stores and manages conversation messages for a session."""

    def __init__(self, max_messages: int | None = None) -> None:
        self._messages: list[Message] = []
        self.max_messages = max_messages

    def add(self, message: Message) -> None:
        self._messages.append(message)
        if self.max_messages and len(self._messages) > self.max_messages:
            # Drop oldest message(s) keeping the list within the window
            self._messages = self._messages[-self.max_messages :]

    def add_user(self, content: str) -> Message:
        msg = Message(role="user", content=content)
        self.add(msg)
        return msg

    def add_assistant(self, content: str, tool_uses: list[ToolUse] | None = None) -> Message:
        msg = Message(role="assistant", content=content, tool_uses=tool_uses or [])
        self.add(msg)
        return msg

    def get_all(self) -> list[Message]:
        return list(self._messages)

    def last_assistant(self) -> Message | None:
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg
        return None

    def clear(self) -> None:
        self._messages.clear()

    def export(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.export(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MessageHistory":
        data = json.loads(Path(path).read_text())
        history = cls()
        for item in data:
            tool_uses = [
                ToolUse(
                    tool_name=t["tool_name"],
                    tool_input=t["tool_input"],
                    tool_result=t.get("tool_result"),
                )
                for t in item.get("tool_uses", [])
            ]
            msg = Message(
                role=item["role"],
                content=item["content"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
                tool_uses=tool_uses,
            )
            history.add(msg)
        return history

    def __len__(self) -> int:
        return len(self._messages)
