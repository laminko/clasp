from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class ToolUse:
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: str | None = None


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tool_uses: list[ToolUse] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_uses": [
                {
                    "tool_name": t.tool_name,
                    "tool_input": t.tool_input,
                    "tool_result": t.tool_result,
                }
                for t in self.tool_uses
            ],
        }
