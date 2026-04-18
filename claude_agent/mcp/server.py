from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPServer:
    """Configuration for a single MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"command": self.command, "args": self.args}
        if self.env:
            cfg["env"] = self.env
        return cfg
