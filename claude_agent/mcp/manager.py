from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ..utils.helpers import get_logger
from .server import MCPServer

logger = get_logger(__name__)


class MCPManager:
    """Manages MCP server configurations and produces the JSON config file
    consumed by the claude CLI via ``--mcp-config``."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}
        self._tmp_file: Path | None = None

    def add_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> "MCPManager":
        self._servers[name] = MCPServer(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
        )
        return self

    def remove_server(self, name: str) -> None:
        self._servers.pop(name, None)

    def to_config(self) -> dict[str, Any]:
        """Return the mcpServers dict that claude CLI expects."""
        return {"mcpServers": {s.name: s.to_dict() for s in self._servers.values()}}

    def write_config_file(self, path: str | Path | None = None) -> str:
        """Write config JSON to *path* (or a temp file) and return the path."""
        config = self.to_config()
        if path is None:
            # Create a temp file that persists until explicitly cleaned up
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="claude_mcp_"
            )
            tmp.write(json.dumps(config, indent=2))
            tmp.close()
            self._tmp_file = Path(tmp.name)
            logger.debug("Wrote MCP config to %s", self._tmp_file)
            return str(self._tmp_file)
        else:
            target = Path(path)
            target.write_text(json.dumps(config, indent=2))
            return str(target)

    def cleanup(self) -> None:
        """Remove any temporary config file created by write_config_file."""
        if self._tmp_file and self._tmp_file.exists():
            self._tmp_file.unlink()
            self._tmp_file = None

    def __repr__(self) -> str:
        return f"MCPManager(servers={list(self._servers)})"
