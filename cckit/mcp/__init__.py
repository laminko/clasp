"""MCP integration: register external MCP servers and author your own tools.

- ``MCPManager`` / ``MCPServer`` — register arbitrary MCP servers (npx-based,
  Python, or any other stdio binary) and produce the JSON config consumed by
  the ``claude`` CLI via ``--mcp-config``.
- ``FastMCP`` — re-exported from the official ``mcp`` SDK. Use it in your
  own Python file to define tools with ``@server.tool()``. That file becomes
  a subprocess that Claude calls over stdio.

See ``docs/custom-tools.md`` for the end-to-end pattern.
"""

from mcp.server.fastmcp import FastMCP

from .manager import MCPManager
from .server import MCPServer

__all__ = ["FastMCP", "MCPManager", "MCPServer"]
