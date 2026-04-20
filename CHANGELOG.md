# Changelog

All notable changes to `cckit` are documented here. This project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-04-20

### Added
- **`cckit.FastMCP`** — re-export of the official `mcp` SDK's `FastMCP` server helper. Use it to author custom tools in Python with auto-generated JSON schemas from type hints.
- **`MCPManager.add_python_server(name, *, script | module, python=..., args=..., env=...)`** — one-line registration for a Python MCP server. Defaults `python` to the current interpreter; pass either a script path or a dotted module name.
- **`docs/custom-tools.md`** — full guide covering the mental model, a minimum viable Python MCP server, async/Pydantic/error-handling patterns, and common pitfalls.
- **`examples/mcp_custom_tool.py`** + **`examples/mcp_servers/math_tools.py`** — runnable end-to-end example where Claude calls a custom `fibonacci` / `is_prime` tool suite.
- **`integration` pytest marker** — registered in `pyproject.toml` and excluded by default. Tests that spawn a real `claude` binary now opt in via `pytest -m integration`.

### Dependencies
- Added runtime dependency `mcp>=1.27.0` (the official Model Context Protocol Python SDK).

### Documentation
- Updated `README.md`, `docs/guides.md`, and `docs/api-reference.md` to cover the new MCP tool-authoring surface.

## [0.1.0] — 2026-04-18

- Initial public release under the `cckit` name (renamed from `claude_agent`).
