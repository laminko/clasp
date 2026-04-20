# Custom tools via MCP

## The idea in one paragraph

Claude Code has a built-in toolset (`Bash`, `Read`, `WebSearch`, …) that you enable with `tools=[...]`. When you need something the built-ins don't cover — a database lookup, a call to your internal API, a domain-specific calculation — you expose it as an **MCP (Model Context Protocol) server**. Your server runs in its own process, speaks MCP over stdio, and Claude calls it as if it were a native tool. cckit ships a thin layer on top of the official [`mcp`](https://pypi.org/project/mcp/) Python SDK so you can write tools as ordinary Python functions and register them with one line.

## Mental model

```
┌──────────────────┐    stdio (one-way NDJSON)    ┌─────────────┐
│  your Python app │ ───────────────────────────▶ │ claude CLI  │
│ (cckit driver)   │ ◀─────────────────────────── │  binary     │
└──────────────────┘                               └──────┬──────┘
                                                          │ stdio (JSON-RPC, bidirectional)
                                                          ▼
                                                   ┌─────────────┐
                                                   │ YOUR MCP    │
                                                   │ SERVER      │
                                                   │ (subprocess │
                                                   │  of claude) │
                                                   └─────────────┘
```

The `claude` binary spawns your MCP server as a child process. Requests and responses flow bidirectionally between them — which is why permission callbacks, schema validation, and rich results *work* at this layer even though they don't between your Python app and the binary.

## Minimum viable server (2 files, ~40 lines total)

**`my_tools.py`** — defines the tools and runs the server:

```python
from cckit import FastMCP

server = FastMCP("my-tools")

@server.tool(description="Look up a user by ID.")
def lookup_user(user_id: str) -> dict:
    # your real logic here (DB, HTTP call, cache, …)
    return {"id": user_id, "name": "Alice", "team": "platform"}

if __name__ == "__main__":
    server.run()
```

That's a complete MCP server. Parameter JSON schema is auto-generated from the type hints.

**`driver.py`** — registers and uses it:

```python
import asyncio
from pathlib import Path
from cckit import CLI, MCPManager, Session

async def main():
    mcp = MCPManager()
    mcp.add_python_server("my-tools", script=Path("my_tools.py"))
    config_path = mcp.write_config_file()

    try:
        session = await Session.create(
            CLI(),
            bare=False,
            mcp_config_path=config_path,
            tools=["mcp__my-tools__lookup_user"],
        )
        response = await session.send("Look up user u_42 and tell me their team.")
        print(response.result)
    finally:
        mcp.cleanup()

asyncio.run(main())
```

Run it:

```bash
uv run python driver.py
```

Claude will call `mcp__my-tools__lookup_user(user_id="u_42")`, receive your dict, and summarise. The full working version is [`examples/mcp_custom_tool.py`](../examples/mcp_custom_tool.py) with [`examples/mcp_servers/math_tools.py`](../examples/mcp_servers/math_tools.py).

## How tool names get exposed to Claude

Every MCP tool is visible to Claude as `mcp__<server_name>__<tool_name>`:

| You write | Claude sees |
|---|---|
| `FastMCP("my-tools")` + `@server.tool()` def `lookup_user` | `mcp__my-tools__lookup_user` |
| `FastMCP("billing")` + `@server.tool(name="invoice")` def `get_invoice` | `mcp__billing__invoice` |

You must list each name in `tools=[...]` to allow it. Anything not listed is denied.

## Async tools, validation, rich returns

`FastMCP` handles all three:

```python
from pydantic import BaseModel, Field
from cckit import FastMCP

server = FastMCP("orders")

class OrderQuery(BaseModel):
    customer_id: str = Field(..., description="Opaque customer ID")
    limit: int = Field(10, ge=1, le=100)

@server.tool(description="List recent orders for a customer.")
async def list_orders(query: OrderQuery) -> list[dict]:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"/api/orders", params=query.model_dump())
        return r.json()
```

- `async def` is supported natively — the SDK awaits your coroutine.
- Pydantic models as parameters become JSON-schema objects with validation.
- Return `dict`, `list`, `str`, `pydantic.BaseModel`, or anything JSON-serialisable.
- Raise exceptions to signal errors — the SDK turns them into MCP error responses.

## Registering servers that aren't yours

`MCPManager.add_server(name, command, args, env)` works for any MCP-speaking binary. Examples:

```python
# Official filesystem server (npm)
mcp.add_server("filesystem", "npx",
               ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"])

# A server written in a different venv
mcp.add_python_server("data-ops", script="servers/ops.py",
                      python="/opt/dataops/.venv/bin/python")

# A Rust/Go MCP server
mcp.add_server("search", "./target/release/mcp-search", env={"INDEX_PATH": "/data"})
```

## What to put in `tools=[...]`

Two categories, combined freely:

```python
tools = [
    "Read", "Grep", "Glob",                     # built-ins in the claude binary
    "mcp__my-tools__lookup_user",               # your custom MCP tool
    "mcp__filesystem__list_directory",          # 3rd-party MCP tool
]
```

Only allow-listed tools are callable. Use `disallowed_tools=[...]` for fine-grained blocks within a group.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Claude says "I don't have that tool" | Name not in `tools=[...]` | Add the full `mcp__<server>__<tool>` string |
| Server starts then exits immediately | Missing `server.run()` in `__main__` | Keep the `if __name__ == "__main__": server.run()` block |
| "module not found" when spawning | Server imports something not in claude's venv | Use `add_python_server(..., python="/path/to/your/venv/bin/python")` |
| Tool runs but returns nothing | Synchronous function returns `None` | Return a value (any JSON-serialisable type) |

## When NOT to write an MCP server

- If your tool is `Bash`-expressible (shell command, simple script), just allow `Bash`.
- If you only want to *observe* tool calls Claude makes, subscribe to `ToolUseEvent` / `ToolResultEvent` via `session.stream(...)`. No server needed.
- If you want to *block* tool calls based on runtime state, use `permission_mode=PermissionMode.PLAN` or the `--disallowedTools` list. Runtime interception from Python to the binary does not exist in the current CLI.

## Reference

- [`cckit.FastMCP`](../cckit/mcp/__init__.py) — re-export of `mcp.server.fastmcp.FastMCP`
- [`cckit.MCPManager.add_python_server`](../cckit/mcp/manager.py)
- Official MCP spec: <https://modelcontextprotocol.io>
- Runnable example: [`examples/mcp_custom_tool.py`](../examples/mcp_custom_tool.py)
