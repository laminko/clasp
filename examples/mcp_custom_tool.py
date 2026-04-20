"""Drive a cckit session that calls into a Python-authored MCP server.

Pipeline:
  1. Register ``examples/mcp_servers/math_tools.py`` via ``MCPManager``.
  2. Launch a cckit session with ``--mcp-config`` pointing at that config.
  3. Ask Claude to use the custom ``fibonacci`` / ``is_prime`` tools.

Claude's output should mention the tool names ``mcp__math-tools__fibonacci``
and ``mcp__math-tools__is_prime`` in its tool-use events.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from cckit import CLI, MCPManager, Session, ToolUseEvent


async def main() -> None:
    server_script = Path(__file__).parent / "mcp_servers" / "math_tools.py"

    mcp = MCPManager()
    mcp.add_python_server("math-tools", script=server_script)
    config_path = mcp.write_config_file()
    print(f"MCP config: {config_path}")

    try:
        cli = CLI()
        session = await Session.create(
            cli,
            bare=False,
            mcp_config_path=config_path,
            tools=[
                "mcp__math-tools__fibonacci",
                "mcp__math-tools__is_prime",
            ],
        )

        prompt = (
            "Using the provided tools only, tell me: "
            "(1) the 15th Fibonacci number, and "
            "(2) whether that number is prime. "
            "Show the tool calls you made."
        )

        print("\n--- streaming events ---")
        async for event in session.stream(prompt):
            if isinstance(event, ToolUseEvent):
                print(f"  [tool_use] {event.tool_name}({event.tool_input})")

        response = await session.send(
            "Summarise the results in one sentence."
        )
        print("\n--- final answer ---")
        print(response.result)
    finally:
        mcp.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
