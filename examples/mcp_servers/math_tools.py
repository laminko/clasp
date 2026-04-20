"""A minimal Python MCP server exposing two tools to Claude.

Run directly to serve over stdio (no arguments)::

    python examples/mcp_servers/math_tools.py

Register it with cckit via ``MCPManager.add_python_server(...)`` — see
``examples/mcp_custom_tool.py`` for the driver.
"""

from __future__ import annotations

from cckit import FastMCP

server = FastMCP("math-tools")


@server.tool(description="Return the nth Fibonacci number (0-indexed).")
def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@server.tool(description="Check whether an integer is prime.")
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


if __name__ == "__main__":
    server.run()
