"""MCP server integration example."""
import asyncio

from claude_agent import ClaudeCLI, MCPManager, Session


async def main() -> None:
    mcp = MCPManager()
    mcp.add_server(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    # Write config to a temp file
    config_path = mcp.write_config_file()
    print(f"MCP config written to: {config_path}")

    try:
        cli = ClaudeCLI()
        session = await Session.create(
            cli,
            bare=True,
            mcp_config_path=config_path,
        )

        response = await session.send("List the files in /tmp using the filesystem MCP server.")
        print(response.result)
    finally:
        mcp.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
