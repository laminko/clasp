"""Stream events in real-time."""
import asyncio

from cckit import CLI, TextChunkEvent, ToolUseEvent


async def main() -> None:
    cli = CLI()

    print("Streaming response:\n")
    async for event in cli.execute_streaming(
        "Count from 1 to 5, one number per line.",
        bare=False,
    ):
        if isinstance(event, TextChunkEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolUseEvent):
            print(f"\n[Tool: {event.tool_name}]")

    print("\n\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
