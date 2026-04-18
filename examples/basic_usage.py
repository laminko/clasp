"""Basic one-shot usage of CLI."""
import asyncio

from cckit import CLI


async def main() -> None:
    cli = CLI()

    response = await cli.execute(
        "What is 2 + 2? Answer in one line.",
        bare=False,
    )

    print("Result:", response.result)
    print("Session ID:", response.session_id)
    print("Duration:", response.duration_ms, "ms")


if __name__ == "__main__":
    asyncio.run(main())
