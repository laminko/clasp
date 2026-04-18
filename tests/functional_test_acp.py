#!/usr/bin/env python3
"""Functional test for the ACP bidirectional JSON-RPC session.

Run directly: uv run python tests/functional_test_acp.py
"""

from __future__ import annotations

import asyncio
import sys

from claude_agent import ACPSession, PermissionPolicy
from claude_agent.streaming.events import (
    MessageCompleteEvent,
    MessageStartEvent,
    ResultEvent,
    TextChunkEvent,
    ToolResultEvent,
    ToolUseEvent,
)


async def test_send() -> bool:
    """Test send() — full response collection."""
    print("\n--- Test 1: send() ---")
    try:
        async with await ACPSession.create(
            permission_policy=PermissionPolicy.AUTO_APPROVE,
        ) as session:
            print(f"  Session created: {session.session_id}")

            response = await session.send(
                "What is 2+2? Reply with ONLY the number, nothing else."
            )
            print(f"  Response: {response.result!r}")
            print(f"  Session ID: {response.session_id}")
            print(f"  Duration: {response.duration_ms}ms")
            print(f"  Is error: {response.is_error}")

            if "4" in response.result:
                print("  PASSED")
                return True
            else:
                print(f"  FAILED — expected '4' in response, got: {response.result!r}")
                return False
    except Exception as e:
        print(f"  FAILED — exception: {e}")
        return False


async def test_stream() -> bool:
    """Test stream() — event-by-event streaming."""
    print("\n--- Test 2: stream() ---")
    try:
        async with await ACPSession.create(
            permission_policy=PermissionPolicy.AUTO_APPROVE,
        ) as session:
            print(f"  Session created: {session.session_id}")

            events = []
            text_parts = []
            print("  Streaming events:")
            async for event in session.stream(
                "Count from 1 to 3, one number per line. Nothing else."
            ):
                events.append(event)
                label = type(event).__name__
                if isinstance(event, TextChunkEvent):
                    text_parts.append(event.text)
                    print(f"    {label}: {event.text!r}")
                elif isinstance(event, ResultEvent):
                    print(f"    {label}: result={event.result!r}")
                elif isinstance(event, ToolUseEvent):
                    print(f"    {label}: {event.tool_name}")
                elif isinstance(event, ToolResultEvent):
                    print(f"    {label}: {event.content[:80]!r}")
                elif isinstance(event, MessageStartEvent):
                    print(f"    {label}")
                elif isinstance(event, MessageCompleteEvent):
                    print(f"    {label}: stop_reason={event.stop_reason}")
                else:
                    print(f"    {label}")

            full_text = "".join(text_parts)
            print(f"  Full text: {full_text!r}")
            print(f"  Total events: {len(events)}")

            if len(events) > 0:
                print("  PASSED")
                return True
            else:
                print("  FAILED — no events received")
                return False
    except Exception as e:
        print(f"  FAILED — exception: {e}")
        return False


async def test_multi_turn() -> bool:
    """Test multi-turn conversation in a single session."""
    print("\n--- Test 3: multi-turn ---")
    try:
        async with await ACPSession.create(
            permission_policy=PermissionPolicy.AUTO_APPROVE,
        ) as session:
            print(f"  Session created: {session.session_id}")

            r1 = await session.send("My favorite color is blue. Just say OK.")
            print(f"  Turn 1: {r1.result!r}")

            r2 = await session.send(
                "What is my favorite color? Reply with ONLY the color, nothing else."
            )
            print(f"  Turn 2: {r2.result!r}")

            if "blue" in r2.result.lower():
                print("  PASSED — context preserved across turns")
                return True
            else:
                print(f"  FAILED — expected 'blue', got: {r2.result!r}")
                return False
    except Exception as e:
        print(f"  FAILED — exception: {e}")
        return False


async def main() -> None:
    print("=" * 60)
    print("ACP Functional Tests")
    print("=" * 60)

    results = []

    results.append(("send()", await test_send()))
    results.append(("stream()", await test_stream()))
    results.append(("multi-turn", await test_multi_turn()))

    print("\n" + "=" * 60)
    print("Results:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
