"""Integration test for ACP lifecycle with a real Claude binary.

Requires the ``claude`` CLI binary to be installed and accessible.
Skipped automatically if not found.
"""

from __future__ import annotations

import shutil

import pytest

from cckit.session.acp_session import ACPSession

pytestmark = pytest.mark.integration

has_claude = shutil.which("claude") is not None


@pytest.mark.skipif(not has_claude, reason="claude binary not found")
class TestACPLifecycle:
    @pytest.mark.asyncio
    async def test_create_prompt_close(self) -> None:
        binary = shutil.which("claude")
        assert binary is not None

        async with await ACPSession.create(binary_path=binary) as session:
            assert session.session_id

            response = await session.send(
                "What is 2+2? Reply with just the number."
            )
            assert "4" in response.result

    @pytest.mark.asyncio
    async def test_streaming(self) -> None:
        binary = shutil.which("claude")
        assert binary is not None

        async with await ACPSession.create(binary_path=binary) as session:
            events = []
            async for event in session.stream("Say hello in one word."):
                events.append(event)

            assert len(events) > 0
