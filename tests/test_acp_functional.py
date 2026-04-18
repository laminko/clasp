"""Functional tests for ACPSession using a mock ACP server.

These tests exercise the full stack: ACPSession -> ACPClient -> RpcTransport
against a mock subprocess that implements the ACP protocol, validating
end-to-end behavior without needing a real claude binary or authentication.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from claude_agent.rpc.client import ACPClient
from claude_agent.rpc.handlers import DefaultHandlers, PermissionPolicy
from claude_agent.rpc.transport import RpcTransport
from claude_agent.session.acp_session import ACPSession
from claude_agent.streaming.events import (
    MessageCompleteEvent,
    MessageStartEvent,
    ResultEvent,
    TextChunkEvent,
)
from claude_agent.types.responses import Response

MOCK_SERVER = str(Path(__file__).parent / "fixtures" / "mock_acp_server.py")


def _make_session_parts(
    permission_policy: PermissionPolicy = PermissionPolicy.AUTO_APPROVE,
) -> tuple[RpcTransport, ACPClient]:
    """Create transport + client pointed at the mock server."""
    transport = RpcTransport([sys.executable, MOCK_SERVER])
    handlers = DefaultHandlers(permission_policy=permission_policy)
    client = ACPClient(transport, handlers=handlers)
    return transport, client


class TestACPSessionSend:
    @pytest.mark.asyncio
    async def test_send_basic(self) -> None:
        """send() collects streaming chunks into a Response."""
        transport, client = _make_session_parts()
        await transport.start()
        try:
            await client.initialize()
            session_id = await client.new_session()
            session = ACPSession(client, session_id)

            response = await session.send("What is 2+2?")

            assert isinstance(response, Response)
            assert "4" in response.result
            assert response.session_id == "mock-session-001"
            assert response.duration_ms == 42
            assert response.is_error is False
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_send_longer_response(self) -> None:
        """send() handles multi-chunk responses."""
        transport, client = _make_session_parts()
        await transport.start()
        try:
            await client.initialize()
            session_id = await client.new_session()
            session = ACPSession(client, session_id)

            response = await session.send("Tell me something")

            assert response.result == "Hello from mock ACP server!"
            assert response.is_error is False
        finally:
            await transport.stop()


class TestACPSessionStream:
    @pytest.mark.asyncio
    async def test_stream_events(self) -> None:
        """stream() yields individual events."""
        transport, client = _make_session_parts()
        await transport.start()
        try:
            await client.initialize()
            session_id = await client.new_session()
            session = ACPSession(client, session_id)

            events = []
            async for event in session.stream("Count to 3"):
                events.append(event)

            # Should have: MessageStart, 3x TextChunk, MessageComplete, Result
            assert any(isinstance(e, TextChunkEvent) for e in events)
            assert any(isinstance(e, ResultEvent) for e in events)
            assert any(isinstance(e, MessageStartEvent) for e in events)
            assert any(isinstance(e, MessageCompleteEvent) for e in events)

            # Check text content
            text_chunks = [
                e.text for e in events if isinstance(e, TextChunkEvent)
            ]
            assert "".join(text_chunks) == "1\n2\n3\n"
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_stream_result_event(self) -> None:
        """stream() includes a ResultEvent at the end."""
        transport, client = _make_session_parts()
        await transport.start()
        try:
            await client.initialize()
            session_id = await client.new_session()
            session = ACPSession(client, session_id)

            result_events = []
            async for event in session.stream("What is 2+2?"):
                if isinstance(event, ResultEvent):
                    result_events.append(event)

            assert len(result_events) == 1
            assert "4" in result_events[0].result
            assert result_events[0].session_id == "mock-session-001"
        finally:
            await transport.stop()


class TestACPSessionMultiTurn:
    @pytest.mark.asyncio
    async def test_multi_turn(self) -> None:
        """Multiple send() calls work on the same session."""
        transport, client = _make_session_parts()
        await transport.start()
        try:
            await client.initialize()
            session_id = await client.new_session()
            session = ACPSession(client, session_id)

            r1 = await session.send("What is 2+2?")
            assert "4" in r1.result

            r2 = await session.send("Count to 3")
            assert "1\n2\n3\n" in r2.result

            r3 = await session.send("Hello")
            assert "Hello from mock ACP server!" in r3.result
        finally:
            await transport.stop()


class TestACPSessionPermissions:
    @pytest.mark.asyncio
    async def test_auto_approve_permission(self) -> None:
        """Permission requests are auto-approved and don't block the response."""
        transport, client = _make_session_parts(
            permission_policy=PermissionPolicy.AUTO_APPROVE
        )
        await transport.start()
        try:
            await client.initialize()
            session_id = await client.new_session()
            session = ACPSession(client, session_id)

            # The mock server sends a permission request before each response
            response = await session.send("What is 2+2?")
            assert "4" in response.result
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_auto_deny_permission(self) -> None:
        """Permission denials don't crash — the mock still responds."""
        transport, client = _make_session_parts(
            permission_policy=PermissionPolicy.AUTO_DENY
        )
        await transport.start()
        try:
            await client.initialize()
            session_id = await client.new_session()
            session = ACPSession(client, session_id)

            # Even with denied permissions, the mock server still sends the response
            response = await session.send("What is 2+2?")
            assert "4" in response.result
        finally:
            await transport.stop()


class TestACPSessionLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """ACPSession works as an async context manager."""
        transport, client = _make_session_parts()
        await transport.start()
        await client.initialize()
        session_id = await client.new_session()

        async with ACPSession(client, session_id) as session:
            response = await session.send("Hello")
            assert response.result == "Hello from mock ACP server!"

    @pytest.mark.asyncio
    async def test_session_id_property(self) -> None:
        """session_id is accessible after creation."""
        transport, client = _make_session_parts()
        await transport.start()
        try:
            await client.initialize()
            session_id = await client.new_session()
            session = ACPSession(client, session_id)
            assert session.session_id == "mock-session-001"
        finally:
            await transport.stop()


class TestACPClientProtocol:
    @pytest.mark.asyncio
    async def test_initialize_handshake(self) -> None:
        """initialize() returns server capabilities."""
        transport, client = _make_session_parts()
        await transport.start()
        try:
            result = await client.initialize(
                client_info={"name": "test-client", "version": "0.1.0"}
            )
            assert result["capabilities"]["streaming"] is True
            assert result["serverInfo"]["name"] == "mock-claude"
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_new_session(self) -> None:
        """new_session() returns a session ID."""
        transport, client = _make_session_parts()
        await transport.start()
        try:
            await client.initialize()
            session_id = await client.new_session(model="sonnet")
            assert session_id == "mock-session-001"
            assert client.session_id == "mock-session-001"
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_load_session(self) -> None:
        """load_session() resumes an existing session."""
        transport, client = _make_session_parts()
        await transport.start()
        try:
            await client.initialize()
            result = await client.load_session("existing-session-42")
            assert client.session_id == "existing-session-42"
            assert result["loaded"] is True
        finally:
            await transport.stop()
