"""Tests for RpcTransport using the echo_rpc.py fixture."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cckit.rpc.transport import RpcTransport
from cckit.utils.errors import CLIError, RpcError, TransportError

ECHO_SERVER = str(Path(__file__).parent / "fixtures" / "echo_rpc.py")


def _make_transport() -> RpcTransport:
    return RpcTransport([sys.executable, ECHO_SERVER])


class TestTransportLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        transport = _make_transport()
        await transport.start()
        assert transport._proc is not None
        assert transport._proc.returncode is None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        transport = _make_transport()
        await transport.start()
        await transport.stop()
        await transport.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_binary_not_found(self) -> None:
        transport = RpcTransport(["/nonexistent/binary"])
        with pytest.raises(CLIError, match="Binary not found"):
            await transport.start()


class TestTransportRequests:
    @pytest.mark.asyncio
    async def test_echo_request(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            result = await transport.request("echo", {"hello": "world"})
            assert result == {"hello": "world"}
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            result = await transport.request("initialize", {})
            assert "capabilities" in result
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_error_response_raises_rpc_error(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            with pytest.raises(RpcError, match="Test error"):
                await transport.request("error", {})
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            with pytest.raises(TransportError, match="timed out"):
                await transport.request("slow", {"delay": 10}, timeout=0.1)
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_request_after_stop_raises(self) -> None:
        transport = _make_transport()
        await transport.start()
        await transport.stop()
        with pytest.raises(TransportError, match="not connected"):
            await transport.request("echo", {})


class TestTransportNotifications:
    @pytest.mark.asyncio
    async def test_send_notification(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            # Should not raise (fire-and-forget)
            await transport.notify("some/event", {"data": 1})
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_receive_notification(self) -> None:
        transport = _make_transport()
        received: list[dict] = []
        transport.on_notification(
            "test/notification", lambda params: received.append(params)
        )

        await transport.start()
        try:
            result = await transport.request("notify_back", {"msg": "hello"})
            assert result["notified"] is True

            # Give the reader loop a moment to process the notification
            import asyncio

            await asyncio.sleep(0.1)
            assert len(received) == 1
            assert received[0]["msg"] == "hello"
        finally:
            await transport.stop()


class TestTransportIncomingRequests:
    @pytest.mark.asyncio
    async def test_handle_incoming_request(self) -> None:
        transport = _make_transport()

        async def handle_callback(params):
            return {"handled": True, "echo": params.get("msg", "")}

        transport.on_request("test/callback", handle_callback)

        await transport.start()
        try:
            result = await transport.request("callback", {"msg": "ping"})
            assert result["callback_result"]["handled"] is True
            assert result["callback_result"]["echo"] == "ping"
        finally:
            await transport.stop()


class TestMultipleRequests:
    @pytest.mark.asyncio
    async def test_sequential_requests(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            r1 = await transport.request("echo", {"n": 1})
            r2 = await transport.request("echo", {"n": 2})
            r3 = await transport.request("echo", {"n": 3})
            assert r1 == {"n": 1}
            assert r2 == {"n": 2}
            assert r3 == {"n": 3}
        finally:
            await transport.stop()
