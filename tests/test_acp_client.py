"""Tests for ACPClient using the echo_rpc.py fixture as a mock server."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cckit.rpc.client import ACPClient
from cckit.rpc.handlers import DefaultHandlers, PermissionPolicy
from cckit.rpc.transport import RpcTransport

ECHO_SERVER = str(Path(__file__).parent / "fixtures" / "echo_rpc.py")


def _make_client(
    permission_policy: PermissionPolicy = PermissionPolicy.AUTO_APPROVE,
) -> ACPClient:
    transport = RpcTransport([sys.executable, ECHO_SERVER])
    handlers = DefaultHandlers(permission_policy=permission_policy)
    return ACPClient(transport, handlers=handlers)


class TestACPClientLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        client = _make_client()
        async with client:
            result = await client.initialize()
            assert "capabilities" in result

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        client = _make_client()
        await client.transport.start()
        try:
            result = await client.initialize(
                client_info={"name": "test", "version": "0.0.1"}
            )
            assert isinstance(result, dict)
        finally:
            await client.transport.stop()


class TestACPClientHandlerRegistration:
    @pytest.mark.asyncio
    async def test_handlers_registered_on_transport(self) -> None:
        client = _make_client()
        transport = client.transport
        assert "session/request_permission" in transport._request_handlers
        assert "fs/read_text_file" in transport._request_handlers
        assert "fs/write_text_file" in transport._request_handlers
        assert "session/elicitation" in transport._request_handlers
        assert "session/update" in transport._notification_handlers


class TestACPClientSessionUpdateCallbacks:
    @pytest.mark.asyncio
    async def test_on_session_update_callback(self) -> None:
        client = _make_client()
        received = []
        client.on_session_update(lambda params: received.append(params))

        client._handle_session_update(
            {"type": "content_delta", "text": "hello"}
        )
        assert len(received) == 1
        assert received[0]["type"] == "content_delta"

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self) -> None:
        client = _make_client()
        results_a = []
        results_b = []
        client.on_session_update(lambda p: results_a.append(p))
        client.on_session_update(lambda p: results_b.append(p))

        client._handle_session_update({"type": "test"})
        assert len(results_a) == 1
        assert len(results_b) == 1

    @pytest.mark.asyncio
    async def test_remove_session_update(self) -> None:
        client = _make_client()
        received = []
        cb = lambda p: received.append(p)  # noqa: E731
        client.on_session_update(cb)
        client._handle_session_update({"type": "a"})
        assert len(received) == 1

        client.remove_session_update(cb)
        client._handle_session_update({"type": "b"})
        assert len(received) == 1  # no new events


class TestDefaultHandlers:
    @pytest.mark.asyncio
    async def test_default_is_auto_deny(self) -> None:
        handlers = DefaultHandlers()
        result = await handlers.handle_permission({"tool_name": "Bash"})
        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_auto_approve_permission(self) -> None:
        handlers = DefaultHandlers(
            permission_policy=PermissionPolicy.AUTO_APPROVE
        )
        result = await handlers.handle_permission({"tool_name": "Bash"})
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_auto_deny_permission(self) -> None:
        handlers = DefaultHandlers(permission_policy=PermissionPolicy.AUTO_DENY)
        result = await handlers.handle_permission({"tool_name": "Bash"})
        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_callback_permission(self) -> None:
        async def my_callback(params):
            return {"approved": params.get("tool_name") == "Read"}

        handlers = DefaultHandlers(
            permission_policy=PermissionPolicy.CALLBACK,
            permission_callback=my_callback,
        )
        result = await handlers.handle_permission({"tool_name": "Read"})
        assert result["approved"] is True

        result = await handlers.handle_permission({"tool_name": "Write"})
        assert result["approved"] is False

    def test_callback_without_callback_raises(self) -> None:
        with pytest.raises(ValueError, match="permission_callback is required"):
            DefaultHandlers(permission_policy=PermissionPolicy.CALLBACK)

    @pytest.mark.asyncio
    async def test_file_read_outside_workspace(self) -> None:
        handlers = DefaultHandlers(workspace_root="/tmp/safe")
        result = await handlers.handle_file_read({"path": "/etc/passwd"})
        assert "error" in result
        assert "outside workspace" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_file_read_nonexistent(self, tmp_path) -> None:
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_read(
            {"path": str(tmp_path / "nonexistent.txt")}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_file_read_existing(self, tmp_path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_read({"path": str(test_file)})
        assert result["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_file_read_symlink_rejected(self, tmp_path) -> None:
        real_file = tmp_path / "real.txt"
        real_file.write_text("secret")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)

        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_read({"path": str(link)})
        assert "error" in result
        assert "symlink" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_file_write(self, tmp_path) -> None:
        target = tmp_path / "output.txt"
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_write(
            {"path": str(target), "content": "written"}
        )
        assert result["success"] is True
        assert target.read_text() == "written"

    @pytest.mark.asyncio
    async def test_file_write_outside_workspace(self, tmp_path) -> None:
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_write(
            {"path": "/tmp/evil.txt", "content": "bad"}
        )
        assert "error" in result
        assert "outside workspace" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_file_read_path_traversal(self, tmp_path) -> None:
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_read(
            {"path": str(tmp_path / ".." / ".." / "etc" / "passwd")}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_elicitation_declined(self) -> None:
        handlers = DefaultHandlers()
        result = await handlers.handle_elicitation(
            {"message": "Enter API key:"}
        )
        assert "error" in result
