"""Tests for JSON-RPC 2.0 protocol message types."""

from __future__ import annotations

import json

from cckit.rpc.protocol import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcError,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
)


class TestJsonRpcRequest:
    def test_to_dict(self) -> None:
        req = JsonRpcRequest(
            method="test/method", params={"key": "value"}, id=1
        )
        d = req.to_dict()
        assert d == {
            "jsonrpc": "2.0",
            "method": "test/method",
            "params": {"key": "value"},
            "id": 1,
        }

    def test_to_line(self) -> None:
        req = JsonRpcRequest(method="echo", params={}, id=42)
        line = req.to_line()
        assert line.endswith("\n")
        parsed = json.loads(line)
        assert parsed["method"] == "echo"
        assert parsed["id"] == 42

    def test_defaults(self) -> None:
        req = JsonRpcRequest(method="m")
        assert req.params == {}
        assert req.id == 0


class TestJsonRpcNotification:
    def test_to_dict_has_no_id(self) -> None:
        notif = JsonRpcNotification(
            method="session/update", params={"type": "delta"}
        )
        d = notif.to_dict()
        assert "id" not in d
        assert d["method"] == "session/update"
        assert d["jsonrpc"] == "2.0"

    def test_to_line(self) -> None:
        notif = JsonRpcNotification(method="ping", params={})
        line = notif.to_line()
        parsed = json.loads(line)
        assert "id" not in parsed


class TestJsonRpcError:
    def test_to_dict_without_data(self) -> None:
        err = JsonRpcError(code=-32600, message="Invalid request")
        d = err.to_dict()
        assert d == {"code": -32600, "message": "Invalid request"}
        assert "data" not in d

    def test_to_dict_with_data(self) -> None:
        err = JsonRpcError(
            code=-32000, message="Custom", data={"detail": "foo"}
        )
        d = err.to_dict()
        assert d["data"] == {"detail": "foo"}

    def test_from_dict(self) -> None:
        raw = {"code": -32700, "message": "Parse error", "data": "bad json"}
        err = JsonRpcError.from_dict(raw)
        assert err.code == -32700
        assert err.message == "Parse error"
        assert err.data == "bad json"

    def test_from_dict_minimal(self) -> None:
        err = JsonRpcError.from_dict({})
        assert err.code == 0
        assert err.message == ""
        assert err.data is None


class TestJsonRpcResponse:
    def test_success_to_dict(self) -> None:
        resp = JsonRpcResponse(id=1, result={"answer": 42})
        d = resp.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["result"] == {"answer": 42}
        assert "error" not in d

    def test_error_to_dict(self) -> None:
        err = JsonRpcError(code=-32603, message="Internal error")
        resp = JsonRpcResponse(id=2, error=err)
        d = resp.to_dict()
        assert "error" in d
        assert d["error"]["code"] == -32603
        assert "result" not in d

    def test_from_dict_success(self) -> None:
        raw = {"jsonrpc": "2.0", "id": 5, "result": "ok"}
        resp = JsonRpcResponse.from_dict(raw)
        assert resp.id == 5
        assert resp.result == "ok"
        assert resp.error is None

    def test_from_dict_error(self) -> None:
        raw = {
            "jsonrpc": "2.0",
            "id": 6,
            "error": {"code": -32601, "message": "Not found"},
        }
        resp = JsonRpcResponse.from_dict(raw)
        assert resp.error is not None
        assert resp.error.code == -32601

    def test_roundtrip(self) -> None:
        original = JsonRpcResponse(id=10, result={"data": [1, 2, 3]})
        line = original.to_line()
        restored = JsonRpcResponse.from_dict(json.loads(line))
        assert restored.id == original.id
        assert restored.result == original.result


class TestErrorCodes:
    def test_standard_codes(self) -> None:
        assert PARSE_ERROR == -32700
        assert INVALID_REQUEST == -32600
        assert METHOD_NOT_FOUND == -32601
        assert INTERNAL_ERROR == -32603
