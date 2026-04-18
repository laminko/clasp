from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Standard JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603


@dataclass
class JsonRpcError:
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JsonRpcError:
        return cls(
            code=data.get("code", 0),
            message=data.get("message", ""),
            data=data.get("data"),
        )


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 request (client → server)."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }

    def to_line(self) -> str:
        return json.dumps(self.to_dict()) + "\n"


@dataclass
class JsonRpcNotification:
    """JSON-RPC 2.0 notification (no id, no response expected)."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
        }

    def to_line(self) -> str:
        return json.dumps(self.to_dict()) + "\n"


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response."""

    id: int | None = None
    result: Any = None
    error: JsonRpcError | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"jsonrpc": "2.0", "id": self.id}
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d

    def to_line(self) -> str:
        return json.dumps(self.to_dict()) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JsonRpcResponse:
        error_data = data.get("error")
        error = JsonRpcError.from_dict(error_data) if error_data else None
        return cls(
            id=data.get("id"),
            result=data.get("result"),
            error=error,
        )
