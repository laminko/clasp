from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Usage":
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_read_tokens=data.get("cache_read_input_tokens", 0),
            cache_write_tokens=data.get("cache_creation_input_tokens", 0),
        )


@dataclass
class Response:
    result: str
    session_id: str = ""
    duration_ms: int = 0
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    model_usage: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Response":
        usage_data = data.get("usage", {})
        return cls(
            result=data.get("result", ""),
            session_id=data.get("session_id", ""),
            duration_ms=data.get("duration_ms", 0),
            usage=Usage.from_dict(usage_data) if usage_data else Usage(),
            stop_reason=data.get("stop_reason", "end_turn"),
            model_usage=data.get("model_usage", {}),
            is_error=data.get("is_error", False),
        )

    @classmethod
    def error(cls, message: str) -> "Response":
        return cls(result=message, is_error=True)


@dataclass
class AgentResult:
    response: Response
    summary: str = ""
    artifacts: list[str] = field(default_factory=list)

    @property
    def result(self) -> str:
        return self.response.result

    @property
    def session_id(self) -> str:
        return self.response.session_id
