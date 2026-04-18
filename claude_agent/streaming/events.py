from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseEvent:
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class TextChunkEvent(BaseEvent):
    """A partial text delta from the assistant."""
    text: str = ""


@dataclass
class ToolUseEvent(BaseEvent):
    """The assistant is invoking a tool."""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str = ""


@dataclass
class ToolResultEvent(BaseEvent):
    """Result returned by a tool."""
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class MessageStartEvent(BaseEvent):
    """Start of a new message."""
    role: str = "assistant"


@dataclass
class MessageCompleteEvent(BaseEvent):
    """A full message has completed."""
    stop_reason: str = "end_turn"
    session_id: str = ""


@dataclass
class UsageEvent(BaseEvent):
    """Token usage report."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ResultEvent(BaseEvent):
    """Final result event (stream-json type=result)."""
    result: str = ""
    session_id: str = ""
    duration_ms: int = 0
    is_error: bool = False


@dataclass
class SystemEvent(BaseEvent):
    """Misc system-level event."""
    subtype: str = ""


# Union type for type narrowing
Event = (
    TextChunkEvent
    | ToolUseEvent
    | ToolResultEvent
    | MessageStartEvent
    | MessageCompleteEvent
    | UsageEvent
    | ResultEvent
    | SystemEvent
)
