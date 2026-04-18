"""Bridge ACP ``session/update`` notifications to existing Event types.

The ACP protocol delivers streaming updates as JSON-RPC notifications
with method ``session/update``. This module maps those notification
params to the Event dataclasses already defined in ``streaming/events.py``,
so that ``ACPSession`` consumers see the same event types as the
one-shot ``Session.stream()`` path.

Mapping based on the Claude Code ACP notification subtypes documented
in the Claude Code CLI ``--output-format stream-json`` specification.
"""

from __future__ import annotations

from typing import Any

from .events import (
    Event,
    MessageCompleteEvent,
    MessageStartEvent,
    ResultEvent,
    SystemEvent,
    TextChunkEvent,
    ToolResultEvent,
    ToolUseEvent,
    UsageEvent,
)


def parse_session_update(params: dict[str, Any]) -> Event | None:
    """Convert an ACP ``session/update`` notification into a typed Event.

    Returns ``None`` for unrecognised subtypes.
    """
    subtype = params.get("type", "")
    raw = params

    if subtype == "content_delta":
        return TextChunkEvent(
            raw=raw,
            text=params.get("delta", {}).get("text", ""),
        )

    if subtype == "tool_call_started":
        tool = params.get("tool", {})
        return ToolUseEvent(
            raw=raw,
            tool_name=tool.get("name", ""),
            tool_input=tool.get("input", {}),
            tool_use_id=tool.get("id", ""),
        )

    if subtype == "tool_call_updated":
        tool = params.get("tool", {})
        return ToolUseEvent(
            raw=raw,
            tool_name=tool.get("name", ""),
            tool_input=tool.get("input", {}),
            tool_use_id=tool.get("id", ""),
        )

    if subtype == "tool_result":
        return ToolResultEvent(
            raw=raw,
            tool_use_id=params.get("tool_use_id", ""),
            content=str(params.get("content", "")),
            is_error=params.get("is_error", False),
        )

    if subtype == "assistant_item_started":
        return MessageStartEvent(
            raw=raw,
            role="assistant",
        )

    if subtype == "assistant_item_completed":
        return MessageCompleteEvent(
            raw=raw,
            stop_reason=params.get("stop_reason", "end_turn"),
            session_id=params.get("session_id", ""),
        )

    if subtype == "usage":
        usage = params.get("usage", {})
        return UsageEvent(
            raw=raw,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        )

    if subtype == "result":
        return ResultEvent(
            raw=raw,
            result=params.get("result", ""),
            session_id=params.get("session_id", ""),
            duration_ms=params.get("duration_ms", 0),
            is_error=params.get("is_error", False),
        )

    if subtype == "session_info_update":
        return SystemEvent(
            raw=raw,
            subtype="session_info_update",
        )

    return None
