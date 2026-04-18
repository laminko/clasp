from __future__ import annotations

import json
from typing import Any

from ..utils.errors import ParseError
from ..utils.helpers import get_logger
from .events import (
    BaseEvent,
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

logger = get_logger(__name__)


def parse_line(line: str) -> Event | None:
    """Parse a single JSON line from stream-json output into a typed Event.

    Returns None for empty lines or lines that cannot be parsed.
    """
    line = line.strip()
    if not line:
        return None

    try:
        data: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Cannot parse JSON: {exc}", raw=line) from exc

    event_type = data.get("type", "")

    # ── top-level stream-json event types ─────────────────────────────────

    if event_type == "system":
        return SystemEvent(raw=data, subtype=data.get("subtype", ""))

    if event_type == "result":
        usage = data.get("usage", {})
        return ResultEvent(
            raw=data,
            result=data.get("result", ""),
            session_id=data.get("session_id", ""),
            duration_ms=data.get("duration_ms", 0),
            is_error=data.get("is_error", False),
        )

    if event_type == "assistant":
        # The content is nested under data["message"]["content"]
        return _parse_assistant_event(data)

    if event_type == "tool_result":
        content = data.get("content", "")
        if isinstance(content, list):
            # flatten list of content blocks
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            content = "\n".join(parts)
        return ToolResultEvent(
            raw=data,
            tool_use_id=data.get("tool_use_id", ""),
            content=str(content),
            is_error=data.get("is_error", False),
        )

    # ── Anthropic Messages API event subtypes (forwarded verbatim) ────────

    if event_type == "message_start":
        role = data.get("message", {}).get("role", "assistant")
        return MessageStartEvent(raw=data, role=role)

    if event_type == "message_delta":
        delta = data.get("delta", {})
        return MessageCompleteEvent(
            raw=data,
            stop_reason=delta.get("stop_reason", "end_turn"),
        )

    if event_type == "message_stop":
        return MessageCompleteEvent(raw=data, stop_reason="end_turn")

    if event_type == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            return TextChunkEvent(raw=data, text=delta.get("text", ""))

    if event_type == "content_block_start":
        block = data.get("content_block", {})
        if block.get("type") == "tool_use":
            return ToolUseEvent(
                raw=data,
                tool_name=block.get("name", ""),
                tool_use_id=block.get("id", ""),
            )

    logger.debug("Unhandled event type %r: %s", event_type, line[:120])
    return BaseEvent(raw=data)  # type: ignore[return-value]


def _parse_assistant_event(data: dict[str, Any]) -> Event:
    message = data.get("message", {})
    content = message.get("content", [])

    # Return the first recognisable content block
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            return TextChunkEvent(raw=data, text=block.get("text", ""))
        if block.get("type") == "tool_use":
            return ToolUseEvent(
                raw=data,
                tool_name=block.get("name", ""),
                tool_input=block.get("input", {}),
                tool_use_id=block.get("id", ""),
            )

    # Fallback: treat entire message as text
    return TextChunkEvent(raw=data, text=str(content))
