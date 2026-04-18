"""Tests for ACP session/update notification → Event mapping."""

from __future__ import annotations

from claude_agent.streaming.acp_parser import parse_session_update
from claude_agent.streaming.events import (
    MessageCompleteEvent,
    MessageStartEvent,
    ResultEvent,
    SystemEvent,
    TextChunkEvent,
    ToolResultEvent,
    ToolUseEvent,
    UsageEvent,
)


class TestParseSessionUpdate:
    def test_content_delta(self) -> None:
        params = {"type": "content_delta", "delta": {"text": "Hello"}}
        event = parse_session_update(params)
        assert isinstance(event, TextChunkEvent)
        assert event.text == "Hello"

    def test_tool_call_started(self) -> None:
        params = {
            "type": "tool_call_started",
            "tool": {"name": "Bash", "id": "tu_1", "input": {"command": "ls"}},
        }
        event = parse_session_update(params)
        assert isinstance(event, ToolUseEvent)
        assert event.tool_name == "Bash"
        assert event.tool_use_id == "tu_1"
        assert event.tool_input == {"command": "ls"}

    def test_tool_call_updated(self) -> None:
        params = {
            "type": "tool_call_updated",
            "tool": {"name": "Read", "id": "tu_2", "input": {"path": "/tmp"}},
        }
        event = parse_session_update(params)
        assert isinstance(event, ToolUseEvent)
        assert event.tool_name == "Read"

    def test_tool_result(self) -> None:
        params = {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "file contents here",
            "is_error": False,
        }
        event = parse_session_update(params)
        assert isinstance(event, ToolResultEvent)
        assert event.tool_use_id == "tu_1"
        assert event.content == "file contents here"
        assert event.is_error is False

    def test_assistant_item_started(self) -> None:
        params = {"type": "assistant_item_started"}
        event = parse_session_update(params)
        assert isinstance(event, MessageStartEvent)
        assert event.role == "assistant"

    def test_assistant_item_completed(self) -> None:
        params = {
            "type": "assistant_item_completed",
            "stop_reason": "end_turn",
            "session_id": "s123",
        }
        event = parse_session_update(params)
        assert isinstance(event, MessageCompleteEvent)
        assert event.stop_reason == "end_turn"
        assert event.session_id == "s123"

    def test_usage(self) -> None:
        params = {
            "type": "usage",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5,
            },
        }
        event = parse_session_update(params)
        assert isinstance(event, UsageEvent)
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.cache_read_tokens == 10
        assert event.cache_write_tokens == 5

    def test_result(self) -> None:
        params = {
            "type": "result",
            "result": "final answer",
            "session_id": "s456",
            "duration_ms": 1234,
            "is_error": False,
        }
        event = parse_session_update(params)
        assert isinstance(event, ResultEvent)
        assert event.result == "final answer"
        assert event.session_id == "s456"
        assert event.duration_ms == 1234

    def test_session_info_update(self) -> None:
        params = {"type": "session_info_update", "session_id": "s789"}
        event = parse_session_update(params)
        assert isinstance(event, SystemEvent)
        assert event.subtype == "session_info_update"

    def test_unknown_type_returns_none(self) -> None:
        params = {"type": "unknown_event"}
        assert parse_session_update(params) is None

    def test_empty_params_returns_none(self) -> None:
        assert parse_session_update({}) is None

    def test_raw_field_preserved(self) -> None:
        params = {
            "type": "content_delta",
            "delta": {"text": "x"},
            "extra": "data",
        }
        event = parse_session_update(params)
        assert event is not None
        assert event.raw == params
