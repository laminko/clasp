"""Tests for streaming parser and handler."""
from __future__ import annotations

import json

import pytest

from claude_agent.streaming.events import (
    MessageCompleteEvent,
    ResultEvent,
    SystemEvent,
    TextChunkEvent,
    ToolUseEvent,
)
from claude_agent.streaming.handler import StreamHandler
from claude_agent.streaming.parser import parse_line
from claude_agent.utils.errors import ParseError


class TestParseLine:
    def test_empty_line_returns_none(self) -> None:
        assert parse_line("") is None
        assert parse_line("   ") is None

    def test_result_event(self) -> None:
        data = {"type": "result", "result": "hello", "session_id": "s1", "duration_ms": 100}
        event = parse_line(json.dumps(data))
        assert isinstance(event, ResultEvent)
        assert event.result == "hello"
        assert event.session_id == "s1"
        assert event.duration_ms == 100

    def test_system_event(self) -> None:
        data = {"type": "system", "subtype": "init"}
        event = parse_line(json.dumps(data))
        assert isinstance(event, SystemEvent)
        assert event.subtype == "init"

    def test_assistant_text_event(self) -> None:
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello world"}],
            },
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, TextChunkEvent)
        assert event.text == "Hello world"

    def test_assistant_tool_use_event(self) -> None:
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "tu_1",
                        "input": {"command": "ls"},
                    }
                ],
            },
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, ToolUseEvent)
        assert event.tool_name == "Bash"
        assert event.tool_input == {"command": "ls"}
        assert event.tool_use_id == "tu_1"

    def test_message_delta_stop(self) -> None:
        data = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
        event = parse_line(json.dumps(data))
        assert isinstance(event, MessageCompleteEvent)
        assert event.stop_reason == "end_turn"

    def test_content_block_delta_text(self) -> None:
        data = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "chunk"},
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, TextChunkEvent)
        assert event.text == "chunk"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_line("{not valid json}")


class TestStreamHandler:
    @pytest.mark.asyncio
    async def test_collect_from_result_event(self) -> None:
        lines = [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps(
                {
                    "type": "result",
                    "result": "final answer",
                    "session_id": "abc",
                    "duration_ms": 500,
                }
            ),
        ]

        async def gen():
            for line in lines:
                yield line

        handler = StreamHandler()
        response = await handler.collect_result(handler.process_stream(gen()))
        assert response.result == "final answer"
        assert response.session_id == "abc"
        assert response.duration_ms == 500

    @pytest.mark.asyncio
    async def test_collect_from_text_chunks(self) -> None:
        lines = [
            json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello "},
            }),
            json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "World"},
            }),
        ]

        async def gen():
            for line in lines:
                yield line

        handler = StreamHandler()
        response = await handler.collect_result(handler.process_stream(gen()))
        assert response.result == "Hello World"

    @pytest.mark.asyncio
    async def test_process_stream_yields_events(self) -> None:
        lines = [
            json.dumps({"type": "result", "result": "done", "session_id": "", "duration_ms": 0}),
        ]

        async def gen():
            for line in lines:
                yield line

        handler = StreamHandler()
        events = []
        async for event in handler.process_stream(gen()):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], ResultEvent)
