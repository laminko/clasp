"""Tests for ACPSession using mock client internals."""

from __future__ import annotations

from claude_agent.session.acp_session import ACPSession
from claude_agent.streaming.events import ResultEvent, TextChunkEvent
from claude_agent.types.responses import Response


class TestEventsToResponse:
    def test_text_chunks_concatenated(self) -> None:
        events = [
            TextChunkEvent(text="Hello "),
            TextChunkEvent(text="World"),
        ]
        resp = ACPSession._events_to_response(events)
        assert isinstance(resp, Response)
        assert resp.result == "Hello World"
        assert resp.is_error is False

    def test_result_event_overrides_chunks(self) -> None:
        events = [
            TextChunkEvent(text="partial"),
            ResultEvent(
                result="final answer", session_id="s1", duration_ms=500
            ),
        ]
        resp = ACPSession._events_to_response(events)
        assert resp.result == "final answer"
        assert resp.session_id == "s1"
        assert resp.duration_ms == 500

    def test_empty_events(self) -> None:
        resp = ACPSession._events_to_response([])
        assert resp.result == ""

    def test_error_result(self) -> None:
        events = [
            ResultEvent(
                result="something went wrong", is_error=True, session_id="s2"
            ),
        ]
        resp = ACPSession._events_to_response(events)
        assert resp.is_error is True
        assert resp.result == "something went wrong"
