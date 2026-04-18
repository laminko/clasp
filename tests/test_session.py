"""Tests for Session and MessageHistory."""
from __future__ import annotations

from datetime import datetime

import pytest

from cckit.session.history import MessageHistory
from cckit.types.messages import Message, ToolUse


class TestMessageHistory:
    def test_add_and_get(self) -> None:
        h = MessageHistory()
        h.add_user("hello")
        h.add_assistant("hi there")
        msgs = h.get_all()
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_max_messages(self) -> None:
        h = MessageHistory(max_messages=3)
        for i in range(5):
            h.add_user(f"msg {i}")
        assert len(h) == 3
        # Should keep the last 3
        assert h.get_all()[0].content == "msg 2"

    def test_last_assistant(self) -> None:
        h = MessageHistory()
        h.add_user("q1")
        h.add_assistant("a1")
        h.add_user("q2")
        last = h.last_assistant()
        assert last is not None
        assert last.content == "a1"

    def test_last_assistant_none(self) -> None:
        h = MessageHistory()
        h.add_user("only user")
        assert h.last_assistant() is None

    def test_clear(self) -> None:
        h = MessageHistory()
        h.add_user("hello")
        h.clear()
        assert len(h) == 0

    def test_export_import(self, tmp_path) -> None:
        h = MessageHistory()
        h.add_user("question")
        h.add_assistant("answer", tool_uses=[ToolUse("Bash", {"command": "ls"}, "file.py")])

        path = tmp_path / "history.json"
        h.save(path)

        loaded = MessageHistory.load(path)
        msgs = loaded.get_all()
        assert len(msgs) == 2
        assert msgs[1].tool_uses[0].tool_name == "Bash"

    def test_to_dict_round_trip(self) -> None:
        h = MessageHistory()
        h.add_user("ping")
        exported = h.export()
        assert exported[0]["role"] == "user"
        assert exported[0]["content"] == "ping"
        assert "timestamp" in exported[0]
