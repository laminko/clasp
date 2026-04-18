"""Tests for agent classes."""
from __future__ import annotations

import pytest

from cckit import CLI, CodeAgent, ConversationAgent, CustomAgent, ResearchAgent
from cckit.agents.base import BaseAgent
from cckit.core.config import SessionConfig


class TestAgentConfig:
    def _make_cli(self) -> CLI:
        return CLI(binary_path="/fake/claude")

    def test_code_agent_default_tools(self) -> None:
        agent = CodeAgent(cli=self._make_cli())
        tools = agent.get_default_tools()
        assert "Read" in tools
        assert "Edit" in tools
        assert "Bash" in tools

    def test_research_agent_default_tools(self) -> None:
        agent = ResearchAgent(cli=self._make_cli())
        tools = agent.get_default_tools()
        assert "WebSearch" in tools
        assert "WebFetch" in tools

    def test_custom_agent_system_prompt(self) -> None:
        agent = CustomAgent(
            name="Poet",
            cli=self._make_cli(),
            system_prompt="Write only haiku.",
        )
        assert agent.get_system_prompt() == "Write only haiku."

    def test_custom_agent_tools(self) -> None:
        agent = CustomAgent(
            cli=self._make_cli(),
            tools=["Read", "Grep"],
        )
        cfg = agent._make_config()
        assert cfg.tools == ["Read", "Grep"]

    def test_with_config_overrides(self) -> None:
        agent = CodeAgent(cli=self._make_cli())
        agent.with_config(model="opus", tools=["Read"])
        cfg = agent._make_config()
        assert cfg.model == "opus"
        assert cfg.tools == ["Read"]

    def test_bare_default_true(self) -> None:
        agent = CodeAgent(cli=self._make_cli())
        cfg = agent._make_config()
        assert cfg.bare is True

    def test_code_agent_system_prompt(self) -> None:
        agent = CodeAgent(cli=self._make_cli())
        assert "engineer" in agent.get_system_prompt().lower()

    def test_conversation_agent_no_tools(self) -> None:
        agent = ConversationAgent(cli=self._make_cli())
        assert agent.get_default_tools() == []
