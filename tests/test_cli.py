"""Tests for ClaudeCLI and CommandBuilder."""
from __future__ import annotations

import pytest

from claude_agent import CLIConfig, ClaudeCLI, CommandBuilder, OutputFormat, PermissionMode
from claude_agent.core.config import SessionConfig


class TestCommandBuilder:
    def test_basic_prompt(self) -> None:
        cmd = CommandBuilder("/usr/bin/claude").with_prompt("hello").build()
        assert cmd[0] == "/usr/bin/claude"
        assert "--print" in cmd
        assert "hello" in cmd

    def test_output_format(self) -> None:
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .with_output_format(OutputFormat.STREAM_JSON)
            .with_prompt("hi")
            .build()
        )
        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "stream-json"

    def test_tools(self) -> None:
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .with_tools(["Read", "Bash"])
            .with_prompt("hi")
            .build()
        )
        assert cmd.count("--allowedTools") == 2
        assert "Read" in cmd
        assert "Bash" in cmd

    def test_permission_mode(self) -> None:
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .with_permission_mode(PermissionMode.BYPASS)
            .with_prompt("hi")
            .build()
        )
        assert "--permission-mode" in cmd
        idx = cmd.index("--permission-mode")
        assert cmd[idx + 1] == "bypassPermissions"

    def test_bare(self) -> None:
        cmd = CommandBuilder("/usr/bin/claude").with_bare().with_prompt("hi").build()
        assert "--bare" in cmd

    def test_resume(self) -> None:
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .with_resume("abc-123")
            .with_prompt("hi")
            .build()
        )
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "abc-123"

    def test_model(self) -> None:
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .with_model("claude-opus-4-6")
            .with_prompt("hi")
            .build()
        )
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4-6"

    def test_max_turns(self) -> None:
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .with_max_turns(3)
            .with_prompt("hi")
            .build()
        )
        assert "--max-turns" in cmd
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "3"


class TestClaudeCLIBuildCommand:
    def _make_cli(self) -> ClaudeCLI:
        return ClaudeCLI(binary_path="/fake/claude")

    def test_bare_default(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command("test", SessionConfig(bare=True))
        assert "--bare" in cmd

    def test_no_bare(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command("test", SessionConfig(bare=False))
        assert "--bare" not in cmd

    def test_tools_in_command(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command("test", SessionConfig(tools=["Read", "Grep"]))
        assert "Read" in cmd
        assert "Grep" in cmd

    def test_model_in_command(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command("test", SessionConfig(model="sonnet"))
        assert "--model" in cmd
        assert "sonnet" in cmd

    def test_system_prompt(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command("test", SessionConfig(system_prompt="Be concise"))
        assert "--system-prompt" in cmd

    def test_resume_flag(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command("test", SessionConfig(), resume="sess-id-99")
        assert "--resume" in cmd
        assert "sess-id-99" in cmd
