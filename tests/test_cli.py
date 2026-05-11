"""Tests for CLI and CommandBuilder."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from cckit import CLIConfig, CLI, CommandBuilder, InputFormat, OutputFormat, PermissionMode
from cckit.core.config import SessionConfig
from cckit.streaming.events import ResultEvent
from cckit.types.enums import PermissionMode as PM
from cckit.utils.errors import CLIError


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

    def test_with_continue(self) -> None:
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .with_continue()
            .with_prompt("hi")
            .build()
        )
        assert "--continue" in cmd

    def test_add_flag_with_and_without_value(self) -> None:
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .add_flag("--no-color")
            .add_flag("--threads", 4)
            .with_prompt("hi")
            .build()
        )
        assert "--no-color" in cmd
        idx = cmd.index("--threads")
        assert cmd[idx + 1] == "4"


class TestCLIBuildCommand:
    def _make_cli(self) -> CLI:
        return CLI(binary_path="/fake/claude")

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

    def test_disallowed_tools(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command(
            "t", SessionConfig(disallowed_tools=["Bash"])
        )
        assert "--disallowedTools" in cmd
        assert "Bash" in cmd

    def test_append_system_prompt(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command(
            "t", SessionConfig(append_system_prompt="extra")
        )
        assert "--append-system-prompt" in cmd

    def test_mcp_config_path(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command(
            "t", SessionConfig(mcp_config_path="/tmp/mcp.json")
        )
        assert "--mcp-config" in cmd
        assert "/tmp/mcp.json" in cmd

    def test_max_turns_and_cwd(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command(
            "t", SessionConfig(max_turns=2, cwd="/tmp")
        )
        assert "--max-turns" in cmd
        assert "2" in cmd
        assert "--cwd" in cmd

    def test_default_model_from_config(self) -> None:
        cli = CLI(
            binary_path="/fake/claude",
            config=CLIConfig(binary_path="/fake/claude", default_model="opus"),
        )
        cmd = cli._build_command("t", SessionConfig())
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "opus"

    def test_default_permission_mode(self) -> None:
        cli = CLI(
            binary_path="/fake/claude",
            config=CLIConfig(
                binary_path="/fake/claude",
                default_permission_mode=PM.BYPASS,
            ),
        )
        cmd = cli._build_command("t", SessionConfig())
        assert "--permission-mode" in cmd

    def test_extra_flags_appended(self) -> None:
        cli = CLI(
            binary_path="/fake/claude",
            config=CLIConfig(
                binary_path="/fake/claude", extra_flags=["--debug"]
            ),
        )
        cmd = cli._build_command("t", SessionConfig())
        assert "--debug" in cmd

    def test_stream_json_always_adds_verbose(self) -> None:
        cli = self._make_cli()
        cmd = cli._build_command(
            "t", SessionConfig(), output_format=OutputFormat.STREAM_JSON
        )
        assert "--verbose" in cmd


class _FakeProcessManager:
    """Stands in for ProcessManager — returns pre-canned output."""

    def __init__(self, *, lines=None, stdout="", stderr="", exit_code=0):
        self.lines = lines or []
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.ran: list[list[str]] = []
        self.stdin_received: list[bytes | None] = []

    def stream_lines(self, cmd, *, cwd=None, stdin=None) -> AsyncIterator[str]:
        self.ran.append(cmd)
        self.stdin_received.append(stdin)

        async def gen():
            for line in self.lines:
                yield line

        return gen()

    async def run(self, cmd, *, cwd=None):
        self.ran.append(cmd)
        return self.stdout, self.stderr, self.exit_code


class TestCLIExecute:
    def _cli_with(self, pm: _FakeProcessManager) -> CLI:
        cli = CLI(binary_path="/fake/claude")
        cli._pm = pm  # type: ignore[assignment]
        return cli

    @pytest.mark.asyncio
    async def test_execute_returns_response_from_stream(self) -> None:
        line = json.dumps(
            {
                "type": "result",
                "result": "done",
                "session_id": "s1",
                "duration_ms": 42,
            }
        )
        pm = _FakeProcessManager(lines=[line])
        cli = self._cli_with(pm)

        response = await cli.execute("hi")
        assert response.result == "done"
        assert response.session_id == "s1"
        assert "/fake/claude" in pm.ran[0][0]

    @pytest.mark.asyncio
    async def test_execute_streaming_yields_events(self) -> None:
        line = json.dumps(
            {"type": "result", "result": "r", "session_id": "", "duration_ms": 0}
        )
        pm = _FakeProcessManager(lines=[line])
        cli = self._cli_with(pm)

        events = []
        async for ev in cli.execute_streaming("hi"):
            events.append(ev)
        assert any(isinstance(e, ResultEvent) for e in events)

    @pytest.mark.asyncio
    async def test_execute_json_returns_dict(self) -> None:
        pm = _FakeProcessManager(stdout='{"result": "done"}')
        cli = self._cli_with(pm)
        data = await cli.execute_json("hi")
        assert data == {"result": "done"}

    @pytest.mark.asyncio
    async def test_execute_json_bad_exit_raises(self) -> None:
        pm = _FakeProcessManager(stdout="", stderr="boom", exit_code=3)
        cli = self._cli_with(pm)
        with pytest.raises(CLIError):
            await cli.execute_json("hi")

    @pytest.mark.asyncio
    async def test_execute_json_bad_output_raises(self) -> None:
        pm = _FakeProcessManager(stdout="not json")
        cli = self._cli_with(pm)
        with pytest.raises(CLIError, match="parse JSON"):
            await cli.execute_json("hi")

    @pytest.mark.asyncio
    async def test_execute_forwards_kwargs(self) -> None:
        line = json.dumps(
            {"type": "result", "result": "", "session_id": "", "duration_ms": 0}
        )
        pm = _FakeProcessManager(lines=[line])
        cli = self._cli_with(pm)
        await cli.execute("hi", tools=["Read"])
        assert "Read" in pm.ran[0]


class TestCommandBuilderInputFormat:
    def test_with_input_format_stream_json_emits_flag(self) -> None:
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .with_input_format(InputFormat.STREAM_JSON)
            .build()
        )
        assert "--input-format" in cmd
        idx = cmd.index("--input-format")
        assert cmd[idx + 1] == "stream-json"

    def test_stream_json_input_omits_positional_prompt(self) -> None:
        """With stream-json input, --print is a bare flag (no positional)."""
        cmd = (
            CommandBuilder("/usr/bin/claude")
            .with_input_format(InputFormat.STREAM_JSON)
            .build()
        )
        # --print should appear, but should NOT be followed by a prompt string
        assert cmd[-1] == "--print"

    def test_stream_json_input_with_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="stream-json"):
            (
                CommandBuilder("/usr/bin/claude")
                .with_input_format(InputFormat.STREAM_JSON)
                .with_prompt("hi")
                .build()
            )


class TestCLIInputMessages:
    def _cli_with(self, pm: _FakeProcessManager) -> CLI:
        cli = CLI(binary_path="/fake/claude")
        cli._pm = pm  # type: ignore[assignment]
        return cli

    @pytest.mark.asyncio
    async def test_execute_streaming_pipes_messages_to_stdin(self) -> None:
        line = json.dumps(
            {"type": "result", "result": "ok", "session_id": "", "duration_ms": 0}
        )
        pm = _FakeProcessManager(lines=[line])
        cli = self._cli_with(pm)

        msgs = [
            {"type": "user", "message": {"role": "user",
                                          "content": [{"type": "text", "text": "hi"}]}}
        ]
        events = []
        async for ev in cli.execute_streaming(input_messages=msgs):
            events.append(ev)

        # Subprocess received NDJSON on stdin
        assert pm.stdin_received[0] is not None
        stdin = pm.stdin_received[0].decode("utf-8")
        assert stdin.endswith("\n")
        assert json.loads(stdin.rstrip("\n")) == msgs[0]

        # Command used stream-json input mode, no positional prompt
        cmd = pm.ran[0]
        assert "--input-format" in cmd and cmd[cmd.index("--input-format") + 1] == "stream-json"
        assert cmd[-1] == "--print"

    @pytest.mark.asyncio
    async def test_execute_streaming_requires_exactly_one_input(self) -> None:
        cli = self._cli_with(_FakeProcessManager())
        with pytest.raises(ValueError, match="exactly one"):
            async for _ in cli.execute_streaming():
                pass
        with pytest.raises(ValueError, match="exactly one"):
            async for _ in cli.execute_streaming(
                "hi", input_messages=[{"type": "user"}]
            ):
                pass

    @pytest.mark.asyncio
    async def test_execute_streaming_text_path_unchanged(self) -> None:
        """Regression: legacy text mode still works and does NOT pipe stdin."""
        line = json.dumps(
            {"type": "result", "result": "ok", "session_id": "", "duration_ms": 0}
        )
        pm = _FakeProcessManager(lines=[line])
        cli = self._cli_with(pm)
        async for _ in cli.execute_streaming("hello"):
            pass
        assert pm.stdin_received[0] is None
        cmd = pm.ran[0]
        assert "--input-format" not in cmd
        idx = cmd.index("--print")
        assert cmd[idx + 1] == "hello"


class TestBaseAgentStreamExecuteMessages:
    @pytest.mark.asyncio
    async def test_agent_forwards_messages_to_cli(self) -> None:
        from cckit import CustomAgent

        line = json.dumps(
            {"type": "result", "result": "ok", "session_id": "", "duration_ms": 0}
        )
        pm = _FakeProcessManager(lines=[line])
        cli = CLI(binary_path="/fake/claude")
        cli._pm = pm  # type: ignore[assignment]
        agent = CustomAgent(cli=cli, system_prompt="be brief", bare=True)

        msgs = [{"type": "user", "message": {"role": "user",
                  "content": [{"type": "text", "text": "hi"}]}}]
        events = []
        async for ev in agent.stream_execute_messages(msgs):
            events.append(ev)

        assert pm.stdin_received[0] is not None
        # Agent's session_config (system_prompt, bare) made it onto the command
        cmd = pm.ran[0]
        assert "--system-prompt" in cmd
        assert "--bare" in cmd
