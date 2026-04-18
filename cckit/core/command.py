from __future__ import annotations

from typing import Any

from ..types.enums import OutputFormat, PermissionMode
from ..utils.helpers import expand_path


class CommandBuilder:
    """Fluent, type-safe builder for claude CLI commands."""

    def __init__(self, binary_path: str = "~/.local/bin/claude") -> None:
        self._binary = expand_path(binary_path)
        self._flags: list[str] = []
        self._prompt: str | None = None

    # ── prompt ──────────────────────────────────────────────────────────────

    def with_prompt(self, prompt: str) -> "CommandBuilder":
        self._prompt = prompt
        return self

    # ── output ──────────────────────────────────────────────────────────────

    def with_output_format(self, fmt: OutputFormat) -> "CommandBuilder":
        self._flags.extend(["--output-format", fmt.value])
        return self

    # ── model ───────────────────────────────────────────────────────────────

    def with_model(self, model: str) -> "CommandBuilder":
        self._flags.extend(["--model", model])
        return self

    # ── tools ───────────────────────────────────────────────────────────────

    def with_tools(self, tools: list[str]) -> "CommandBuilder":
        for tool in tools:
            self._flags.extend(["--allowedTools", tool])
        return self

    def with_disallowed_tools(self, tools: list[str]) -> "CommandBuilder":
        for tool in tools:
            self._flags.extend(["--disallowedTools", tool])
        return self

    # ── permissions ─────────────────────────────────────────────────────────

    def with_permission_mode(self, mode: PermissionMode) -> "CommandBuilder":
        self._flags.extend(["--permission-mode", mode.value])
        return self

    # ── session ─────────────────────────────────────────────────────────────

    def with_resume(self, session_id: str) -> "CommandBuilder":
        self._flags.extend(["--resume", session_id])
        return self

    def with_continue(self) -> "CommandBuilder":
        self._flags.append("--continue")
        return self

    # ── system prompt ────────────────────────────────────────────────────────

    def with_system_prompt(self, prompt: str) -> "CommandBuilder":
        self._flags.extend(["--system-prompt", prompt])
        return self

    def with_append_system_prompt(self, prompt: str) -> "CommandBuilder":
        self._flags.extend(["--append-system-prompt", prompt])
        return self

    # ── MCP ─────────────────────────────────────────────────────────────────

    def with_mcp_config(self, config_path: str) -> "CommandBuilder":
        self._flags.extend(["--mcp-config", config_path])
        return self

    # ── misc ─────────────────────────────────────────────────────────────────

    def with_bare(self) -> "CommandBuilder":
        self._flags.append("--bare")
        return self

    def with_verbose(self) -> "CommandBuilder":
        self._flags.append("--verbose")
        return self

    def with_max_turns(self, n: int) -> "CommandBuilder":
        self._flags.extend(["--max-turns", str(n)])
        return self

    def with_cwd(self, path: str) -> "CommandBuilder":
        self._flags.extend(["--cwd", expand_path(path)])
        return self

    def add_flag(self, flag: str, value: Any = None) -> "CommandBuilder":
        self._flags.append(flag)
        if value is not None:
            self._flags.append(str(value))
        return self

    # ── build ────────────────────────────────────────────────────────────────

    def build(self) -> list[str]:
        cmd = [self._binary]
        cmd.extend(self._flags)
        if self._prompt is not None:
            cmd.extend(["--print", self._prompt])
        return cmd
