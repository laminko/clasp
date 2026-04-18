from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator

from ..streaming.events import Event
from ..streaming.handler import StreamHandler
from ..types.enums import OutputFormat, PermissionMode
from ..types.responses import Response
from ..utils.errors import CLIError
from ..utils.helpers import expand_path, get_logger
from .command import CommandBuilder
from .config import CLIConfig, SessionConfig
from .process import ProcessManager

logger = get_logger(__name__)


class CLI:
    """High-level async wrapper around the claude CLI binary."""

    def __init__(
        self,
        binary_path: str = "~/.local/bin/claude",
        timeout: float | None = None,
        config: CLIConfig | None = None,
    ) -> None:
        self._config = config or CLIConfig(binary_path=binary_path, timeout=timeout)
        self._pm = ProcessManager(timeout=self._config.timeout)
        self._stream_handler = StreamHandler()

    # ── public API ───────────────────────────────────────────────────────────

    async def execute(
        self,
        prompt: str,
        *,
        session_config: SessionConfig | None = None,
        resume: str | None = None,
        **kwargs,
    ) -> Response:
        """Execute a one-shot prompt and return a Response.

        Extra keyword arguments are forwarded as SessionConfig fields.
        """
        cfg = session_config or SessionConfig(**{k: v for k, v in kwargs.items() if v is not None})
        cmd = self._build_command(prompt, cfg, output_format=OutputFormat.STREAM_JSON, resume=resume)
        events = self._stream_handler.process_stream(self._pm.stream_lines(cmd))
        return await self._stream_handler.collect_result(events)

    async def execute_streaming(
        self,
        prompt: str,
        *,
        session_config: SessionConfig | None = None,
        resume: str | None = None,
        **kwargs,
    ) -> AsyncIterator[Event]:
        """Yield typed Events as the CLI produces them."""
        cfg = session_config or SessionConfig(**{k: v for k, v in kwargs.items() if v is not None})
        cmd = self._build_command(prompt, cfg, output_format=OutputFormat.STREAM_JSON, resume=resume)
        async for event in self._stream_handler.process_stream(self._pm.stream_lines(cmd)):
            yield event

    async def execute_json(
        self,
        prompt: str,
        *,
        session_config: SessionConfig | None = None,
        resume: str | None = None,
        **kwargs,
    ) -> dict:
        """Execute a prompt and return the raw JSON output dict."""
        cfg = session_config or SessionConfig(**{k: v for k, v in kwargs.items() if v is not None})
        cmd = self._build_command(prompt, cfg, output_format=OutputFormat.JSON, resume=resume)
        stdout, stderr, exit_code = await self._pm.run(cmd)
        if exit_code != 0:
            raise CLIError(f"CLI exited {exit_code}", exit_code=exit_code, stderr=stderr)
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise CLIError(f"Could not parse JSON output: {exc}") from exc

    # ── command construction ─────────────────────────────────────────────────

    def _build_command(
        self,
        prompt: str,
        cfg: SessionConfig,
        output_format: OutputFormat = OutputFormat.STREAM_JSON,
        resume: str | None = None,
    ) -> list[str]:
        builder = CommandBuilder(self._config.binary_path)
        builder.with_output_format(output_format)

        # Model
        model = cfg.model or self._config.default_model
        if model:
            builder.with_model(model)

        # Permission mode
        mode = cfg.permission_mode if cfg.permission_mode != PermissionMode.DEFAULT else self._config.default_permission_mode
        if mode != PermissionMode.DEFAULT:
            builder.with_permission_mode(mode)

        # Tools
        if cfg.tools:
            builder.with_tools(cfg.tools)
        if cfg.disallowed_tools:
            builder.with_disallowed_tools(cfg.disallowed_tools)

        # System prompt
        if cfg.system_prompt:
            builder.with_system_prompt(cfg.system_prompt)
        if cfg.append_system_prompt:
            builder.with_append_system_prompt(cfg.append_system_prompt)

        # MCP
        if cfg.mcp_config_path:
            builder.with_mcp_config(cfg.mcp_config_path)

        # Session resume
        if resume:
            builder.with_resume(resume)

        # Misc
        if cfg.max_turns is not None:
            builder.with_max_turns(cfg.max_turns)
        if cfg.cwd:
            builder.with_cwd(cfg.cwd)
        if cfg.bare:
            builder.with_bare()
        # stream-json requires --verbose to produce any stdout output
        if cfg.verbose or output_format == OutputFormat.STREAM_JSON:
            builder.with_verbose()

        # Extra global flags
        for flag in self._config.extra_flags:
            builder.add_flag(flag)

        builder.with_prompt(prompt)
        return builder.build()
