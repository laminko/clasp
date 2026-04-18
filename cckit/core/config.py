from __future__ import annotations

from dataclasses import dataclass, field

from ..types.enums import PermissionMode


@dataclass
class SessionConfig:
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    model: str | None = None
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    mcp_config_path: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    cwd: str | None = None
    bare: bool = True
    verbose: bool = False


@dataclass
class CLIConfig:
    binary_path: str = "~/.local/bin/claude"
    timeout: float | None = None
    default_model: str | None = None
    default_permission_mode: PermissionMode = PermissionMode.DEFAULT
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class ACPConfig:
    """Configuration for ACP (Agent Communication Protocol) sessions.

    Used by ``ACPSession`` to configure the long-lived bidirectional
    JSON-RPC connection to the Claude CLI subprocess.
    """

    binary_path: str = "~/.local/bin/claude"
    model: str | None = None
    permission_policy: str = "auto_approve"
    client_name: str = "cckit"
    client_version: str = "0.1.0"
    request_timeout: float = 30.0
    shutdown_timeout: float = 5.0
