"""Default handlers for agent→client JSON-RPC requests.

These handlers respond to requests the Claude subprocess sends back to
the client, such as permission checks, file I/O, and elicitation.

Architecture inspired by the ACP (Agent Communication Protocol) spec
for bidirectional agent↔client communication over JSON-RPC 2.0.
"""

from __future__ import annotations

import inspect
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils.helpers import get_logger

logger = get_logger(__name__)

# Maximum file size (bytes) that handle_file_read will return.
MAX_READ_SIZE = 10 * 1024 * 1024  # 10 MiB


class PermissionPolicy(Enum):
    """Policy for how the client responds to permission requests."""

    AUTO_APPROVE = "auto_approve"
    AUTO_DENY = "auto_deny"
    CALLBACK = "callback"


class DefaultHandlers:
    """Provides default handler implementations for agent→client requests.

    The handler methods match the JSON-RPC methods the Claude subprocess
    may call back into the client (e.g. ``session/request_permission``,
    ``fs/read_text_file``).

    Args:
        permission_policy: How to respond to tool-permission requests.
            Defaults to ``AUTO_DENY`` for safety.
        permission_callback: Required when ``permission_policy`` is
            ``CALLBACK``. Called with the permission params dict.
        workspace_root: Root directory for file I/O confinement. File
            read/write requests are rejected if the resolved path falls
            outside this directory. Defaults to ``Path.cwd()``.
    """

    def __init__(
        self,
        permission_policy: PermissionPolicy = PermissionPolicy.AUTO_DENY,
        permission_callback: Any | None = None,
        workspace_root: Path | str | None = None,
    ) -> None:
        if (
            permission_policy == PermissionPolicy.CALLBACK
            and permission_callback is None
        ):
            raise ValueError(
                "permission_callback is required when "
                "permission_policy is CALLBACK"
            )

        self.permission_policy = permission_policy
        self._permission_callback = permission_callback
        self._workspace_root = Path(
            workspace_root if workspace_root is not None else Path.cwd()
        ).resolve()

    async def handle_permission(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``session/request_permission`` request."""
        logger.info(
            "Permission request: tool=%s",
            params.get("tool_name", "unknown"),
        )

        if self.permission_policy == PermissionPolicy.AUTO_APPROVE:
            return {"approved": True}
        if self.permission_policy == PermissionPolicy.AUTO_DENY:
            return {"approved": False, "reason": "Denied by policy"}
        if self.permission_policy == PermissionPolicy.CALLBACK:
            # Callback is guaranteed non-None by __init__ validation
            result = self._permission_callback(params)
            if inspect.isawaitable(result):
                result = await result
            return result

        # Fail-closed: deny on any unknown policy value
        return {"approved": False, "reason": "Unknown policy"}

    async def handle_file_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``fs/read_text_file`` request.

        Confined to ``workspace_root``. Rejects symlinks, paths outside
        the workspace, and files larger than ``MAX_READ_SIZE``.
        """
        file_path = params.get("path", "")
        logger.info("File read request: %s", file_path)

        try:
            resolved = self._resolve_confined_path(file_path)
        except ValueError as exc:
            return {"error": str(exc)}

        if not resolved.is_file():
            return {"error": "File not found"}

        try:
            size = resolved.stat().st_size
            if size > MAX_READ_SIZE:
                return {
                    "error": f"File too large: {size} bytes "
                    f"(max {MAX_READ_SIZE})"
                }
            content = resolved.read_text(encoding="utf-8")
            return {"content": content}
        except Exception:
            return {"error": "Failed to read file"}

    async def handle_file_write(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``fs/write_text_file`` request.

        Confined to ``workspace_root``. Rejects symlinks and paths
        outside the workspace.
        """
        file_path = params.get("path", "")
        content = params.get("content", "")
        logger.info("File write request: %s", file_path)

        try:
            resolved = self._resolve_confined_path(file_path)
        except ValueError as exc:
            return {"error": str(exc)}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return {"success": True}
        except Exception:
            return {"error": "Failed to write file"}

    async def handle_elicitation(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a ``session/elicitation`` request.

        Elicitation requests ask the user for additional input. The default
        implementation returns an error since there is no interactive user.
        """
        logger.debug(
            "Elicitation request (auto-declining): %s",
            params.get("message", ""),
        )
        return {"error": "Elicitation not supported in non-interactive mode"}

    # ── path confinement ─────────────────────────────────────────────────

    def _resolve_confined_path(self, file_path: str) -> Path:
        """Resolve a path and verify it is inside the workspace root.

        Raises ``ValueError`` if the path escapes the workspace or
        traverses a symlink.
        """
        if not file_path:
            raise ValueError("Empty path")

        path = Path(file_path)

        # Reject paths with .. segments before resolution
        try:
            # Use strict=False so we can check non-existent paths
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Cannot resolve path: {file_path}") from exc

        # Confinement check: resolved path must be inside workspace_root
        try:
            resolved.relative_to(self._workspace_root)
        except ValueError:
            raise ValueError(f"Path outside workspace: {file_path}") from None

        # Symlink check: if the path exists, verify it's not a symlink
        if path.exists() and path.is_symlink():
            raise ValueError(f"Symlinks not allowed: {file_path}")

        return resolved
