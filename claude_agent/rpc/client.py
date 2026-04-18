"""ACP client — typed wrapper around RpcTransport for the Claude ACP protocol.

Implements the client side of the Agent Communication Protocol (ACP),
which uses JSON-RPC 2.0 over stdio for bidirectional communication
between a client and a long-lived Claude subprocess.

Reference: Claude Code CLI documentation on ``--input-format stream-json``
and ``--output-format stream-json`` for persistent agent sessions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..utils.errors import SessionError
from ..utils.helpers import get_logger
from .handlers import DefaultHandlers
from .transport import RpcTransport

logger = get_logger(__name__)


class ACPClient:
    """High-level ACP client wrapping an RpcTransport.

    Manages the protocol handshake (``initialize``), session lifecycle
    (``session/new``, ``session/load``), and prompt submission. Registers
    default handlers for agent→client callbacks.
    """

    def __init__(
        self,
        transport: RpcTransport,
        handlers: DefaultHandlers | None = None,
    ) -> None:
        self._transport = transport
        self._handlers = handlers or DefaultHandlers()
        self._session_id: str | None = None
        self._session_update_callbacks: list[Callable[..., Any]] = []

        # Register agent→client request handlers
        self._transport.on_request(
            "session/request_permission", self._handlers.handle_permission
        )
        self._transport.on_request(
            "fs/read_text_file", self._handlers.handle_file_read
        )
        self._transport.on_request(
            "fs/write_text_file", self._handlers.handle_file_write
        )
        self._transport.on_request(
            "session/elicitation", self._handlers.handle_elicitation
        )

        # Register notification handler for session updates
        self._transport.on_notification(
            "session/update", self._handle_session_update
        )

    # ── ACP protocol methods ─────────────────────────────────────────────

    async def initialize(
        self,
        client_info: dict[str, Any] | None = None,
        capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send the ``initialize`` handshake request."""
        params: dict[str, Any] = {
            "clientInfo": client_info
            or {"name": "claude-agent", "version": "0.1.0"},
        }
        if capabilities:
            params["capabilities"] = capabilities

        result = await self._transport.request("initialize", params)
        logger.debug("Initialized: %s", result)
        return result

    async def new_session(
        self,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        cwd: str | None = None,
    ) -> str:
        """Create a new conversation session, returning the session_id."""
        params: dict[str, Any] = {}
        if model:
            params["model"] = model
        if system_prompt:
            params["systemPrompt"] = system_prompt
        if cwd:
            params["cwd"] = cwd

        result = await self._transport.request("session/new", params)
        self._session_id = (
            result.get("sessionId", "") if isinstance(result, dict) else ""
        )
        logger.debug("New session: %s", self._session_id)
        return self._session_id

    async def load_session(self, session_id: str) -> dict[str, Any]:
        """Load (resume) an existing session by ID."""
        result = await self._transport.request(
            "session/load", {"sessionId": session_id}
        )
        self._session_id = session_id
        return result

    async def close_session(self) -> None:
        """Close the current session."""
        if self._session_id:
            await self._transport.request(
                "session/close", {"sessionId": self._session_id}
            )
            self._session_id = None

    async def prompt(self, message: str) -> None:
        """Submit a prompt to the current session.

        The response streams back via ``session/update`` notifications
        rather than as a direct JSON-RPC response.
        """
        if not self._session_id:
            raise SessionError("No active session — call new_session() first")

        await self._transport.request(
            "session/prompt",
            {"sessionId": self._session_id, "message": message},
        )

    async def cancel(self) -> None:
        """Cancel the current in-flight prompt."""
        if self._session_id:
            await self._transport.notify(
                "session/cancel",
                {"sessionId": self._session_id},
            )

    # ── callbacks ────────────────────────────────────────────────────────

    def on_session_update(self, callback: Callable[..., Any]) -> None:
        """Register a callback for ``session/update`` notifications."""
        self._session_update_callbacks.append(callback)

    def remove_session_update(self, callback: Callable[..., Any]) -> None:
        """Remove a previously registered session-update callback."""
        try:
            self._session_update_callbacks.remove(callback)
        except ValueError:
            pass

    def _handle_session_update(self, params: dict[str, Any]) -> None:
        """Dispatch session/update notifications to registered callbacks."""
        for cb in list(self._session_update_callbacks):
            try:
                cb(params)
            except Exception:
                logger.exception("session/update callback raised")

    # ── properties ───────────────────────────────────────────────────────

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def transport(self) -> RpcTransport:
        return self._transport

    # ── context manager ──────────────────────────────────────────────────

    async def __aenter__(self) -> ACPClient:
        await self._transport.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            await self.close_session()
        except Exception:
            logger.debug("Error closing session on exit", exc_info=True)
        await self._transport.stop()
