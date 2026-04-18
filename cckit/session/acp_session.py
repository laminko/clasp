"""Persistent ACP session using bidirectional JSON-RPC over stdio.

``ACPSession`` mirrors the ``Session`` API (``send()`` / ``stream()``),
but uses a long-lived subprocess with NDJSON communication instead of
spawning a new process per message. This enables permission handling,
elicitation, and true multi-turn conversations over a single process.

The ACP (Agent Communication Protocol) approach is based on the Claude
Code CLI's ``--input-format stream-json --output-format stream-json``
mode, which accepts JSON-RPC 2.0 messages on stdin and emits them on
stdout.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ..core.config import ACPConfig
from ..rpc.client import ACPClient
from ..rpc.handlers import DefaultHandlers, PermissionPolicy
from ..rpc.transport import RpcTransport
from ..streaming.acp_parser import parse_session_update
from ..streaming.events import Event, ResultEvent, TextChunkEvent, UsageEvent
from ..types.responses import Response, Usage
from ..utils.helpers import expand_path, get_logger

logger = get_logger(__name__)


class ACPSession:
    """A persistent conversation session over the ACP protocol.

    Use ``create()`` to start a fresh session or ``connect()`` to resume
    an existing one. Both return an ``ACPSession`` that supports
    ``send()`` and ``stream()`` with the same return types as ``Session``.
    """

    def __init__(self, client: ACPClient, session_id: str) -> None:
        self._client = client
        self._session_id = session_id

    # ── factory methods ──────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        *,
        binary_path: str = "~/.local/bin/claude",
        model: str | None = None,
        system_prompt: str | None = None,
        cwd: str | None = None,
        permission_policy: PermissionPolicy = PermissionPolicy.AUTO_DENY,
        config: ACPConfig | None = None,
    ) -> ACPSession:
        """Create a new ACP session, starting the subprocess.

        Args:
            binary_path: Path to the ``claude`` binary.
            model: Model to use for the session.
            system_prompt: System prompt for the session.
            cwd: Working directory for the Claude process.
            permission_policy: How to respond to permission requests.
                Defaults to ``AUTO_DENY`` for safety.
            config: Full configuration (overrides individual args if provided).
        """
        cfg = config or ACPConfig()
        bp = expand_path(cfg.binary_path if config else binary_path)

        cmd = [
            bp,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
        ]

        handlers = DefaultHandlers(
            permission_policy=permission_policy
            if not config
            else PermissionPolicy(cfg.permission_policy),
        )
        transport = RpcTransport(cmd)
        client = ACPClient(transport, handlers=handlers)

        # H1: ensure transport is stopped if initialization fails
        await transport.start()
        try:
            await client.initialize(
                client_info={
                    "name": cfg.client_name if config else "cckit",
                    "version": cfg.client_version if config else "0.1.0",
                }
            )

            session_id = await client.new_session(
                model=model or cfg.model,
                system_prompt=system_prompt,
                cwd=cwd,
            )
        except Exception:
            await transport.stop()
            raise

        return cls(client, session_id)

    @classmethod
    async def connect(
        cls,
        session_id: str,
        *,
        binary_path: str = "~/.local/bin/claude",
        permission_policy: PermissionPolicy = PermissionPolicy.AUTO_DENY,
    ) -> ACPSession:
        """Connect to an existing session by ID."""
        bp = expand_path(binary_path)
        cmd = [
            bp,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
        ]

        handlers = DefaultHandlers(permission_policy=permission_policy)
        transport = RpcTransport(cmd)
        client = ACPClient(transport, handlers=handlers)

        # H1: ensure transport is stopped if initialization fails
        await transport.start()
        try:
            await client.initialize()
            await client.load_session(session_id)
        except Exception:
            await transport.stop()
            raise

        return cls(client, session_id)

    # ── core turn methods ────────────────────────────────────────────────

    async def send(self, message: str) -> Response:
        """Send a message and collect the full response.

        Blocks until the assistant finishes replying, returning a ``Response``
        with the accumulated text, usage, and session metadata.
        """
        events = []
        async for event in self.stream(message):
            events.append(event)

        return self._events_to_response(events)

    async def stream(self, message: str) -> AsyncIterator[Event]:
        """Stream events for a message as they arrive.

        Yields the same ``Event`` types as ``Session.stream()``.
        """
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        def on_update(params: dict[str, Any]) -> None:
            event = parse_session_update(params)
            if event is not None:
                queue.put_nowait(event)
            # A result event signals completion
            if params.get("type") == "result":
                queue.put_nowait(None)

        self._client.on_session_update(on_update)
        try:
            await self._client.prompt(message)

            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            self._client.remove_session_update(on_update)

    # ── utility ──────────────────────────────────────────────────────────

    async def cancel(self) -> None:
        """Cancel the current in-flight prompt."""
        await self._client.cancel()

    async def close(self) -> None:
        """Close the session and stop the subprocess."""
        try:
            await self._client.close_session()
        except Exception:
            logger.debug("Error closing session", exc_info=True)
        await self._client.transport.stop()

    @property
    def session_id(self) -> str:
        return self._session_id

    # ── context manager ──────────────────────────────────────────────────

    async def __aenter__(self) -> ACPSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _events_to_response(events: list[Event]) -> Response:
        """Build a Response from collected events."""
        text_parts: list[str] = []
        session_id = ""
        duration_ms = 0
        usage = Usage()
        is_error = False

        for event in events:
            if isinstance(event, TextChunkEvent):
                text_parts.append(event.text)
            elif isinstance(event, ResultEvent):
                if event.result:
                    text_parts = [event.result]
                session_id = event.session_id
                duration_ms = event.duration_ms
                is_error = event.is_error
            elif isinstance(event, UsageEvent):
                usage = Usage(
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                    cache_read_tokens=event.cache_read_tokens,
                    cache_write_tokens=event.cache_write_tokens,
                )

        return Response(
            result="".join(text_parts),
            session_id=session_id,
            duration_ms=duration_ms,
            usage=usage,
            is_error=is_error,
        )
