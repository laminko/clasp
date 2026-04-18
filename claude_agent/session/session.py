from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from ..core.cli import ClaudeCLI
from ..core.config import SessionConfig
from ..streaming.events import Event
from ..types.messages import Message
from ..types.responses import Response
from ..utils.helpers import get_logger
from .history import MessageHistory

logger = get_logger(__name__)


class Session:
    """A single conversation session that preserves context across turns.

    The Claude CLI uses ``--resume <session_id>`` to re-attach to an existing
    conversation.  On the first turn we run without --resume and capture the
    session_id from the result; subsequent turns pass it back.
    """

    def __init__(
        self,
        cli: ClaudeCLI,
        config: SessionConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self._cli = cli
        self.config = config or SessionConfig()
        self.session_id: str = session_id or ""
        self._local_id: str = str(uuid.uuid4())
        self.history = MessageHistory()

    # ── factory methods ─────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        cli: ClaudeCLI,
        *,
        tools: list[str] | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        bare: bool = True,
        **kwargs,
    ) -> "Session":
        config = SessionConfig(
            tools=tools,
            model=model,
            system_prompt=system_prompt,
            bare=bare,
            **kwargs,
        )
        return cls(cli, config=config)

    @classmethod
    def resume(cls, cli: ClaudeCLI, session_id: str, config: SessionConfig | None = None) -> "Session":
        """Re-attach to an existing CLI session by its session_id."""
        return cls(cli, config=config, session_id=session_id)

    # ── core turn methods ────────────────────────────────────────────────────

    async def send(self, message: str) -> Response:
        """Send a message and return the full Response."""
        self.history.add_user(message)
        response = await self._cli.execute(
            message,
            session_config=self.config,
            resume=self.session_id or None,
        )
        if response.session_id:
            self.session_id = response.session_id
        self.history.add_assistant(response.result)
        logger.debug("Session %s: got response (session_id=%s)", self._local_id, self.session_id)
        return response

    async def stream(self, message: str) -> AsyncIterator[Event]:
        """Stream events for a message, updating history on completion."""
        self.history.add_user(message)
        chunks: list[str] = []
        async for event in self._cli.execute_streaming(
            message,
            session_config=self.config,
            resume=self.session_id or None,
        ):
            from ..streaming.events import ResultEvent, TextChunkEvent
            if isinstance(event, TextChunkEvent):
                chunks.append(event.text)
            elif isinstance(event, ResultEvent):
                if event.session_id:
                    self.session_id = event.session_id
                if event.result:
                    chunks = [event.result]
            yield event

        self.history.add_assistant("".join(chunks))

    # ── utility ─────────────────────────────────────────────────────────────

    def get_history(self) -> list[Message]:
        return self.history.get_all()

    def fork(self) -> "Session":
        """Return a new Session with the same config and session_id (shared context)."""
        forked = Session(self._cli, config=self.config, session_id=self.session_id)
        # Copy history snapshot
        for msg in self.history.get_all():
            forked.history.add(msg)
        return forked

    def clear_history(self) -> None:
        self.history.clear()
        self.session_id = ""
