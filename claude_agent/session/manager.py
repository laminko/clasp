from __future__ import annotations

from ..core.cli import ClaudeCLI
from ..core.config import SessionConfig
from ..utils.helpers import get_logger
from .session import Session

logger = get_logger(__name__)


class ConversationManager:
    """Manages a collection of named Sessions."""

    def __init__(self, cli: ClaudeCLI) -> None:
        self._cli = cli
        self._sessions: dict[str, Session] = {}

    async def new_session(
        self,
        name: str | None = None,
        *,
        config: SessionConfig | None = None,
        **kwargs,
    ) -> Session:
        session = await Session.create(self._cli, **(kwargs or {}))
        if config:
            session.config = config
        key = name or session._local_id
        self._sessions[key] = session
        logger.debug("Created session %s", key)
        return session

    def get(self, name: str) -> Session | None:
        return self._sessions.get(name)

    def resume(self, session_id: str, name: str | None = None) -> Session:
        session = Session.resume(self._cli, session_id)
        key = name or session_id
        self._sessions[key] = session
        return session

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def remove(self, name: str) -> None:
        self._sessions.pop(name, None)

    def clear(self) -> None:
        self._sessions.clear()
