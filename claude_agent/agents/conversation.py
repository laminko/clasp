from __future__ import annotations

from ..core.cli import ClaudeCLI
from ..session.session import Session
from ..types.responses import AgentResult
from .base import BaseAgent


class ConversationAgent(BaseAgent):
    """A general-purpose multi-turn chat agent."""

    def __init__(
        self,
        cli: ClaudeCLI | None = None,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        bare: bool = True,
        binary_path: str = "~/.local/bin/claude",
    ) -> None:
        super().__init__(
            cli,
            model=model,
            system_prompt=system_prompt,
            bare=bare,
            binary_path=binary_path,
        )
        self._session: Session | None = None

    def get_default_tools(self) -> list[str]:
        return []

    def get_system_prompt(self) -> str:
        return ""

    async def start(self) -> "ConversationAgent":
        """Initialise the internal session."""
        self._session = await Session.create(self._cli)
        self._session.config = self._make_config()
        return self

    async def chat(self, message: str) -> AgentResult:  # type: ignore[override]
        """Send a message and return a result.  Automatically creates a session."""
        if self._session is None:
            await self.start()
        assert self._session is not None
        response = await self._session.send(message)
        return self._make_result(response)

    def get_session(self) -> Session | None:
        return self._session

    def reset(self) -> None:
        """Clear session state to start a new conversation."""
        if self._session:
            self._session.clear_history()
