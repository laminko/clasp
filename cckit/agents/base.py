from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..core.cli import CLI
from ..core.config import SessionConfig
from ..session.session import Session
from ..streaming.events import Event
from ..types.responses import AgentResult, Response
from ..utils.helpers import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        cli: CLI | None = None,
        *,
        model: str | None = None,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        bare: bool = True,
        binary_path: str = "~/.local/bin/claude",
    ) -> None:
        self._cli = cli or CLI(binary_path=binary_path)
        self._model = model
        self._tools = tools
        self._system_prompt = system_prompt
        self._bare = bare

    # ── abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def get_default_tools(self) -> list[str]:
        """Return the default tool set for this agent type."""

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent-specific system prompt fragment."""

    # ── concrete execute methods ─────────────────────────────────────────────

    async def execute(self, task: str) -> AgentResult:
        """Execute a task and return an AgentResult."""
        response = await self._cli.execute(task, session_config=self._make_config())
        return self._make_result(response)

    async def stream_execute(self, task: str) -> AsyncIterator[Event]:
        """Yield events while executing a task."""
        async for event in self._cli.execute_streaming(task, session_config=self._make_config()):
            yield event

    async def chat(self, session: Session | None = None) -> Session:
        """Return a Session pre-configured for this agent (for multi-turn use)."""
        if session is None:
            session = await Session.create(self._cli)
            session.config = self._make_config()
        return session

    # ── configuration ────────────────────────────────────────────────────────

    def with_config(
        self,
        *,
        model: str | None = None,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        bare: bool | None = None,
    ) -> "BaseAgent":
        """Return self after applying config overrides (fluent API)."""
        if model is not None:
            self._model = model
        if tools is not None:
            self._tools = tools
        if system_prompt is not None:
            self._system_prompt = system_prompt
        if bare is not None:
            self._bare = bare
        return self

    def _make_config(self) -> SessionConfig:
        tools = self._tools if self._tools is not None else self.get_default_tools()
        system = self._system_prompt or self.get_system_prompt() or None
        return SessionConfig(
            tools=tools or None,
            model=self._model,
            system_prompt=system,
            bare=self._bare,
        )

    def _make_result(self, response: Response) -> AgentResult:
        return AgentResult(response=response, summary=response.result[:200])
