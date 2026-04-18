from __future__ import annotations

from ..core.cli import ClaudeCLI
from .base import BaseAgent


class CustomAgent(BaseAgent):
    """A fully user-defined agent with arbitrary tools and system prompt."""

    def __init__(
        self,
        name: str = "CustomAgent",
        *,
        cli: ClaudeCLI | None = None,
        system_prompt: str = "",
        tools: list[str] | None = None,
        model: str | None = None,
        bare: bool = True,
        binary_path: str = "~/.local/bin/claude",
    ) -> None:
        super().__init__(
            cli,
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            bare=bare,
            binary_path=binary_path,
        )
        self.name = name

    def get_default_tools(self) -> list[str]:
        return []

    def get_system_prompt(self) -> str:
        return self._system_prompt or ""
