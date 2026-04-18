from __future__ import annotations

from ..core.cli import ClaudeCLI
from .base import BaseAgent

_RESEARCH_TOOLS = ["Read", "Grep", "Glob", "WebSearch", "WebFetch"]

_RESEARCH_SYSTEM_PROMPT = (
    "You are a thorough research assistant. "
    "Gather information from multiple sources, synthesise key findings, "
    "and present clear, well-structured summaries with supporting evidence."
)


class ResearchAgent(BaseAgent):
    """Agent pre-configured for research and analysis tasks."""

    def __init__(
        self,
        cli: ClaudeCLI | None = None,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
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

    def get_default_tools(self) -> list[str]:
        return _RESEARCH_TOOLS

    def get_system_prompt(self) -> str:
        return _RESEARCH_SYSTEM_PROMPT
