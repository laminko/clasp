from __future__ import annotations

from ..core.cli import ClaudeCLI
from .base import BaseAgent

_CODE_TOOLS = ["Read", "Edit", "Write", "MultiEdit", "Bash", "Grep", "Glob"]

_CODE_SYSTEM_PROMPT = (
    "You are an expert software engineer. "
    "Focus on correctness, clarity, and idiomatic code. "
    "When making changes, explain your reasoning briefly."
)


class CodeAgent(BaseAgent):
    """Agent pre-configured for coding tasks."""

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
        return _CODE_TOOLS

    def get_system_prompt(self) -> str:
        return _CODE_SYSTEM_PROMPT
