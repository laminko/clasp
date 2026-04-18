from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..utils.errors import CLIError, TimeoutError
from ..utils.helpers import get_logger

logger = get_logger(__name__)


class ProcessManager:
    """Manages async subprocess lifecycle for the claude CLI."""

    def __init__(self, timeout: float | None = None) -> None:
        self.timeout = timeout  # seconds; None = no limit

    async def run(self, cmd: list[str], *, cwd: str | None = None) -> tuple[str, str, int]:
        """Run a command to completion, returning (stdout, stderr, exit_code)."""
        logger.debug("Running: %s", " ".join(cmd))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise TimeoutError(
                    f"Command timed out after {self.timeout}s"
                ) from exc

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            return stdout, stderr, proc.returncode or 0

        except FileNotFoundError as exc:
            raise CLIError(f"Binary not found: {cmd[0]}") from exc

    async def stream_lines(
        self, cmd: list[str], *, cwd: str | None = None
    ) -> AsyncIterator[str]:
        """Yield stdout lines from a command as they arrive."""
        logger.debug("Streaming: %s", " ".join(cmd))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            raise CLIError(f"Binary not found: {cmd[0]}") from exc

        assert proc.stdout is not None

        try:
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                yield line_bytes.decode("utf-8", errors="replace").rstrip("\n")
        finally:
            # Drain stderr so the process doesn't block
            if proc.stderr:
                stderr_bytes = await proc.stderr.read()
                if stderr_bytes:
                    logger.debug("stderr: %s", stderr_bytes.decode("utf-8", errors="replace"))
            await proc.wait()
            if proc.returncode and proc.returncode != 0:
                logger.debug("Process exited with code %d", proc.returncode)
