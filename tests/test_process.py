"""Tests for ProcessManager (async subprocess wrapper)."""
from __future__ import annotations

import sys

import pytest

from cckit.core.process import ProcessManager
from cckit.utils.errors import CLIError, TimeoutError


class TestProcessRun:
    @pytest.mark.asyncio
    async def test_run_captures_stdout(self) -> None:
        pm = ProcessManager()
        stdout, stderr, code = await pm.run(
            [sys.executable, "-c", "print('hello')"]
        )
        assert "hello" in stdout
        assert code == 0

    @pytest.mark.asyncio
    async def test_run_captures_stderr(self) -> None:
        pm = ProcessManager()
        stdout, stderr, code = await pm.run(
            [sys.executable, "-c", "import sys; sys.stderr.write('oops')"]
        )
        assert "oops" in stderr
        assert code == 0

    @pytest.mark.asyncio
    async def test_run_nonzero_exit(self) -> None:
        pm = ProcessManager()
        _, _, code = await pm.run(
            [sys.executable, "-c", "import sys; sys.exit(3)"]
        )
        assert code == 3

    @pytest.mark.asyncio
    async def test_run_binary_not_found(self) -> None:
        pm = ProcessManager()
        with pytest.raises(CLIError, match="Binary not found"):
            await pm.run(["/nonexistent/binary/xyz"])

    @pytest.mark.asyncio
    async def test_run_timeout(self) -> None:
        pm = ProcessManager(timeout=0.2)
        with pytest.raises(TimeoutError, match="timed out"):
            await pm.run(
                [sys.executable, "-c", "import time; time.sleep(5)"]
            )


class TestProcessStreamLines:
    @pytest.mark.asyncio
    async def test_stream_lines_yields_lines(self) -> None:
        pm = ProcessManager()
        lines = []
        async for line in pm.stream_lines(
            [sys.executable, "-c", "print('a'); print('b'); print('c')"]
        ):
            lines.append(line)
        assert lines == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_stream_lines_binary_not_found(self) -> None:
        pm = ProcessManager()
        with pytest.raises(CLIError, match="Binary not found"):
            async for _ in pm.stream_lines(["/nonexistent/binary/xyz"]):
                pass

    @pytest.mark.asyncio
    async def test_stream_lines_drains_stderr(self) -> None:
        pm = ProcessManager()
        code = (
            "import sys; "
            "print('line1'); "
            "sys.stderr.write('err1\\n'); "
            "print('line2')"
        )
        lines = []
        async for line in pm.stream_lines([sys.executable, "-c", code]):
            lines.append(line)
        assert "line1" in lines
        assert "line2" in lines

    @pytest.mark.asyncio
    async def test_stream_lines_nonzero_exit(self) -> None:
        pm = ProcessManager()
        code = "import sys; print('out'); sys.exit(2)"
        lines = []
        async for line in pm.stream_lines([sys.executable, "-c", code]):
            lines.append(line)
        assert "out" in lines

    @pytest.mark.asyncio
    async def test_stream_lines_stdin_piped_when_provided(self) -> None:
        """When stdin= is supplied, the child reads it on its stdin pipe."""
        pm = ProcessManager()
        # Child echoes its stdin back, one line at a time.
        code = "import sys\nfor line in sys.stdin:\n    sys.stdout.write('got:' + line)"
        lines = []
        async for line in pm.stream_lines(
            [sys.executable, "-c", code], stdin=b"alpha\nbeta\n"
        ):
            lines.append(line)
        assert lines == ["got:alpha", "got:beta"]

    @pytest.mark.asyncio
    async def test_stream_lines_no_stdin_when_not_provided(self) -> None:
        """When stdin=None (default), reading from stdin in the child sees EOF immediately
        (because the parent's stdin is inherited and is typically not a pipe in this test
        process). Confirms we don't accidentally open a pipe when not asked to."""
        pm = ProcessManager()
        # Child prints 'before', tries to read stdin (gets EOF or whatever parent supplies),
        # then prints 'after'. We assert both 'before' and 'after' appear, meaning the read
        # didn't hang waiting for our test process to feed bytes.
        code = (
            "import sys\n"
            "print('before')\n"
            "try:\n"
            "    sys.stdin.read()\n"
            "except Exception:\n"
            "    pass\n"
            "print('after')\n"
        )
        lines = []
        async for line in pm.stream_lines([sys.executable, "-c", code]):
            lines.append(line)
        assert "before" in lines
        assert "after" in lines
