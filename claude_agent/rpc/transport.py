from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from ..utils.errors import CLIError, RpcError, TransportError
from ..utils.helpers import get_logger
from .protocol import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    JsonRpcError,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
)

logger = get_logger(__name__)


class RpcTransport:
    """Bidirectional JSON-RPC 2.0 transport over a subprocess's stdin/stdout.

    Manages a long-lived child process, sending NDJSON requests via stdin
    and reading NDJSON responses/notifications from stdout.
    """

    def __init__(self, cmd: list[str]) -> None:
        self._cmd = cmd
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id: int = 1
        self._request_handlers: dict[str, Callable[..., Any]] = {}
        self._notification_handlers: dict[str, Callable[..., Any]] = {}
        self._closed = False
        self._started = False

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn the subprocess and start the reader loop."""
        if self._started and not self._closed:
            raise TransportError("Transport already started")

        logger.debug("Starting transport: %s", " ".join(self._cmd))
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise CLIError(f"Binary not found: {self._cmd[0]}") from exc

        self._closed = False
        self._started = True
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._stderr_drain_loop())

    async def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop the subprocess."""
        if self._closed:
            return
        self._closed = True

        # Cancel background tasks
        for task in (self._reader_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()

        # Reject any pending requests
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(TransportError("Transport closed"))
        self._pending.clear()

    # ── sending ──────────────────────────────────────────────────────────

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
    ) -> Any:
        """Send a JSON-RPC request and await the response."""
        if self._closed or not self._proc or not self._proc.stdin:
            raise TransportError("Transport is not connected")

        req_id = self._next_id
        self._next_id += 1

        req = JsonRpcRequest(method=method, params=params or {}, id=req_id)
        line = req.to_line().encode("utf-8")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = fut

        logger.debug("→ request id=%d method=%s", req_id, method)
        try:
            self._proc.stdin.write(line)
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            self._pending.pop(req_id, None)
            raise TransportError("Broken pipe writing to subprocess") from exc

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(req_id, None)
            raise TransportError(
                f"Request {method} timed out after {timeout}s"
            ) from exc

    async def notify(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (fire-and-forget, no response expected)."""
        if self._closed or not self._proc or not self._proc.stdin:
            raise TransportError("Transport is not connected")

        notif = JsonRpcNotification(method=method, params=params or {})
        line = notif.to_line().encode("utf-8")

        logger.debug("→ notification method=%s", method)
        try:
            self._proc.stdin.write(line)
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise TransportError("Broken pipe writing to subprocess") from exc

    # ── handler registration ─────────────────────────────────────────────

    def on_request(self, method: str, handler: Callable[..., Any]) -> None:
        """Register a handler for incoming requests from the subprocess."""
        self._request_handlers[method] = handler

    def on_notification(self, method: str, handler: Callable[..., Any]) -> None:
        """Register a handler for incoming notifications from the subprocess."""
        self._notification_handlers[method] = handler

    # ── stderr drain (H2) ────────────────────────────────────────────────

    async def _stderr_drain_loop(self) -> None:
        """Background task that drains stderr to prevent pipe buffer deadlock."""
        assert self._proc and self._proc.stderr

        try:
            while not self._closed:
                line_bytes = await self._proc.stderr.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.debug("subprocess stderr: %s", line[:500])
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug("stderr drain loop ended")

    # ── reader loop ──────────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        """Background task that reads lines from stdout and dispatches them."""
        assert self._proc and self._proc.stdout

        try:
            while not self._closed:
                line_bytes = await self._proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug(
                        "Non-JSON line from subprocess: %s", line[:200]
                    )
                    continue

                await self._on_message(data)

        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Reader loop crashed")
            # H5: terminate the subprocess on reader crash
            if self._proc and self._proc.returncode is None:
                self._proc.terminate()
        finally:
            # Process ended — reject remaining pending requests
            if not self._closed:
                self._closed = True
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(TransportError("Subprocess exited"))
                self._pending.clear()

    async def _on_message(self, data: dict[str, Any]) -> None:
        """Route an incoming JSON message to the appropriate handler."""
        msg_id = data.get("id")
        method = data.get("method")

        # Response to our request: has id + (result or error), no method
        if msg_id is not None and method is None:
            fut = self._pending.pop(msg_id, None)
            if fut is None:
                logger.debug("Response for unknown id=%s", msg_id)
                return
            resp = JsonRpcResponse.from_dict(data)
            if resp.error:
                fut.set_exception(
                    RpcError(
                        resp.error.message,
                        code=resp.error.code,
                        data=resp.error.data,
                    )
                )
            else:
                fut.set_result(resp.result)
            return

        # Incoming request from subprocess: has id + method
        if msg_id is not None and method is not None:
            # H3: dispatch handler as a task so slow handlers don't stall
            asyncio.create_task(
                self._handle_incoming_request(msg_id, method, data)
            )
            return

        # Notification: has method, no id
        if method is not None:
            handler = self._notification_handlers.get(method)
            if handler:
                try:
                    result = handler(data.get("params", {}))
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception(
                        "Notification handler for %s raised", method
                    )
            else:
                logger.debug("Unhandled notification: %s", method)
            return

        logger.debug("Unrecognized message: %s", str(data)[:200])

    async def _handle_incoming_request(
        self, msg_id: Any, method: str, data: dict[str, Any]
    ) -> None:
        """Handle an incoming request from the subprocess in its own task."""
        handler = self._request_handlers.get(method)
        if handler is None:
            err_resp = JsonRpcResponse(
                id=msg_id,
                error=JsonRpcError(
                    code=METHOD_NOT_FOUND,
                    message=f"Method not found: {method}",
                ),
            )
            await self._send_response(err_resp)
            return

        try:
            result = handler(data.get("params", {}))
            if asyncio.iscoroutine(result):
                result = await result
            resp = JsonRpcResponse(id=msg_id, result=result)
        except Exception:
            # H4: don't leak exception details to the remote end
            logger.exception("Handler for %s raised", method)
            resp = JsonRpcResponse(
                id=msg_id,
                error=JsonRpcError(
                    code=INTERNAL_ERROR, message="Internal handler error"
                ),
            )
        await self._send_response(resp)

    async def _send_response(self, resp: JsonRpcResponse) -> None:
        """Write a response back to the subprocess via stdin."""
        if self._closed or not self._proc or not self._proc.stdin:
            return
        line = resp.to_line().encode("utf-8")
        try:
            self._proc.stdin.write(line)
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Could not send response — pipe broken")
