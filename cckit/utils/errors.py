from typing import Any


class CckitError(Exception):
    """Base exception for all cckit errors."""


class CLIError(CckitError):
    """Raised when the claude CLI process returns a non-zero exit code or stderr."""

    def __init__(
        self, message: str, exit_code: int | None = None, stderr: str = ""
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class AuthError(CLIError):
    """Raised when authentication fails."""


class SessionError(CckitError):
    """Raised on session lifecycle errors."""


class TimeoutError(CckitError):  # noqa: A001
    """Raised when a CLI call exceeds the configured timeout."""


class ParseError(CckitError):
    """Raised when stream-json output cannot be parsed."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


class TransportError(CckitError):
    """Raised when the RPC transport encounters a connection/pipe error."""


class RpcError(CckitError):
    """Raised when the remote end returns a JSON-RPC error response."""

    def __init__(self, message: str, code: int = 0, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class ProtocolError(CckitError):
    """Raised on JSON-RPC protocol violations (malformed messages, unexpected state)."""
