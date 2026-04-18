# Bidirectional JSON-RPC over stdio (ACP Protocol)

**Date:** 2026-04-18
**Status:** Implemented

## Context

The `cckit` SDK originally spawned a **new subprocess per message** (`--print --output-format stream-json`). This was fire-and-forget: send a prompt, consume stdout, process exits. There was no way to handle permissions, elicitation, or maintain a persistent connection.

This work adds a **bidirectional JSON-RPC 2.0 over stdio** path -- a single long-lived `claude` process exchanging NDJSON messages over stdin/stdout. The existing one-shot API (`CLI.execute()`, `Session.send()`) stays untouched. The new ACP path is purely additive.

### Sources & References

- **JSON-RPC 2.0 Specification** (jsonrpc.org) -- message format, error codes, request/response/notification semantics
- **Claude Code CLI documentation** -- `--input-format stream-json` and `--output-format stream-json` flags for persistent agent sessions
- **ACP (Agent Communication Protocol)** -- bidirectional agent-client communication pattern over JSON-RPC 2.0

## Architecture

```
+-------------------+
|  User code /      |
|  ACPSession       |  -- same Response/Event types as existing Session
+--------+----------+
         |
+--------v----------+
|   ACPClient       |  -- initialize, session/new, session/prompt, etc.
+--------+----------+
         |
+--------v----------+
|  RpcTransport     |  -- async subprocess, NDJSON stdin/stdout, request/response matching
+--------+----------+
         |
+--------v----------------------------------------------------+
|  claude process (long-lived, --input-format stream-json      |
|                  --output-format stream-json --verbose)       |
+--------------------------------------------------------------+
```

## Dependency Graph (implementation order)

```
rpc/protocol.py (JSON-RPC message types)
    |
    v
rpc/transport.py (bidirectional subprocess pipe)  <-- utils/errors.py (RpcError, TransportError)
    |
    v
rpc/client.py (ACP client methods)  <-- rpc/handlers.py (permission/file/elicitation handlers)
    |
    v
streaming/acp_parser.py (ACP notifications -> Event types)
    |
    v
session/acp_session.py (user-facing ACPSession)  <-- core/config.py (ACPConfig)
    |
    v
__init__.py (export new public API)
```

## New Files

| File | Purpose |
|------|---------|
| `cckit/rpc/__init__.py` | Package exports |
| `cckit/rpc/protocol.py` | JSON-RPC 2.0 message dataclasses (`JsonRpcRequest`, `JsonRpcResponse`, `JsonRpcNotification`, `JsonRpcError`) |
| `cckit/rpc/transport.py` | `RpcTransport` -- bidirectional subprocess NDJSON pipe with request/response matching |
| `cckit/rpc/handlers.py` | `DefaultHandlers` + `PermissionPolicy` -- handles agent->client callbacks (permissions, file I/O, elicitation) |
| `cckit/rpc/client.py` | `ACPClient` -- typed wrapper for ACP protocol methods (initialize, session/new, prompt, etc.) |
| `cckit/streaming/acp_parser.py` | `parse_session_update()` -- maps ACP notifications to existing Event types |
| `cckit/session/acp_session.py` | `ACPSession` -- user-facing class with `send()` / `stream()` matching Session API |

## Modified Files

| File | Change |
|------|--------|
| `cckit/utils/errors.py` | Added `TransportError`, `RpcError`, `ProtocolError` |
| `cckit/core/config.py` | Added `ACPConfig` dataclass |
| `cckit/__init__.py` | New public API exports |
| `cckit/core/__init__.py` | Added `ACPConfig` export |
| `cckit/session/__init__.py` | Added `ACPSession` export |
| `cckit/streaming/__init__.py` | Added `parse_session_update` export |
| `cckit/utils/__init__.py` | Added new error exports |

## Unchanged Files

The legacy one-shot path is completely untouched:
- `core/cli.py`, `core/process.py`, `core/command.py`
- `session/session.py`
- `agents/*`
- `streaming/events.py`, `streaming/parser.py`, `streaming/handler.py`

## Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_rpc_protocol.py` | 15 | Serialization, deserialization, roundtrip, error codes |
| `tests/test_rpc_transport.py` | 12 | Start/stop, echo requests, errors, timeouts, notifications, incoming requests |
| `tests/test_acp_client.py` | 12 | Lifecycle, handler registration, session updates, permission policies, file I/O |
| `tests/test_acp_parser.py` | 12 | All event type mappings, unknown types, raw field preservation |
| `tests/test_acp_session.py` | 4 | Response building from events |
| `tests/integration/test_acp_lifecycle.py` | 2 | Full lifecycle with real `claude` binary (skip if not found) |
| `tests/fixtures/echo_rpc.py` | -- | Fake JSON-RPC server for transport tests |

## Key Design Decisions

1. **Purely additive** -- No changes to existing Session/CLI code paths
2. **Reuse existing types** -- ACPSession returns the same `Response`, `Event`, `Usage` types as Session
3. **Plain dataclasses** -- Follows the project's existing style (no pydantic for internal types)
4. **Real subprocess tests** -- Transport tests use a real Python subprocess (`echo_rpc.py`), not mocks

## Verification

```bash
# All unit tests (95 pass, 0 failures)
uv run pytest tests/ -v --ignore=tests/integration

# Lint clean
uv run ruff check cckit/ tests/
uv run ruff format --check cckit/ tests/

# Integration test (requires claude binary)
uv run pytest tests/integration/ -v -m integration
```

## Usage Example

```python
from cckit import ACPSession

async with await ACPSession.create() as session:
    # Full response
    response = await session.send("What is 2+2?")
    print(response.result)

    # Streaming
    async for event in session.stream("Count to 3"):
        print(event)
```
