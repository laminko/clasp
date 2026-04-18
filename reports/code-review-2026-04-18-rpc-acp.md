# Code Review — RPC / ACP Additions

**Reviewer:** Senior engineer / architect pass
**Date:** 2026-04-18
**Branch:** `main` (working tree, un-committed)
**Scope:**

- New files: `cckit/rpc/{protocol,transport,client,handlers}.py`, `cckit/streaming/acp_parser.py`, `cckit/session/acp_session.py`
- Modified files: `cckit/utils/errors.py`, `cckit/core/config.py`, plus package `__init__.py` surfaces
- New tests: `tests/test_rpc_*.py`, `tests/test_acp_*.py`, `tests/integration/test_acp_lifecycle.py`, `tests/fixtures/echo_rpc.py`

**Overall verdict:** Direction is solid. Layering is clean (protocol → transport → client → session), type hints and dataclasses are consistent, and the echo-server fixture is a nice touch for transport testing. However, there are **two blocker-class issues** (stderr-drain deadlock in `RpcTransport`, and an unsandboxed filesystem handler reachable over RPC), a handful of correctness bugs, and several smaller polish items. Do **not** merge in current form — the blockers are latent failures that will bite in production.

---

## 1. Blockers (fix before merge)

### 1.1 `RpcTransport` never drains the subprocess's stderr → deadlock on chatty child

**File:** `cckit/rpc/transport.py:43-48`

```python
self._proc = await asyncio.create_subprocess_exec(
    *self._cmd,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
```

Stderr is PIPEd but nothing ever reads it. The reader loop only consumes `proc.stdout`. Once stderr's OS pipe buffer (~16–64 KiB on macOS / 64 KiB on Linux) fills, the child's `write(2)` on stderr blocks, and a long-lived Claude CLI with `--verbose` emits a lot on stderr. Result: the child silently stops making progress, our `request()` calls time out with no obvious cause.

The existing `ProcessManager.stream_lines` (`core/process.py:69-74`) gets this right — it drains stderr on the finally path. The new transport needs an equivalent, but **concurrently** with the stdout reader since the process is long-lived (can't wait until close).

**Recommendation:** spawn a second background task at `start()` that loops on `proc.stderr.readline()` and logs (or discards) each line. Cancel it in `stop()` alongside `_reader_task`. Either that, or switch stderr to `asyncio.subprocess.DEVNULL` if we genuinely don't care about the output. The current code picks the worst option: capture without drain.

Also: there is no test that exercises this. Add a fixture that writes a few MB to stderr and verify the transport keeps servicing requests.

### 1.2 Unsandboxed file I/O in `DefaultHandlers`

**File:** `cckit/rpc/handlers.py:66-92`

```python
async def handle_file_read(self, params: dict[str, Any]) -> dict[str, Any]:
    file_path = params.get("path", "")
    ...
    content = path.read_text(encoding="utf-8")
    return {"content": content}

async def handle_file_write(self, params: dict[str, Any]) -> dict[str, Any]:
    ...
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
```

The agent-side subprocess can ask the client to read or write **any** path on the host — `/etc/passwd`, `~/.ssh/id_rsa`, `~/.aws/credentials`, a user's Git config, etc. There is no allowlist, no path-traversal check, no tie-in to `handle_permission`, and the default `PermissionPolicy.AUTO_APPROVE` means even the permission hook is a rubber stamp. This is the classic "confused deputy": we act on behalf of a sub-process with the full authority of the caller's UID.

Even in the happy case where we trust the Claude binary, the capability exists and is exposed by a public class. Some threat-model surface:

- A compromised or substituted `claude` binary.
- A prompt-injection chain that causes the model to request reads/writes the user didn't intend.
- An MCP server with a bug that triggers odd `fs/*` calls.

**Recommendations (minimum):**

1. Constrain file paths to a configurable root (default: `SessionConfig.cwd` or `Path.cwd()`). Reject paths whose resolved form escapes the root.
2. Route `fs/read_text_file` and `fs/write_text_file` through `handle_permission` first, with the resolved absolute path in the prompt. Even `AUTO_APPROVE` users should get an audit log line per file touched.
3. Cap read size (e.g. 10 MB) to prevent OOM on a misbehaving agent.
4. Consider making the fs handlers opt-in rather than installed by default.

Ship with the defaults **safe**; power users can relax.

---

## 2. Correctness / protocol bugs

### 2.1 JSON-RPC `id` is over-narrowed to `int`

`JsonRpcRequest.id: int` and `RpcTransport._pending: dict[int, Future]` encode the assumption that every `id` is an integer. The JSON-RPC 2.0 spec explicitly allows **string**, **number**, or **null**. This works today because we are the only one minting request IDs for outbound calls, but:

- Incoming *requests* from the server may carry string IDs, and they round-trip through `_on_message` → `_send_response` unchanged — that half is fine.
- Responses from the server to our outbound calls are keyed on the id we sent, so this is internally consistent.
- Still, `JsonRpcResponse.from_dict` accepts `data.get("id")` as `Any`, then compares it to our int key in `self._pending.pop(msg_id, ...)`. If a server ever echoes the id as a string (some implementations do), the lookup silently fails and we'd hit the "Response for unknown id" debug line.

**Recommendation:** use `int | str` for the type and key `_pending` on `int | str`. The cost is trivial; the payoff is spec compliance.

### 2.2 `JsonRpcRequest.id` defaults to 0

`id: int = 0` (protocol.py:43). Zero is a valid, normal id. If a caller ever constructs `JsonRpcRequest(method="x")` without an id, it silently collides with whatever the transport later assigns. Today `RpcTransport.request()` explicitly overrides, so this is latent — but it's a foot-gun for anyone who uses `JsonRpcRequest` directly.

**Recommendation:** make `id: int | None = None` and raise in `to_dict()` / `to_line()` if unset. Or mark the dataclass frozen and require id at construction.

### 2.3 `ACPClient.prompt()` uses `request()` with a 30 s default timeout

**File:** `cckit/rpc/client.py:118-131`

```python
await self._transport.request(
    "session/prompt",
    {"sessionId": self._session_id, "message": message},
)
```

`RpcTransport.request` defaults `timeout=30.0`. Real prompts routinely run longer than 30 s (agents doing multi-step work easily exceed that). The docstring also says the *response* arrives via `session/update` notifications, implying the request-response of `session/prompt` is just an ack — but we don't pass an override.

Two issues rolled together:

1. Pick an explicit, generous timeout (or `None` / math.inf) for prompts.
2. Clarify whether the JSON-RPC *response* for `session/prompt` is the ack or the final result. The docstring says one thing; the 30 s default presumes the other. Whichever is right, make it match.

### 2.4 `_handle_session_update` drops async callbacks silently

**File:** `cckit/rpc/client.py:146-152`

```python
def _handle_session_update(self, params: dict[str, Any]) -> None:
    for cb in self._session_update_callbacks:
        try:
            cb(params)
        except Exception:
            logger.exception("session/update callback raised")
```

If a user registers an `async def` callback, `cb(params)` returns a coroutine that is **never awaited** — no-op and a `"coroutine was never awaited"` warning at GC time. `on_session_update` is typed `Callable[..., Any]` which accepts coroutine functions, so nothing stops this.

**Recommendation:** detect with `inspect.iscoroutinefunction`/`inspect.isawaitable` and schedule with `asyncio.create_task`, OR document and type-narrow to sync-only callbacks (`Callable[[dict], None]`).

### 2.5 Concurrent `ACPSession.stream()` calls interleave

**File:** `cckit/session/acp_session.py:147-174`

Every call to `stream()` registers a callback that feeds **its own** queue, but the `ACPClient._session_update_callbacks` list is shared session-global. If two coroutines call `stream()` concurrently (e.g., an async pipeline that kicks off a second prompt before the first has drained), both queues receive every event, and the first `type == "result"` terminates **both** iterators regardless of which prompt produced it.

ACP likely doesn't allow parallel prompts on a single session anyway — but right now there's nothing enforcing that invariant.

**Recommendation:** either

- Gate `prompt()` / `stream()` with an `asyncio.Lock` on the session (one outstanding prompt at a time), raising a clear error on contention, or
- Correlate `session/update` notifications to the originating prompt (if the protocol has a correlation id) and filter per-stream.

Also: the `finally` block reaches into `self._client._session_update_callbacks` by name — private-member access. Expose `remove_session_update_callback()` on `ACPClient` so `ACPSession` doesn't rely on attribute stability.

### 2.6 `_events_to_response` never populates `Usage`

**File:** `cckit/session/acp_session.py:204-229`

```python
usage = Usage()
for event in events:
    if isinstance(event, TextChunkEvent): ...
    elif isinstance(event, ResultEvent): ...
```

`UsageEvent` is produced by `parse_session_update` but never consumed here. The returned `Response.usage` is always zero for ACP sessions, even though the data is right there in the event stream. That's a regression vs. the one-shot `Session` path, which at least can infer usage from the final result.

**Recommendation:** add an `isinstance(event, UsageEvent)` branch that accumulates into `usage`.

### 2.7 `parse_session_update`: speculative schema, stringified content

**File:** `cckit/streaming/acp_parser.py`

Two concerns:

- The comment says the mapping is "based on the Claude Code ACP notification subtypes documented in the Claude Code CLI `--output-format stream-json` specification". But the subtype names (`content_delta`, `tool_call_started`, `assistant_item_started`, `assistant_item_completed`) **don't match** what `parser.py` emits for the same output format (`content_block_delta`, `message_start`, `message_stop`, `message_delta`, `assistant`, etc.). One of these is wrong. If `parse_session_update` is targeting the newer ACP notification schema, the integration test `test_create_prompt_close` would be where it proves itself — but that test only checks that `"4"` appears in the final result, which would pass even if every `session/update` was silently dropped (the `ResultEvent` carries the final text).
- `content=str(params.get("content", ""))` (line 67). `content` is often a list of content blocks in the stream-json / ACP format; stringifying a list yields `"[{'type': 'text', 'text': '...'}]"` which is garbage downstream. `parser.py:60-67` handles this case correctly; the ACP parser should share that code.

**Recommendation:** verify the ACP subtype names against the actual CLI output (capture one live session and save to `tests/fixtures/`), fix schema mismatches, and factor the content-block-flattening helper out of `parser.py` for reuse.

### 2.8 `DefaultHandlers`: error responses look like success to the RPC layer

**File:** `cckit/rpc/handlers.py:66-92`

Handlers return `{"error": "..."}` on failure. But in the transport, the return value is wrapped into a JSON-RPC `result`, not an `error`. The remote therefore sees a successful response whose body happens to contain an `error` field — it will **not** raise on the other side, and callers have to manually check. This is a protocol violation if the ACP contract expects proper JSON-RPC error shape.

**Recommendation:** when a handler encounters an error it wants to surface, raise an exception. `_on_message` already converts exceptions into JSON-RPC `error` responses with code `-32603` (lines 229-234), which is the correct semantics. Reserve dict returns for actual success payloads.

### 2.9 Default fall-through in `handle_permission` approves

**File:** `cckit/rpc/handlers.py:45-64`

If `permission_policy=CALLBACK` but no callback is set, control falls through to `return {"approved": True}`. Denying by default would be safer — the current behavior converts a configuration error into silent auto-approval.

### 2.10 `TimeoutError` shadows the builtin without inheriting

**File:** `cckit/utils/errors.py:27-29`

```python
class TimeoutError(CckitError):  # noqa: A001
    ...
```

The `# noqa: A001` suppresses the flake8 warning, but a user writing `except TimeoutError:` at module scope could get either the builtin or ours depending on import order and namespacing. Since the transport also catches `asyncio.TimeoutError` (a `builtins.TimeoutError` subclass since 3.11), confusion is guaranteed.

**Recommendation:** rename to `CLITimeoutError` / `AgentTimeoutError`, OR inherit from `builtins.TimeoutError` so `except TimeoutError` catches both. The `# noqa` is treating the symptom, not the cause.

### 2.11 `ACPConfig.permission_policy` is `str`, not the enum

**File:** `cckit/core/config.py:43`

```python
permission_policy: str = "auto_approve"
```

Then `ACPSession.create` does `PermissionPolicy(cfg.permission_policy)` which raises `ValueError` for any typo. Use the enum in the dataclass — the whole point of having `PermissionPolicy` is to make illegal values unrepresentable.

---

## 3. Design / architecture

### 3.1 Two parsers, overlapping responsibilities

`streaming/parser.py` (existing) targets the stream-json shape from Anthropic's Messages API tunneled through `claude --output-format stream-json`. `streaming/acp_parser.py` (new) targets `session/update` notification params from the same CLI in a different invocation mode. Both emit the same `Event` union.

Problems:

- The two schemas diverge in ways (content-block flattening, tool-call id naming) that suggest code duplication drift over time.
- There is no shared contract tested against a real CLI, so the two can silently disagree on the same underlying event.

**Recommendation:** factor a `ContentBlockParser` (or similar) that both parsers call into. Capture real CLI output fixtures (both one-shot stream-json and ACP) under `tests/fixtures/` and round-trip them through the parsers in tests. That single exercise will flush out the schema questions I flagged in §2.7.

### 3.2 `Session` vs. `ACPSession` drift

Both have `send()`, `stream()`, `_events_to_response`-style logic, factory methods, context-manager support. They share no base class and no common interface. As soon as agents start accepting "a session" polymorphically, the shapes will diverge.

**Recommendation:** define a `Protocol` (PEP 544) such as `SupportsPrompt` with `send` / `stream` and make both `Session` and `ACPSession` conform. Consumers (agents, utilities) can then depend on the abstraction. Low cost, preserves both implementations.

### 3.3 `ACPSession.create()` and `connect()` duplicate setup

Command construction and transport/client wiring are copy-pasted. Extract a private `_bootstrap(cmd, handlers) -> ACPClient` helper.

### 3.4 `ACPConfig` vs. argument overloading

`ACPSession.create(..., config=...)` accepts a full config *or* individual args, with inconsistent preference:

```python
bp = expand_path(cfg.binary_path if config else binary_path)
handlers = DefaultHandlers(
    permission_policy=permission_policy
    if not config
    else PermissionPolicy(cfg.permission_policy),
)
```

This "which one wins" logic is easy to get wrong and already reads oddly. Pick one shape:

- **Option A:** `create(config: ACPConfig)` only; no individual kwargs.
- **Option B:** individual kwargs only; no `config` param.
- **Option C:** kwargs override config, uniformly, driven by `dataclasses.replace`.

Right now it's a half-and-half that invites bugs.

### 3.5 Always passing `--verbose`

`ACPSession` unconditionally adds `--verbose` (acp_session.py:77, 121). Combined with the undrained stderr (§1.1), this is the compounding failure mode: verbose output → full stderr buffer → deadlock. Verify this flag is actually required for stream-json output in ACP mode; if not, drop it. If yes, all the more reason to drain stderr.

### 3.6 Minor: hardcoded error codes in `transport._on_message`

Lines 218 and 233 hardcode `-32601` / `-32603` instead of importing `METHOD_NOT_FOUND` / `INTERNAL_ERROR` from `protocol`. Cosmetic, but the constants exist for a reason.

---

## 4. Tests

### Coverage

- **`test_rpc_protocol.py`:** solid, covers serialization roundtrips. ✅
- **`test_rpc_transport.py`:** good — uses a real subprocess fixture, exercises lifecycle, timeout, error response, notification, incoming-request callback, and sequential requests. ✅
- **`test_acp_client.py`:** OK, but most tests call private methods (`_handle_session_update`, `transport._request_handlers`) — they'd pass even if the public wiring broke. Add a test that sends a fake `session/update` line through the real transport stdin and verifies the callback fires end-to-end.
- **`test_acp_parser.py`:** fine, but only verifies the shapes *I* picked for the subtypes, not the ones the CLI actually emits. See §2.7.
- **`test_acp_session.py`:** only covers the pure-function `_events_to_response`. The whole streaming/queue/cancel path is untested at the unit level. This is a significant gap given the concurrency bug in §2.5.
- **`test_acp_lifecycle.py`:** skipped if `claude` isn't on PATH; two scenarios (prompt, streaming). Integration-test-as-smoke-test is fine but we should also assert specific events appear in the stream to catch schema drift, not just `len(events) > 0`.

### Missing scenarios

- Concurrent outbound `request()`s (ordering, interleaving).
- stderr-flooding child process (the §1.1 deadlock).
- Reader-loop crash recovery.
- Subprocess exits mid-request (pending futures get rejected — partially covered by `test_request_after_stop_raises` but not for "process dies on its own").
- Large response payloads (line buffering, multi-KB JSON).
- Path-traversal attempts on `handle_file_read` / `handle_file_write`.
- `PermissionPolicy.CALLBACK` with an `async def` callback (the `hasattr(result, "__await__")` path).

### Style

- Private attribute access (`transport._proc`, `client._handle_session_update`) is pervasive. Fine for a small codebase, but tighten once the API stabilises.
- `asyncio.sleep(0.1)` in `test_receive_notification` is a race-y timing hack. An `asyncio.Event` signalled from the handler would be deterministic.

---

## 5. Smaller issues & nits

| Location | Issue |
|---|---|
| `client.py:125` | `raise RuntimeError(...)` — use `SessionError` from `utils.errors` for consistency. |
| `client.py:171-174` | `__aexit__` catches bare `Exception` on `close_session` — at minimum log it. |
| `handlers.py:41` | `permission_callback: Any | None` — type as `Callable[[dict], dict \| Awaitable[dict]] \| None`. |
| `handlers.py:60` | `hasattr(result, "__await__")` — prefer `inspect.isawaitable(result)`. |
| `transport.py:140-146` | `on_request` / `on_notification` silently overwrite existing handlers. Either reject duplicates or log a warning. |
| `transport.py:39-53` | No guard against calling `start()` twice — leaks the first subprocess. |
| `acp_session.py:186-188` | `close()` catches bare `Exception` on `close_session` — swallows. |
| `acp_session.py:196-200` | `__aenter__` assumes the transport was already started by the factory. A user who instantiates `ACPSession(...)` directly gets a silent no-op enter. Mark `__init__` as internal-use-only (or move setup into `__aenter__`). |
| `errors.py` | `ProtocolError` is defined and exported but I don't see it raised anywhere. Either use it or drop it. |
| `protocol.py:47-51` | `to_dict` always includes `params` — the spec allows omission. Minor. |
| `config.py:47-48` | `request_timeout` / `shutdown_timeout` defined but `ACPSession` doesn't plumb them through to `RpcTransport.request` / `stop`. |
| `acp_parser.py` | `tool_call_started` and `tool_call_updated` produce identical `ToolUseEvent`s — the "updated" signal is lost. If progressive tool input matters, differentiate. |

---

## 6. What to do next (suggested order)

1. **Fix §1.1 (stderr drain)** — two-line addition, big reliability win.
2. **Fix §1.2 (fs sandbox)** — lock down before anyone starts using this for real.
3. **Resolve §2.7 (schema verification)** — capture real CLI output, turn into fixtures, delete the parts of `acp_parser.py` that turn out to be wrong. Until this is done, the ACP path has no ground truth.
4. **Plumb usage through `_events_to_response`** (§2.6) and fix the default-approve fall-through (§2.9). Small, obvious wins.
5. **Add the "concurrent `stream()`" guard or protocol correlation** (§2.5) — this is easy to ignore until a user hits it and hard to debug when they do.
6. **Protocol / type cleanups** (ids, enum vs. string, `TimeoutError` naming) — bundle as one PR.
7. **Introduce a `SupportsPrompt` Protocol** (§3.2) so agents and utilities can consume both session kinds uniformly.

Happy to pair on any of these if useful.
