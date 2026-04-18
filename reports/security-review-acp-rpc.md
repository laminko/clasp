# Security Review — ACP / JSON-RPC Transport

**Branch:** `main`
**Scope:** Pending changes introducing `claude_agent/rpc/*`, `claude_agent/session/acp_session.py`, `claude_agent/streaming/acp_parser.py`, plus edits to `core/config.py`, `utils/errors.py`, and package re-exports.
**Reviewer:** Security review (automated, thorough)
**Date:** 2026-04-18

---

## Summary

The new ACP subsystem establishes a long-lived JSON-RPC 2.0 channel to the `claude` CLI subprocess and handles agent→client callbacks (permission requests, filesystem I/O, elicitation). The design is functionally sound, but the **default security posture is unsafe**: tool permissions are auto-approved, and the filesystem callbacks have no path confinement, size limits, or symlink protection. A malicious or prompt-injected subprocess — or a compromised Claude binary — can read or overwrite any file the host user can access, with no user visibility. Several additional hardening gaps compound the risk: unbounded line reads, a stderr pipe that is never drained (process-hang DoS), leaked exception strings, and a resource leak if `initialize()` fails after `transport.start()`.

**Findings:** 2 Critical, 5 High, 7 Medium, 6 Low.
**Recommendation:** Do not merge in current state. Address all Critical + High findings before shipping, or gate the feature behind an explicit opt-in that loudly signals the risk.

---

## Threat model

The transport treats the subprocess as **privileged** — it approves tool calls and services arbitrary filesystem read/write RPCs on its behalf. In practice the subprocess:

- Executes model output, which can be influenced by untrusted inputs (files read, URLs fetched, user prompts).
- May invoke sub-tools over which the client has no further audit.
- Is started from a path resolved through `~` and `$VAR` expansion plus symlink resolution.

So the subprocess should be treated as **semi-trusted at best**. Anything it asks the client to do needs the same scrutiny a local web server would apply to an unauthenticated HTTP API.

---

## Critical

### C1. Default permission policy auto-approves every tool call
**Files:** `claude_agent/rpc/handlers.py:39`, `claude_agent/session/acp_session.py:54,110`, `claude_agent/core/config.py:43`

`DefaultHandlers.__init__` defaults to `PermissionPolicy.AUTO_APPROVE`. `ACPSession.create()` and `ACPSession.connect()` both default to the same policy, and `ACPConfig.permission_policy = "auto_approve"`. Every `session/request_permission` call — including `Bash`, `Write`, network/MCP tools — returns `{"approved": True}` with no user confirmation and no audit surface beyond a debug log line.

The entire point of the permission prompt in the Claude CLI is to give the user a chance to stop destructive actions. This code silently defeats it.

**Impact:** Prompt injection against the model → arbitrary shell execution on the host, without user awareness.

**Fix:**
- Change the default to `PermissionPolicy.CALLBACK` **and require a callback** (raise at construction time if none provided), or default to `AUTO_DENY`.
- If a non-interactive auto-approve is genuinely required for tests, make it an explicit, named factory (e.g. `DefaultHandlers.for_testing()`) that clearly signals intent.
- At minimum, log the full `params` payload of every approved permission call at `INFO` (not `DEBUG`) so a reviewer can diff actions after the fact.

---

### C2. Unrestricted filesystem read/write via `fs/read_text_file` and `fs/write_text_file`
**Files:** `claude_agent/rpc/handlers.py:66-92`

```python
path = Path(file_path)                  # no normalization, no confinement
...
content = path.read_text(encoding="utf-8")
...
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(content, encoding="utf-8")
```

The path comes from the subprocess and is used verbatim. Consequences:

- **Read exfiltration**: subprocess can request `~/.ssh/id_rsa`, `~/.aws/credentials`, `~/.config/gh/hosts.yml`, `/etc/passwd`, shell history, etc. The contents are returned in the RPC response where the subprocess can then upload them via a tool call, embed them in its reply, or similar.
- **Write tampering**: subprocess can overwrite `~/.zshrc`, `~/.bashrc`, `~/.ssh/authorized_keys`, `~/.config/**`, or any project file outside the intended workspace. `mkdir(parents=True)` silently creates any missing directories.
- **Symlink attacks**: no `O_NOFOLLOW`-equivalent. A pre-planted symlink (e.g. a project file symlinked to `/etc/hosts`) gets followed.
- **Path traversal**: no check that the resolved path is inside a workspace root.
- **Unbounded read size**: `read_text()` buffers the whole file into memory — request `/dev/zero` or a large log file and the process OOMs (DoS).
- **No atomic write**: crash mid-write leaves a truncated file.

**Impact:** Full read of any file the process user can read; write to any file the process user can write. Combined with C1, this is effectively local code execution via shell rc files.

**Fix:**
- Require an explicit `workspace_root: Path` on `DefaultHandlers`. Reject any path whose `Path(p).resolve()` is not a descendant of `workspace_root.resolve()`.
- Open files with `os.open(..., O_NOFOLLOW)` or walk the path manually rejecting symlinks — `Path.resolve()` alone is not enough because it silently follows symlinks.
- Enforce a max read size (e.g. 10 MiB) using incremental `.read(size+1)` to detect overrun.
- Use atomic write: write to a sibling tempfile + `os.replace`.
- Reject any path segment that starts with `..` even after resolution.
- Log every read/write path at `INFO`.

---

## High

### H1. Resource leak on failed session initialization
**Files:** `claude_agent/session/acp_session.py:87-102`, `127-131`

```python
await transport.start()                 # subprocess spawned
await client.initialize(...)            # if this raises, subprocess is leaked
session_id = await client.new_session(...)
return cls(client, session_id)
```

If `initialize()` or `new_session()` raises (timeout, protocol error, binary bug), `transport.stop()` is never called. The subprocess is orphaned, holding stdin/stdout pipes, and `_pending` futures/reader task outlive the caller. Under test-flakiness or retry loops this becomes a process fork-bomb and an fd leak.

**Fix:** wrap in `try/except` and call `await transport.stop()` before re-raising.

---

### H2. Subprocess stderr is piped but never drained
**File:** `claude_agent/rpc/transport.py:43-48`

```python
self._proc = await asyncio.create_subprocess_exec(
    *self._cmd,
    stdin=..., stdout=..., stderr=asyncio.subprocess.PIPE,
)
```

`stderr=PIPE` creates an OS pipe that nothing reads. Once the pipe buffer fills (typically 64 KB on Linux, 16 KB on macOS) the child **blocks on stderr writes** — which deadlocks the entire session without any error surface. Claude's `--verbose` mode (hard-coded at `acp_session.py:76,120`) increases the likelihood of hitting this.

Stderr from the subprocess is also the primary place auth/config errors surface; silently discarding it makes these sessions very hard to debug and hides potentially security-relevant signals.

**Fix:** either use `stderr=asyncio.subprocess.DEVNULL` (accepts loss), inherit the parent's stderr, or spawn a second reader task that drains stderr to the logger.

---

### H3. Reader loop is serial — slow handlers stall the transport
**File:** `claude_agent/rpc/transport.py:150-254`

`_read_loop` awaits each `_on_message` call synchronously. If any request handler (`handle_file_read` on a large file, `handle_permission` with a blocking callback, `handle_file_write`) takes non-trivial time, the reader cannot process subsequent messages — including responses to our own pending requests. A malicious subprocess can issue one slow `fs/read_text_file` (e.g. on a FIFO or device) and freeze the transport; all other pending requests eventually hit the 30 s timeout.

**Fix:** dispatch handlers on `asyncio.create_task(...)` and await them independently; bound per-handler execution with `asyncio.wait_for`.

---

### H4. Exception strings leaked to the remote end
**Files:** `claude_agent/rpc/transport.py:229-234`, `claude_agent/rpc/handlers.py:74-78, 86-92`

```python
resp = JsonRpcResponse(id=msg_id, error=JsonRpcError(code=-32603, message=str(exc)))
...
return {"error": str(exc)}
```

`str(exc)` for Python exceptions usually contains absolute filesystem paths, environment variables embedded in paths, and sometimes file contents (e.g. `UnicodeDecodeError` can include surrounding bytes). These are sent back to the subprocess, which can relay them into model output or tool calls. This is both an information disclosure (paths, usernames, structure of `$HOME`) and a log-scraping hazard.

**Fix:** return a constant `message` like `"internal error"` on the wire; log the full exception locally only.

---

### H5. Reader loop swallows fatal errors silently
**File:** `claude_agent/rpc/transport.py:173-184`

`except Exception: logger.exception("Reader loop crashed")` leaves `_closed` unset until the `finally` clause runs, but the subprocess itself is not terminated. Callers with no pending future do not get any signal the transport is dead. `stop()` is now the user's responsibility and is never invoked automatically.

**Fix:** on reader-loop crash, also `terminate()` the subprocess and set an error flag that causes subsequent `request()` / `notify()` calls to raise `TransportError` immediately.

---

## Medium

### M1. JSON-RPC ID type is not validated on incoming responses
**File:** `claude_agent/rpc/transport.py:188,193`

`_pending` is keyed by `int`, but `data.get("id")` may be a JSON string, float, or null. If a server ever responds with `"id": "1"`, the lookup silently fails and the request eventually times out with no diagnostic. This also means a non-standard `id` can be used to exhaust `_pending` entries (pile-up of never-resolved futures — memory growth).

**Fix:** normalize/validate the incoming id type; drop or error on mismatched ids and log a warning.

---

### M2. Unbounded line length in reader
**File:** `claude_agent/rpc/transport.py:156`

`await self._proc.stdout.readline()` has no length limit. A hostile or malfunctioning subprocess can send a multi-GB line (e.g. binary file content base64-encoded) and the transport attempts to buffer it all. Asyncio's default `limit` is 64 KB and overflow raises `LimitOverrunError`, which is caught by the broad `Exception` handler and silently crashes the reader loop (see H5).

**Fix:** construct the subprocess with an explicit `limit` set to a reasonable value (e.g. 16 MiB) on the `StreamReader`, and handle `LimitOverrunError` by dropping the session with a clear `TransportError`.

---

### M3. `expand_path` resolves through symlinks silently
**Files:** `claude_agent/utils/helpers.py:9-11`, `claude_agent/session/acp_session.py:68,113`

```python
return str(Path(os.path.expandvars(os.path.expanduser(path))).resolve())
```

`Path.resolve()` follows symlinks. If an attacker can write `~/.local/bin/claude` (the default binary path) as a symlink to `/tmp/evil`, the attacker's binary runs with the user's privileges. Also, `os.path.expandvars` expands `$VAR` from the *current process* environment, so env vars set by a less-trusted parent (e.g. a CI runner) affect binary selection.

**Fix:** do not `resolve()` the binary path; use `Path.expanduser()` only. Optionally verify the binary is owned by the current user and not world-writable before spawning. Consider documenting that `binary_path` must come from trusted config.

---

### M4. `PermissionPolicy.CALLBACK` with no callback silently approves
**File:** `claude_agent/rpc/handlers.py:55-64`

```python
if (self.permission_policy == PermissionPolicy.CALLBACK
        and self._permission_callback):
    ...
return {"approved": True}
```

If `policy=CALLBACK` is set but `permission_callback=None`, control falls through to the final `return {"approved": True}` — i.e. auto-approve on misconfiguration. Classic fail-open.

**Fix:** raise `ValueError` at construction time if `CALLBACK` is used without a callback, and change the terminal fallback to `{"approved": False, "reason": "no callback configured"}`.

---

### M5. `ACPConfig.permission_policy` is a string, parsed unsafely
**File:** `claude_agent/session/acp_session.py:82`

```python
permission_policy=PermissionPolicy(cfg.permission_policy),
```

If a user loads `ACPConfig` from a YAML/JSON file and mistypes the policy (`"auto_aprove"`), `PermissionPolicy(...)` raises `ValueError` and leaks the invalid value. Worse, this dataclass field has no Enum validation at construction time, so the mis-configuration is not caught until a session is created.

**Fix:** make `ACPConfig.permission_policy` typed as `PermissionPolicy`, or validate in `__post_init__` with a fail-closed default of `AUTO_DENY`.

---

### M6. Subprocess inherits full parent environment
**File:** `claude_agent/rpc/transport.py:43-48`

`create_subprocess_exec` is called without `env=`, so the child inherits everything in `os.environ` — `ANTHROPIC_API_KEY`, `AWS_*`, `GITHUB_TOKEN`, etc. For the real `claude` binary this is expected, but any plausibly-compromised or swapped binary (see M3) instantly exfiltrates all of them. The blast radius of the binary-substitution attack is therefore "every secret in the user's environment."

**Fix:** when possible, pass a scrubbed env that contains only the variables the CLI actually needs (`HOME`, `PATH`, a curated allowlist). At minimum, document that the subprocess inherits the full environment.

---

### M7. Private attribute reach-through breaks encapsulation and can race
**File:** `claude_agent/session/acp_session.py:172-174`

```python
if on_update in self._client._session_update_callbacks:
    self._client._session_update_callbacks.remove(on_update)
```

`ACPSession` mutates `ACPClient`'s private list. If `_handle_session_update` is iterating this list (in another task) during `remove`, behavior is undefined. Also, because `stream()` registers the callback each call, two concurrent callers of `stream()` on the same session will receive each other's events.

**Fix:** expose a proper `off_session_update` API on `ACPClient`, and guard `stream()` against concurrent invocations (e.g. raise if a prompt is in-flight).

---

## Low / Informational

### L1. `handle_file_read` reads entire file into memory (`handlers.py:75`)
Without a size cap a large file will OOM the process. Already covered under C2; flagged here because even with path confinement, in-repo `.git` packfiles or test fixtures can trivially be >100 MiB.

### L2. `handle_file_write` is not atomic (`handlers.py:87-90`)
Partial write on crash. Use tempfile + `os.replace`.

### L3. Elicitation handler rejects, but via `{"error": ...}` not JSON-RPC error (`handlers.py:106`)
Protocol-level: the ACP spec treats RPC errors differently from result-level errors. Returning `{"error": "..."}` as a **result** means the subprocess cannot distinguish "declined" from "succeeded with that content." Harmless today but brittle.

### L4. `--verbose` is hard-coded (`acp_session.py:76,120`)
Verbose mode can emit tool inputs/outputs to logs. Make this opt-in via `ACPConfig.verbose`.

### L5. `TimeoutError` shadows built-in (`utils/errors.py:27`)
`# noqa: A001` suppresses the linter but any `except TimeoutError` in consumer code that expected `asyncio.TimeoutError` / `builtins.TimeoutError` will misbehave.

### L6. `safe_json_loads` return type annotation (`utils/helpers.py:25`)
Annotated `dict | None` but `json.loads` may return list/str/int/None. Unrelated to this PR but adjacent.

---

## Not issues (checked and cleared)

- **JSON-RPC id counter overflow**: Python `int` is unbounded. No wraparound.
- **Request race on `_next_id`**: asyncio is single-threaded, so the increment is atomic in the scheduler sense.
- **TLS / auth on the channel**: local stdio pipe owned by the same UID — no network exposure, no auth needed.
- **Shell injection in `cmd` list**: `create_subprocess_exec` uses `exec`, not a shell, and `cmd` is built from fixed strings plus a single binary path. No injection vector via `acp_session.py:70-77`. (`binary_path` itself is still sensitive — see M3.)

---

## Suggested remediation order

1. **C1, C2** — fix defaults and add path confinement. These are the blocking issues.
2. **H1, H4** — lifecycle cleanup + error-string scrubbing. Small diffs, large risk reduction.
3. **H2, H3, H5** — harden the transport reader loop against slow/hostile peers.
4. **M1–M7** — config and edge-case hardening.
5. **L1–L6** — polish.

## Test gaps worth filling

- A handler that writes outside `workspace_root` should raise.
- A handler on a symlinked path should refuse.
- A `LimitOverrunError` from an oversized line should surface as `TransportError`.
- An `initialize()` failure should cleanly terminate the subprocess (assert `returncode is not None`).
- A slow request handler should not block a concurrent `transport.request(...)` response.
- `CALLBACK` with no callback should raise at construction, not auto-approve.
