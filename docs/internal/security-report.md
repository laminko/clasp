# Security Review Report

**Date:** 2026-04-15
**Reviewer:** Claude Security Agent
**Repository:** `cckit`

---

## Executive Summary

The `cckit` repository is a well-structured, early-stage Python framework (v0.1.0) that wraps the Claude Code CLI binary. The codebase is relatively small, clean, and free of the most critical classes of vulnerability (no hardcoded secrets, no `eval`/`exec`, no shell injection via `shell=True`). All subprocess execution uses `asyncio.create_subprocess_exec` with argument lists, which is the correct approach.

However, several medium-to-high severity issues were identified, primarily centered on:

1. **Insufficient input validation** for values that flow directly into CLI arguments (prompt injection, path traversal through `expand_path`, unsanitized `extra_flags`, and unvalidated `session_id`).
2. **Overly permissive default agent configurations** (`PermissionMode.BYPASS` exposed, `CodeAgent` grants `Bash` access, `ResearchAgent` grants `WebSearch`/`WebFetch` without restriction).
3. **Insecure temporary file handling** for MCP configuration (world-readable file containing server commands and env vars, no cleanup guarantee).
4. **Sensitive data logged at DEBUG level** (full command lines containing prompts and system prompts, stderr output).
5. **Missing security controls** such as no prompt-length limits, no sanitization of LLM-returned `session_id`, and deserialization of arbitrary JSON from a file in `MessageHistory.load()`.

No critical findings (hardcoded secrets, shell injection, unsafe deserialization of binary data) were identified. The overall risk posture is **Medium**.

---

## Findings

### [HIGH] Unsanitized `extra_flags` Allows Arbitrary CLI Argument Injection

- **File:** `cckit/core/config.py:30`, `cckit/core/cli.py:138-139`
- **Description:** `CLIConfig.extra_flags` is a `list[str]` that is appended verbatim to every built command via `builder.add_flag(flag)`. There is no validation of what flags may be supplied. A caller (or any code path that constructs a `CLIConfig`) can inject any `claude` CLI flag, including `--permission-mode bypassPermissions`, `--dangerouslySkipPermissions`, `--add-dir /`, or `--output-format` overrides. This is a trust-boundary issue: if `CLIConfig` is ever constructed from user-supplied data (e.g., a config file, API request, or environment variable), an attacker could escalate the effective permissions of the agent without the application author intending it.
- **Recommendation:** Maintain an allowlist of permitted extra flags, or validate each flag against a set of known-safe flags before appending. At minimum, document that `extra_flags` is a privileged, internal-only field and must never accept external input without validation.

---

### [HIGH] Prompt Content Logged at DEBUG Level (Sensitive Data Exposure)

- **File:** `cckit/core/process.py:21`, `cckit/core/process.py:50`
- **Description:** Both `ProcessManager.run()` and `ProcessManager.stream_lines()` log the full command at `DEBUG` level using `logger.debug("Running: %s", " ".join(cmd))` and `logger.debug("Streaming: %s", " ".join(cmd))`. The command always includes the user's prompt (as the value of `--print`), and may include `--system-prompt` and `--append-system-prompt` content. If the application configures `DEBUG` logging (which is common during development or in verbose mode), the full text of every prompt and system prompt is written to logs. Depending on the deployment context (log aggregation systems, container stdout, shared log files), this can expose sensitive user queries or proprietary system prompts.
- **Recommendation:** Redact the prompt argument when logging. Log only the binary name and non-sensitive flags (e.g., `--model`, `--output-format`), or log at `TRACE`/`DEBUG` only the command length, not its full content. Consider a `sanitize_command_for_log()` helper that replaces prompt and system-prompt values with `<redacted>`.

---

### [HIGH] `PermissionMode.BYPASS` Exposed as a First-Class Enum Value

- **File:** `cckit/types/enums.py:14`
- **Description:** The `PermissionMode` enum includes `BYPASS = "bypassPermissions"`. This value, when passed to the `claude` CLI, instructs it to skip all permission checks—allowing the agent to read, write, execute, and modify any file or run any command on the system without prompting. It is a first-class, easily discoverable option in the public API (`from cckit import PermissionMode; PermissionMode.BYPASS`). There is no guardrail, warning, confirmation step, or documentation warning attached to this value. Any agent or example code that mistakenly sets this mode (e.g., a copy-paste error) would silently grant the LLM full system access.
- **Recommendation:** Add a prominent docstring warning to the `BYPASS` enum member. Consider requiring an explicit opt-in mechanism (e.g., a separate method `with_bypass_permissions(confirm=True)` on `CommandBuilder` that raises unless `confirm=True` is passed) to make accidental use obvious. The enum value itself should remain for completeness, but its danger must be surfaced.

---

### [MEDIUM] `CodeAgent` Grants Unrestricted `Bash` Tool by Default

- **File:** `cckit/agents/code.py:6`
- **Description:** `_CODE_TOOLS = ["Read", "Edit", "Write", "MultiEdit", "Bash", "Grep", "Glob"]`. The `Bash` tool allows the LLM to execute arbitrary shell commands on the host system. Granting this by default to all `CodeAgent` instances means any task executed through a `CodeAgent`—including tasks constructed from user input—runs with the LLM having shell access. Combined with a prompt injection vulnerability (see Finding below), this creates a path to arbitrary command execution on the host.
- **Recommendation:** Remove `Bash` from the default tool set, or split into two agent modes: a read-only `CodeAgent` and a `FullCodeAgent` that requires an explicit opt-in. At minimum, document clearly that `CodeAgent` grants shell execution and should only be used with fully trusted, non-user-supplied task inputs.

---

### [MEDIUM] `ResearchAgent` Grants Unrestricted `WebSearch` and `WebFetch` by Default

- **File:** `cckit/agents/research.py:6`
- **Description:** `_RESEARCH_TOOLS = ["Read", "Grep", "Glob", "WebSearch", "WebFetch"]`. The `WebFetch` tool allows the LLM to make arbitrary HTTP requests to any URL. If user-supplied input is incorporated into a research task, an attacker could craft a prompt that directs the agent to fetch attacker-controlled URLs, exfiltrate data via outbound requests, or interact with internal network services (SSRF). This is an LLM-specific SSRF risk.
- **Recommendation:** Restrict `WebFetch` to an allowlist of permitted domains where possible. If unrestricted fetching is required, ensure tasks given to `ResearchAgent` are never derived from untrusted input. Document the SSRF risk prominently.

---

### [MEDIUM] Insecure Temporary File for MCP Config (World-Readable, No Guaranteed Cleanup)

- **File:** `cckit/mcp/manager.py:49-56`
- **Description:** `MCPManager.write_config_file()` creates a temporary file using `tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="claude_mcp_")`. Two issues:
  1. The file is created with default OS permissions (typically `0o600` on most Unix systems, but this is not enforced by the code and can vary). On some systems or with unusual umasks, this file may be world-readable.
  2. `delete=False` means the file persists on disk until `MCPManager.cleanup()` is explicitly called. If the calling code crashes, raises an exception between `write_config_file()` and `cleanup()`, or simply forgets to call `cleanup()`, the MCP config file (which may contain `env` variables with secrets such as API keys) remains on disk in `/tmp` indefinitely.
  
  The `mcp_integration.py` example does use a `finally` block to call `cleanup()`, but there is no enforcement at the library level (e.g., a context manager).
- **Recommendation:** (1) Explicitly set file permissions to `0o600` after creation using `os.chmod(tmp.name, 0o600)`. (2) Implement `MCPManager` as a context manager (`__enter__`/`__exit__`) so cleanup is automatic. (3) Consider whether the `env` dict (which callers use to pass secrets to MCP servers) should ever be written to disk at all—an in-memory approach would be safer.

---

### [MEDIUM] Unvalidated `session_id` from LLM Response Used in CLI Arguments

- **File:** `cckit/session/session.py:74-75`, `cckit/core/cli.py:124-125`
- **Description:** The `session_id` stored in a `Session` object is sourced directly from the LLM's JSON response (`response.session_id`, which comes from `data.get("session_id", "")` in `parser.py`). This value is then passed directly to the `--resume` CLI flag: `builder.with_resume(session_id)` → `self._flags.extend(["--resume", session_id])`. Because the command is built as a list and executed via `create_subprocess_exec`, there is no shell injection risk. However, there is no validation that `session_id` conforms to an expected format (e.g., a UUID). A malformed or adversarially crafted `session_id` in a JSON response (e.g., a very long string, or a string containing null bytes or special characters) could cause unexpected CLI behavior or errors that are not properly handled.
- **Recommendation:** Validate that `session_id` matches an expected pattern (e.g., `re.match(r'^[a-zA-Z0-9\-_]{1,128}$', session_id)`) before storing or using it. Raise a `ParseError` or `SessionError` if the format is invalid.

---

### [MEDIUM] Prompt Injection Risk — No Sanitization of User Input Before Passing to LLM

- **File:** `cckit/core/cli.py:141`, `cckit/agents/base.py:47-50`
- **Description:** All `execute()`, `execute_streaming()`, and `execute_json()` methods accept a `prompt: str` parameter and pass it directly to the LLM without any sanitization, length limits, or injection detection. The `system_prompt` and `append_system_prompt` fields in `SessionConfig` are similarly unvalidated. If application code constructs prompts by embedding user-supplied content (e.g., `f"Summarize this document: {user_input}"`), a user can inject LLM instructions that override the system prompt, change tool usage, exfiltrate information, or manipulate agent behavior. This is the classic prompt injection vulnerability in agentic systems.
- **Recommendation:** This is an inherent challenge in LLM-based systems, but mitigations include: (1) Never embed untrusted user input directly in the system prompt. (2) Use structural delimiters and XML-style tags to separate instructions from user data. (3) Apply a maximum length limit to user-supplied inputs. (4) Log and monitor for suspicious patterns (e.g., "ignore previous instructions"). (5) Prefer using the `user` role context separation where possible.

---

### [MEDIUM] `MessageHistory.load()` Deserializes Arbitrary JSON from Disk Without Schema Validation

- **File:** `cckit/session/history.py:53-72`
- **Description:** `MessageHistory.load(path)` reads a file path and deserializes it with `json.loads()`, then constructs `Message` and `ToolUse` objects by directly accessing dict keys (`item["role"]`, `item["content"]`, `t["tool_name"]`, `t["tool_input"]`). If a malicious or corrupted file is loaded, the code will raise unhandled `KeyError` or `TypeError` exceptions that bubble up without context. More importantly, the `tool_input` field (a `dict[str, Any]`) is loaded without any schema enforcement—a manipulated history file could inject arbitrary tool inputs that get replayed into a live session, potentially tricking the agent into running malicious tool calls.
- **Recommendation:** (1) Validate the loaded data against a schema before constructing objects (Pydantic is already a project dependency—use it). (2) Never load a history file from a path supplied by untrusted input. (3) Add explicit error handling around deserialization to surface `KeyError`/`ValueError` as `ParseError`.

---

### [MEDIUM] `expand_path` Resolves Environment Variables — Potential Path Manipulation

- **File:** `cckit/utils/helpers.py:9-11`
- **Description:** `expand_path()` calls both `os.path.expandvars()` and `os.path.expanduser()` before resolving the path. If any caller passes a path that originates from user or LLM-controlled input (e.g., `cfg.cwd`, `cfg.mcp_config_path`, `binary_path`), an attacker can embed environment variable references (e.g., `$HOME/../../../etc/passwd`) to redirect path resolution. The `cwd` parameter in `SessionConfig` is passed through `expand_path()` before being used as the `--cwd` flag, which sets the working directory for the claude CLI process. A manipulated `cwd` could point the agent's operations at unintended filesystem locations.
- **Recommendation:** Do not expand environment variables in paths that originate from user-supplied input. For `cwd` and `mcp_config_path`, accept only absolute paths or validate/normalize them after expansion to ensure they remain within expected boundaries. Consider using `pathlib.Path.resolve()` without `expandvars` for user-controlled paths.

---

### [LOW] `datetime.utcnow()` Deprecated — Use Timezone-Aware Datetimes

- **File:** `cckit/types/messages.py:19`
- **Description:** `timestamp: datetime = field(default_factory=datetime.utcnow)` uses the deprecated `datetime.utcnow()` (deprecated in Python 3.12). This produces naive datetime objects with no timezone information. While not a direct security vulnerability, naive datetimes can cause subtle bugs in timestamp comparison, log correlation, and session tracking, which can complicate incident response and forensic analysis.
- **Recommendation:** Replace with `datetime.now(tz=timezone.utc)` from `datetime` to produce timezone-aware timestamps.

---

### [LOW] No Timeout Enforced by Default on Process Execution

- **File:** `cckit/core/config.py:28`, `cckit/core/process.py:15`
- **Description:** `CLIConfig.timeout` defaults to `None`, and `ProcessManager` sets `self.timeout = timeout` without a default. When `timeout` is `None`, `asyncio.wait_for()` in `ProcessManager.run()` is called with `timeout=None`, which disables the timeout entirely. A hung or slow `claude` CLI process will block the event loop indefinitely. In `stream_lines()`, there is no timeout at all—the `while True` readline loop can run forever if the process produces no output.
- **Recommendation:** Set a sensible default timeout (e.g., 300 seconds) in `CLIConfig`. Apply a read timeout to `stream_lines()` using `asyncio.wait_for()` on each `readline()` call, or wrap the entire streaming loop in an outer timeout.

---

### [LOW] `add_flag()` on `CommandBuilder` Accepts Arbitrary Strings Without Validation

- **File:** `cckit/core/command.py:97-101`
- **Description:** `CommandBuilder.add_flag(flag, value)` appends an arbitrary string flag and optional value to the command list. While not subject to shell injection (due to `exec`-style invocation), it allows any caller to inject unknown or dangerous `claude` CLI flags. This is the underlying mechanism exploited by `extra_flags` (Finding 1), but it is also directly callable by library users.
- **Recommendation:** Consider making `add_flag()` a private method (`_add_flag`) or adding documentation warning that it bypasses all argument validation. An allowlist of recognized flags would be the strongest mitigation.

---

### [LOW] MCP Server `env` Dict May Contain Secrets Written to World-Accessible Temp File

- **File:** `cckit/mcp/manager.py:49-56`, `cckit/mcp/server.py:16-19`
- **Description:** `MCPServer.to_dict()` includes the `env` dict in the serialized configuration. Callers may place API keys, tokens, or passwords in this dict (e.g., `env={"GITHUB_TOKEN": "ghp_..."}`) for MCP server authentication. These values are then written to the temporary JSON file on disk (see Finding 5). Even with `0o600` permissions, the secrets are now persisted on disk where they may appear in backups, core dumps, or forensic images.
- **Recommendation:** Consider supporting environment variable *names* rather than *values* in the MCP config (i.e., the server reads `os.environ["GITHUB_TOKEN"]` directly), avoiding the need to write secret values to disk. If values must be written, ensure the temp file is created on a `tmpfs` mount or in a memory-backed directory where available.

---

### [LOW] `ParseError.raw` Field Stores Raw LLM Output — Potential for Sensitive Data in Exceptions

- **File:** `cckit/utils/errors.py:26-31`, `cckit/streaming/handler.py:29`
- **Description:** `ParseError` stores the raw unparseable line in `self.raw`. In `StreamHandler.process_stream()`, it is logged: `logger.warning("ParseError on line: %s | %s", exc.raw[:80], exc)`. If an exception propagates to an unhandled handler or is serialized (e.g., sent to an error tracking system like Sentry), the `raw` field may contain fragments of LLM responses that include sensitive information from the conversation context.
- **Recommendation:** Truncate `exc.raw` to a safe length (already done with `[:80]` in logging, but not in the `ParseError` object itself). Consider hashing or omitting the raw content in non-debug builds.

---

### [INFORMATIONAL] No `.env` File or Secret Management in Place

- **File:** Repository root
- **Description:** No `.env` file, `os.environ` reads for API keys, or secret management is present in the codebase. The framework delegates authentication entirely to the `claude` CLI binary. This is appropriate for the current design but means there is no in-library mechanism to rotate credentials or detect expired tokens.
- **Recommendation:** Document this clearly. If the framework is extended to call the Anthropic API directly, implement secret management via environment variables (not hardcoded values) and add `ANTHROPIC_API_KEY` to `.gitignore` / secret scanning rules.

---

### [INFORMATIONAL] No Input Length Limits on Prompts

- **File:** `cckit/core/cli.py:35-50`
- **Description:** There is no maximum length enforced on the `prompt` parameter. Very large prompts can cause high token costs, slow responses, or CLI argument length errors on some operating systems (Linux: 2MB `ARG_MAX`). While not a security vulnerability in isolation, this can be weaponized for denial-of-service if the execute functions are exposed in a service.
- **Recommendation:** Add an optional `max_prompt_length` field to `CLIConfig` and enforce it in `_build_command()`.

---

### [INFORMATIONAL] Dependency on External Binary (`~/.local/bin/claude`) Not Verified

- **File:** `cckit/core/config.py:27`, `cckit/core/command.py:12`
- **Description:** The framework executes `~/.local/bin/claude` (or a user-specified binary path) without verifying its integrity (checksum, signature, or version). If the binary is replaced by a malicious actor (supply chain attack, compromised `~/.local/bin/`), the framework will silently execute the replacement. The `expand_path()` function resolves symlinks, so a symlink attack would also work.
- **Recommendation:** Optionally verify the binary's SHA-256 hash against a known-good value at startup. Log the resolved binary path and version (`claude --version`) at INFO level on first use to enable detection of unexpected binary changes.

---

## Dependency Analysis

**File:** `pyproject.toml` / `uv.lock`

| Package | Version | Notes |
|---------|---------|-------|
| `pydantic` | `>=2.0` (locked: 2.13.1) | Modern v2, no known critical CVEs in 2.13.x. Lower bound (`>=2.0`) is wide — an older 2.x release with a vulnerability would satisfy the constraint. Consider `>=2.10`. |
| `pydantic-core` | 2.46.1 (locked) | Rust-backed core for pydantic; no known CVEs in this version. |
| `annotated-types` | 0.7.0 | No known issues. |
| `typing-extensions` | 4.15.0 | No known issues. |
| `typing-inspection` | 0.4.2 | No known issues. |
| `pytest` | `>=9.0.3` (locked: 9.0.3) | Dev dependency only. No known critical issues. |
| `pytest-asyncio` | `>=1.3.0` (locked: 1.3.0) | Dev dependency only. No known critical issues. |
| `colorama` | 0.4.6 | Transitive dep. No known issues. |
| `pygments` | 2.20.0 | Transitive dep. No known issues. |

**Key observations:**
- The dependency surface is very small, which is positive from a supply-chain security perspective.
- No `anthropic` SDK is declared as a dependency (the framework calls the CLI binary, not the API directly). This avoids the risk of SDK-level vulnerabilities but also means no automatic updates when the SDK patches security issues.
- The lower bound on `pydantic` (`>=2.0`) is wider than necessary. The locked version (2.13.1) is fine, but a fresh install on a system without `uv.lock` could resolve to an older 2.x with potential issues. Tighten to `>=2.10`.
- No security-scanning tools (`bandit`, `safety`, `pip-audit`) are configured in `pyproject.toml` or as dev dependencies.

**Recommendation:** Add `bandit` and `pip-audit` (or `uv audit`) as dev dependencies and integrate them into the CI pipeline.

---

## Summary Table

| # | Severity | Title | File |
|---|----------|-------|------|
| 1 | High | Unsanitized `extra_flags` allows arbitrary CLI argument injection | `cckit/core/config.py:30`, `core/cli.py:138` |
| 2 | High | Prompt content logged at DEBUG level (sensitive data exposure) | `cckit/core/process.py:21,50` |
| 3 | High | `PermissionMode.BYPASS` exposed without guardrails | `cckit/types/enums.py:14` |
| 4 | Medium | `CodeAgent` grants unrestricted `Bash` tool by default | `cckit/agents/code.py:6` |
| 5 | Medium | `ResearchAgent` grants unrestricted `WebSearch`/`WebFetch` (SSRF risk) | `cckit/agents/research.py:6` |
| 6 | Medium | Insecure temporary file for MCP config (world-readable, no guaranteed cleanup) | `cckit/mcp/manager.py:49-56` |
| 7 | Medium | Unvalidated `session_id` from LLM response used in CLI arguments | `cckit/session/session.py:74-75` |
| 8 | Medium | Prompt injection risk — no sanitization of user input | `cckit/core/cli.py:141`, `agents/base.py:47` |
| 9 | Medium | `MessageHistory.load()` deserializes arbitrary JSON without schema validation | `cckit/session/history.py:53-72` |
| 10 | Medium | `expand_path` resolves env vars — potential path manipulation | `cckit/utils/helpers.py:9-11` |
| 11 | Low | `datetime.utcnow()` deprecated — use timezone-aware datetimes | `cckit/types/messages.py:19` |
| 12 | Low | No default timeout on process execution | `cckit/core/config.py:28`, `process.py:15` |
| 13 | Low | `add_flag()` accepts arbitrary strings without validation | `cckit/core/command.py:97-101` |
| 14 | Low | MCP server `env` dict may contain secrets written to disk | `cckit/mcp/manager.py:49-56`, `mcp/server.py:16` |
| 15 | Low | `ParseError.raw` may contain sensitive LLM output in exceptions | `cckit/utils/errors.py:26-31`, `streaming/handler.py:29` |
| 16 | Informational | No secret management — delegates entirely to CLI binary | Repository root |
| 17 | Informational | No input length limits on prompts | `cckit/core/cli.py:35-50` |
| 18 | Informational | External binary not integrity-verified at startup | `cckit/core/config.py:27` |
