# OCES — OpenAI-Compatible Example Server (design spec)

**Status:** Ready for review
**Date:** 2026-05-07
**Target file:** `examples/openai_server.py` (single file, ~600–800 LOC)
**Reference contract:** `examples/openapi.yml` (`/chat/completions` only, this iteration)

---

## 1. Goal & non-goals

**Goal:** Ship a single-file FastAPI example that exposes an OpenAI-compatible `POST /chat/completions` endpoint, backed by cckit driving the real `claude` CLI binary. The endpoint must work transparently with the official `openai` Python SDK, raw `httpx` SSE consumers, and `curl --no-buffer` — that's the bar.

**Non-goals (this iteration):**
- `/completions` (legacy), `/embeddings`, `/models` — out of scope.
- Persistent sessions / server-side conversation memory.
- Multiple-key auth, key-per-user accounting, billing.
- Real OpenAI parity for `n>1`, `logprobs`, `temperature` (claude CLI exposes no equivalents).
- Tokenizer integration — we use char-based heuristics for `max_tokens`.

**Hard constraint:** Code must NOT import or depend on `claude_agent_sdk` / `@anthropic-ai/claude-agent-sdk`. cckit wraps the CLI directly; OCES uses cckit only.

---

## 2. Architecture

```
client (openai SDK / httpx / curl)
       │   POST /chat/completions   Bearer <OCES_API_KEY>
       ▼
OCES (FastAPI, examples/openai_server.py)
   • bearer auth (validates OCES key)
   • Pydantic request validation
   • prompt builder: messages[] → claude system+user prompt
   • cckit driver (TextChunkEvent stream)
   • sentinel-envelope parser (state machine over text)
   • OpenAI chunk emitter (SSE) or response collector (JSON)
       │   subprocess argv + stdin
       ▼
cckit (cckit.CLI / cckit.CustomAgent)
   • spawns: claude -p --print --bare --tools "" --include-partial-messages
                    --output-format stream-json --verbose --system-prompt <…> <user-prompt>
       │
       ▼
claude CLI binary
   • own auth: oauth/keychain or ANTHROPIC_API_KEY
       │   HTTPS
       ▼
Anthropic Models API
```

OCES is a **stateless translator**. Each HTTP request rebuilds claude's view from scratch via `messages[]`. No server-side history, no claude session reuse across HTTP requests.

**Auth decoupling:** `API_KEY` env var validates OCES clients only. Anthropic credentials are claude's concern; OCES never reads or forwards them.

---

## 3. Wire-format contract (load-bearing)

### 3.1 Routes

OCES mounts the route at both:
- `POST /chat/completions`
- `POST /v1/chat/completions`

Same handler. `/v1/...` is the path most OpenAI-SDK installs are configured for.

### 3.2 Request — `ChatCompletionRequest`

Pydantic v2 model mirroring `examples/openapi.yml`'s `ChatCompletionRequest`. Fields accepted, semantics summarized:

| Field | Required | Behavior |
|---|---|---|
| `model` | yes | **Pass-through-when-claude** (case-insensitive): if it matches `^claude-` or alias `sonnet|opus|haiku`, forwarded via `--model`; otherwise ignored (claude default). Always echoed verbatim in response `model`. |
| `messages` | yes (≥1) | Flattened into single prompt (§5). |
| `stream` | no | Boolean. Selects SSE vs JSON branch. |
| `stream_options.include_usage` | no | Emits a usage chunk before terminator. **400 if true while `stream=false`.** |
| `tools` | no | Triggers sentinel-envelope mode. |
| `tool_choice` | no | Renders into envelope instructions. |
| `parallel_tool_calls` | no | Renders into envelope instructions. |
| `response_format` | no | `text` (default), `json_object`, `json_schema` — best-effort prompt injection (§5.4). |
| `stop` | no | Post-process truncate at first occurrence. |
| `max_tokens` / `max_completion_tokens` | no | Char-budget truncate (~4 chars/token), `finish_reason=length`. |
| `n` | no | **400 if `n>1`.** |
| `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `seed`, `logit_bias`, `logprobs`, `top_logprobs`, `user` | no | Accepted, **silently ignored**. One INFO log per request. |

### 3.3 Response — non-streaming

`Content-Type: application/json`. Body shape matches `ChatCompletionResponse` in the OpenAPI spec verbatim:

```json
{
  "id": "chatcmpl-<24-char-hex>",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "<echoed verbatim>",
  "system_fingerprint": "cckit-oces-0.1",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "<text or null when tool_calls present>",
      "tool_calls": [
        {"id": "call_xxx", "type": "function",
         "function": {"name": "...", "arguments": "<JSON-encoded string>"}}
      ]
    },
    "finish_reason": "stop" | "length" | "tool_calls"
  }],
  "usage": {
    "prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M,
    "prompt_tokens_details": {"cached_tokens": K}
  }
}
```

`tool_calls` key is **omitted entirely** when there are none (not `[]`) — matches OpenAI behavior.

### 3.4 Response — streaming (SSE)

`Content-Type: text/event-stream`. Headers:

```
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

Each chunk is a single line `data: <json>\n\n` followed by either more chunks or the terminator `data: [DONE]\n\n`.

**Chunk sequence (text-only response):**

```
data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"...",
       "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"...","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Chunk sequence (tool-call response):**

```
data: {"...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"...","choices":[{"index":0,"delta":{"tool_calls":[
  {"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}
]},"finish_reason":null}]}

data: {"...","choices":[{"index":0,"delta":{"tool_calls":[
  {"index":0,"function":{"arguments":"{\"city\":\"Paris\"}"}}
]},"finish_reason":null}]}

data: {"...","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

For multiple parallel tool calls, repeat the header+args chunk pair with `"index": 1, 2, ...`.

**With `stream_options.include_usage=true`**, one extra chunk before the terminator:

```
data: {"...","choices":[],"usage":{"prompt_tokens":N,...}}

data: [DONE]
```

`id`, `object: "chat.completion.chunk"`, `created`, `model` repeat on every chunk.

### 3.5 Error envelope

Every non-2xx response (and mid-stream errors) uses:

```json
{"error": {"message": "…", "type": "<type>", "param": null, "code": "<code>"}}
```

For mid-stream errors, the final chunk before `data: [DONE]` is:

```json
{"choices":[{"index":0,"delta":{},"finish_reason":"error"}],
 "error":{"message":"…","type":"server_error","code":"…"}}
```

`finish_reason: "error"` is non-spec but documented in README as the least-disruptive option.

---

## 4. Auth & validation

### 4.1 Bearer

`API_KEY` env var, single token. `verify_bearer` FastAPI dependency:

```python
def verify_bearer(authorization: str = Header(default="")) -> None:
    if not API_KEY:
        raise HTTPException(500, _err("server_error", "API_KEY not configured"))
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, _err("invalid_request_error", "Missing bearer token"))
    if not hmac.compare_digest(authorization[7:].strip(), API_KEY):
        raise HTTPException(401, _err("invalid_request_error", "Invalid API key"))
```

Constant-time compare via `hmac.compare_digest`.

### 4.2 Validation rules (HTTP 400 → invalid_request_error)

| Rule | Message |
|---|---|
| `messages` empty | "messages must be a non-empty array" |
| `n > 1` | "n>1 not supported" |
| `stream_options.include_usage=true` while `stream=false` | "stream_options.include_usage requires stream=true" |
| `tool_choice` references tool not in `tools` | "tool_choice.function.name not in tools" |
| `tool_choice="required"` while `tools=[]` | "tool_choice=required requires tools" |
| `response_format.type=json_schema` without `json_schema.schema` | "json_schema.schema is required" |
| Pydantic validation failure | re-formatted from Pydantic errors |

Implemented as a single `validate_request(req)` pure function returning `HTTPException | None`.

A global `RequestValidationError` handler reshapes FastAPI's default 422 body into the OpenAI envelope.

### 4.3 Error mapping (cckit / claude → HTTP)

| Source condition | HTTP | type | code |
|---|---|---|---|
| `cckit.AuthError` | 503 | `server_error` | `claude_auth_unavailable` |
| `cckit.CLIError` (exit_code≠0) | 502 | `server_error` | `claude_cli_failed` |
| `cckit.TimeoutError` | 504 | `server_error` | `claude_timeout` |
| `cckit.ParseError` | 502 | `server_error` | `claude_parse_error` |
| Bridge sees malformed JSON inside `<<<TOOL_CALLS>>>` | 502 | `server_error` | `bridge_parse_error` |
| Bridge sees no envelope when `tools` non-empty | 502 | `server_error` | `bridge_envelope_missing` |
| stderr indicates rate limit / overload | 429 | `rate_limit_error` | claude code |
| anything else | 500 | `server_error` | `unexpected` |

Single `@app.exception_handler(Exception)` catches and shapes via `_err()`. cckit-specific exceptions matched first.

---

## 5. Internal pipeline

### 5.1 Module layout (top-down, single file)

1. Imports + constants (`SSE_HEADERS`, `ENVELOPE_*` tag constants, regex for claude-model match).
2. Pydantic models (`ChatCompletionRequest`, `ChatCompletionResponse`, `ChatCompletionChunk`, `ChatMessage`, `ContentPart`, `Tool`, `ToolCall`, `ToolChoice`, `ResponseFormat`, `ErrorResponse`).
3. Helpers: `_err`, `make_request_id`, `resolve_claude_model`, `truncate_at_stop`, `truncate_max_tokens`, `map_usage`.
4. `verify_bearer` dependency.
5. `validate_request` pure function.
6. `build_prompt(req) -> tuple[str, str]` — system + user.
7. `parse_envelope_stream(chunks, tools_present) -> AsyncIterator[ParserEvent]` — sentinel state machine.
8. `drive_claude(req, final_usage) -> AsyncIterator[str]` — cckit invocation.
9. `stream_openai(parser_events, ..., final_usage) -> AsyncIterator[str]` — SSE chunks.
10. `collect_openai(parser_events, ..., final_usage) -> dict` — non-streaming response.
11. `app = FastAPI(...)` + route handlers + global exception handler.
12. `if __name__ == "__main__"` — uvicorn launcher reading `HOST`, `PORT`, `API_KEY`.

No classes. Pure functions + async generators. KISS.

### 5.2 `build_prompt(req)` — pure function

Returns `(system_prompt, user_prompt)`.

**System prompt construction:**
1. Concat all `system` and `developer` role messages, in order, joined by `\n\n`.
2. If `req.tools` non-empty: append envelope instructions and tool list.
3. If `req.response_format.type` in `{json_object, json_schema}`: append JSON-only instruction inside the `<<<CONTENT>>>` block; for `json_schema`, also include the schema JSON.

**Envelope instructions (verbatim, when tools present):**

```
You will reply in EXACTLY this format. No prose before or after.

<<<TOOL_CALLS>>>
<JSON array of tool calls, or [] if you are not calling any tool>
<<</TOOL_CALLS>>>
<<<CONTENT>>>
<your natural-language reply, possibly empty if a tool was called>
<<</CONTENT>>>

Each tool call is: {"id":"call_<short-hex>","name":"<tool_name>","arguments":<JSON object>}
Use double quotes everywhere. Do not wrap in markdown fences.

Available tools:
- name: <tool[0].function.name>
  description: <…>
  parameters: <JSON Schema>
- name: ...

Tool selection rules:
- tool_choice="<value>": <expanded instruction per choice>
- parallel_tool_calls=<bool>: <expanded>
```

**User prompt construction** (the conversation transcript):
1. Walk remaining messages (`user`, `assistant`, `tool`).
2. For `user`/`assistant`: render as `[User]: …` / `[Assistant]: …`. Multi-part `content` arrays → text parts joined with `\n`; image parts replaced with `[image: <truncated-url>]` (CLI is text-only).
3. For `tool` role messages (tool result returns): render as `[Tool <tool_call_id> result]: <content>`.
4. Trailing marker: `\n[Assistant]: ` so claude continues from the assistant turn.

**Sampling-knob ignorance**: build a list of present-but-ignored fields and log INFO per request. Do not modify the prompt.

### 5.3 Sentinel-envelope parser — state machine

Constants:
```python
TOOL_OPEN  = "<<<TOOL_CALLS>>>"
TOOL_CLOSE = "<<</TOOL_CALLS>>>"
CONT_OPEN  = "<<<CONTENT>>>"
CONT_CLOSE = "<<</CONTENT>>>"
TAG_MAX    = max(len(t) for t in [TOOL_OPEN, TOOL_CLOSE, CONT_OPEN, CONT_CLOSE])  # 17 (= len of TOOL_CLOSE)
```

States (single int):
- `0` searching for `TOOL_OPEN`
- `1` buffering tool-calls JSON until `TOOL_CLOSE`
- `2` searching for `CONT_OPEN`
- `3` emitting content text deltas until `CONT_CLOSE`
- `4` done; ignore trailing chars

Implementation:
```python
async def parse_envelope_stream(chunks, tools_present) -> AsyncIterator[dict]:
    if not tools_present:
        # Pass-through — strip nothing, just pipe text deltas
        async for chunk in chunks:
            if chunk:
                yield {"kind": "text_delta", "text": chunk}
        yield {"kind": "finish", "reason": "stop"}
        return

    state = 0
    buf = ""  # rolling buffer for tag-boundary safety
    tool_json_buf = ""
    async for chunk in chunks:
        buf += chunk
        # State transitions (see implementation pseudocode below)
        ...
```

Tag-boundary handling: buf retains the last `TAG_MAX-1` chars after every match attempt so a tag split across chunks is reassembled.

**Output events:**
- `{"kind": "text_delta", "text": "..."}` — emit as OpenAI content delta.
- `{"kind": "tool_calls", "calls": [{"id":..., "name":..., "arguments":<dict>}]}` — emit two OpenAI chunks per call.
- `{"kind": "finish", "reason": "stop" | "tool_calls" | "length"}` — emit final OpenAI delta chunk.
- `{"kind": "error", "code": "bridge_parse_error" | "bridge_envelope_missing", "message": "..."}` — surface as 502 (non-streaming) or terminating-error chunk (streaming).

**Markdown-fence resilience:** if `tool_json_buf.strip()` starts with ` ```json ` or ` ``` `, strip the fence and trailing ` ``` `. (Single concession to claude occasionally over-formatting.)

**Empty tool_calls handling:** when `tool_json_buf.strip()` parses to `[]`, finish_reason is `stop` (text-only response); when non-empty, finish_reason is `tool_calls` (content may still be present but is typically empty).

### 5.4 `response_format` handling

| `type` | System-prompt addition | Post-process |
|---|---|---|
| `text` (default) | none | none |
| `json_object` | "Inside `<<<CONTENT>>>`, reply with a single valid JSON object. No code fences." | `json.loads()` the content; on failure, return 422 (or terminating-error chunk) with `code=invalid_response_format` |
| `json_schema` | json_object instruction + "Conform to this schema:" + the schema JSON | none (instructions-only, best-effort) |

**Runtime JSON validation is intentionally deferred** — `response_format` is implemented as system-prompt instructions only. Clients receiving JSON-shaped responses are expected to validate themselves (matches the "best-effort prompt injection" framing in §3.2). Adding runtime validation is a viable follow-up but breaks naturally on streaming (would have to buffer the full response). Out of scope for this iteration.

`jsonschema` is still listed in the `[examples]` extras group as a forward-looking dep — useful when validation is added later, no harm if unused now.

### 5.5 cckit driver — `drive_claude(req, final_usage)`

```python
async def drive_claude(req, final_usage):
    cli = CLI(config=CLIConfig(
        binary_path=os.environ.get("CLAUDE_BINARY", "~/.local/bin/claude"),
        extra_flags=["--tools", "", "--include-partial-messages"],
    ))
    sys_prompt, user_prompt = build_prompt(req)
    agent = CustomAgent(
        cli=cli,
        system_prompt=sys_prompt,
        bare=True,
        model=resolve_claude_model(req.model),
    )
    async for event in agent.stream_execute(user_prompt):
        if isinstance(event, TextChunkEvent):
            yield event.text
        elif isinstance(event, ResultEvent):
            final_usage.set_result(map_usage(event.raw.get("usage", {})))
```

**`map_usage` reads from raw, not typed fields**: cckit's `ResultEvent` exposes `result`, `session_id`, `duration_ms`, `is_error` as typed fields but does NOT promote `usage` (parser.py extracts it locally and discards). So OCES reads it back via `event.raw["usage"]`. Treat as a temporary cckit quirk — possibly worth a follow-up upstream PR, but out of scope here.

**Why `extra_flags=["--tools", "", "--include-partial-messages"]`:**
- `--tools ""` disables claude's built-in tools; with `--bare`, claude becomes a near-pure text generator (no Bash/Read/Edit/etc. running).
- `--include-partial-messages` makes claude emit `content_block_delta` with `text_delta`, which cckit's parser maps 1:1 to `TextChunkEvent`. This is what gives us real token-level streaming.

**Why `system_prompt=` (full replace) and not `append_system_prompt=`:**
- Default Claude-Code system prompt mentions internal tools and Claude-Code identity that would confuse OpenAI clients. Full replace keeps OCES's behavior deterministic and harness-portable.

### 5.6 Composition (route handler)

```python
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _=Depends(verify_bearer)):
    err = validate_request(req)
    if err:
        raise err
    final_usage = asyncio.get_event_loop().create_future()
    chunks = drive_claude(req, final_usage)
    parser = parse_envelope_stream(chunks, tools_present=bool(req.tools))
    if req.stream:
        return StreamingResponse(
            stream_openai(parser, make_request_id(), req.model, int(time.time()),
                          bool(req.stream_options and req.stream_options.include_usage),
                          final_usage),
            media_type="text/event-stream", headers=SSE_HEADERS,
        )
    return JSONResponse(await collect_openai(parser, make_request_id(), req.model,
                                              int(time.time()), final_usage))
```

---

## 6. Testing

### 6.1 Layout

- `tests/test_openai_server.py` — fast suite, fully mocked, default. Runs without claude binary.
- `tests/integration/test_openai_server_real.py` — `@pytest.mark.integration`, opt-in via `pytest -m integration`. Spawns real claude.
- `tests/fixtures/oces_stream_golden.txt` — byte-for-byte SSE snapshot for golden test.

No new core deps. Add `openai>=1.0` and `jsonschema` to `[dependency-groups] dev`.

### 6.2 Mock strategy

FastAPI dependency override on a `get_agent_factory` callable. Tests provide a `make_fake_agent(text_chunks: list[str], usage: Usage)` whose `.stream_execute()` yields the chunks then a `ResultEvent`. Production wiring untouched.

### 6.3 Test categories

**Unit — pure functions:**
- `build_prompt`: system+user only; with developer role; with tools envelope; with response_format=json_object; tool result roles in history; multimodal image elision.
- `parse_envelope_stream`: tools off (passthrough); tools on, empty `[]`+content; one tool call; multiple parallel; tag boundary split (`<<<TOOL` then `_CALLS>>>`); malformed JSON → error event; markdown fence stripping; missing CONTENT block.
- `map_usage`: full Usage; zero usage.
- `resolve_claude_model`: gpt-4o → None; claude-sonnet-4-6 → pass; sonnet → pass; opus → pass; empty → None.
- `truncate_at_stop`, `truncate_max_tokens`.

**Endpoint (`TestClient` + fake agent):**
- Happy path non-streaming, streaming.
- Tool-call response in both modes.
- Auth failures (missing, wrong) → 401.
- All §4.2 validation rejections → 400.
- `stream_options.include_usage=true` with `stream=true` → usage chunk emitted.
- Bridge parse error → 502 (non-stream); mid-stream error chunk + `[DONE]` (stream).
- cckit timeout → 504.
- Both `/chat/completions` and `/v1/chat/completions` routes → identical behavior.

**Harness conformance (LOAD-BEARING — wire compat is the bar):**
- `openai>=1.0` Python SDK pointed at `TestClient` via custom transport: `client.chat.completions.create()` non-stream + stream + tool calls. SDK must parse without error.
- Raw httpx SSE consumer: every chunk is parseable JSON; `data: [DONE]` arrives; content-type starts with `text/event-stream`.
- Golden snapshot: stream output diffed byte-for-byte against `tests/fixtures/oces_stream_golden.txt`. Any drift fails. Maintainer regenerates intentionally.

**Integration (opt-in, real claude):**
- One smoke test: prompt "reply with the word 'pong' and nothing else" → response contains "pong". Run via `pytest -m integration`. Skipped by default per existing `pyproject.toml` config.

---

## 7. Packaging & ops

### 7.1 Dependency group

In `pyproject.toml`:

```toml
[project.optional-dependencies]
examples = ["fastapi>=0.110", "uvicorn[standard]>=0.27", "jsonschema>=4.0"]

[dependency-groups]
dev = [
  "pytest>=9.0.3", "pytest-asyncio>=1.3.0",
  "openai>=1.0",  # for harness conformance tests
]
```

cckit's core deps stay unchanged. Users opt in: `uv pip install "cckit[examples]"`.

### 7.2 Running

```bash
export API_KEY=sk-test-1234567890
export CLAUDE_BINARY=~/.local/bin/claude  # optional
uv run python examples/openai_server.py
# OCES on http://0.0.0.0:8000

# verify with curl
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"hi"}],"stream":true}'

# or with the openai SDK
python -c '
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-test-1234567890")
print(c.chat.completions.create(model="sonnet",
  messages=[{"role":"user","content":"hi"}]).choices[0].message.content)
'
```

`HOST` and `PORT` env vars override defaults.

### 7.3 README

Add a section to `examples/openai_server.py` module docstring (KISS — no separate README) covering:
- env vars
- curl example
- openai SDK example
- known limitations: no n>1, no logprobs, sampling knobs ignored, char-based max_tokens, mid-stream errors use `finish_reason="error"`
- mention that `/chat/completions` and `/v1/chat/completions` are aliases
- explicit note: "Tested with `openai>=1.0`, raw httpx, and curl. Other harnesses should work."

---

## 8. Out of scope / explicit deferrals

- `/completions`, `/embeddings`, `/models` — possible follow-up examples, separate files.
- Multiple-key auth, Anthropic-key forwarding — would compromise the "stateless translator" property; intentionally not included.
- Tokenizer-accurate `max_tokens` — char-budget is good enough for an example. If users complain, swap in `tiktoken` later.
- Streaming partial-JSON tool arguments — claude doesn't emit them progressively; documented limitation.
- Caching, rate-limiting, request idempotency — operator concerns, not example concerns.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Claude occasionally violates the envelope (forgets tags, adds prose) | Strict parser → 502 with diagnostic code; users can retry. README documents this isn't 100%. |
| Wire-format drift breaks downstream harnesses | Golden-snapshot test + openai-SDK roundtrip test catch shape changes immediately. |
| cckit gains a new flag and OCES bypasses it | OCES uses `CLIConfig.extra_flags` deliberately for the two flags cckit doesn't model (`--tools ""`, `--include-partial-messages`). When/if cckit grows builders for these, OCES can switch — purely refactor. |
| Stale parser state on premature client disconnect | `try/finally` in `drive_claude` and `stream_openai`; `proc.terminate()` propagated via cckit's process manager. Verified with a disconnect test. |
| Hidden agent-sdk import sneaks in | Explicit `feedback_no_agent_sdk` rule in project memory; CI grep `claude_agent_sdk` would fail (suggested follow-up, not part of this PR). |

---

## 10. Done criteria

- `uv pip install "cckit[examples]"` succeeds.
- `examples/openai_server.py` exists, single file, ≤900 LOC.
- All tests in `tests/test_openai_server.py` pass: `pytest tests/test_openai_server.py`.
- Golden SSE snapshot matches.
- `pytest -m integration tests/integration/test_openai_server_real.py` passes against a logged-in claude.
- Both curl and openai-SDK invocations from §7.2 work end-to-end against a running server.
- No `claude_agent_sdk` reference anywhere in the diff.
