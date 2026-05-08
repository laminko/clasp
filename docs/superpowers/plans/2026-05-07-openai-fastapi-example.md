# OCES (OpenAI-Compatible Example Server) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `examples/openai_server.py` — a single-file FastAPI app exposing an OpenAI-compatible `POST /chat/completions` endpoint backed by cckit driving the real `claude` CLI binary, working transparently with the official `openai` Python SDK and raw `httpx`/`curl` clients.

**Architecture:** Stateless HTTP translator. Every request flattens `messages[]` into a single claude prompt (system + transcript), drives `claude -p --tools "" --include-partial-messages` via cckit, and re-shapes the streamed `TextChunkEvent`s into OpenAI `ChatCompletionChunk` SSE / `ChatCompletionResponse` JSON. Tool-calling uses a sentinel-envelope protocol (`<<<TOOL_CALLS>>>...<<<CONTENT>>>...`) instructed via the system prompt, since claude doesn't expose a "stop at tool_use" mode. KISS, procedural, functional — pure functions + async generators, Pydantic models as data carriers only.

**Tech Stack:**
- Python 3.11+, asyncio
- FastAPI + Pydantic v2 + uvicorn (added under `[project.optional-dependencies] examples`)
- jsonschema (for `response_format=json_schema` validation)
- pytest, pytest-asyncio (already in dev deps)
- openai>=1.0 (added to dev deps for harness-conformance tests)
- cckit (the package this lives in — uses `cckit.CLI`, `cckit.CustomAgent`, `cckit.CLIConfig`, streaming events)

**Hard constraint:** No `claude_agent_sdk` import anywhere. cckit only.

**Reference spec:** `docs/superpowers/specs/2026-05-07-openai-fastapi-example-design.md`. Read it first; this plan implements §1–§10 of the spec.

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Add `[project.optional-dependencies] examples` + `openai`/`jsonschema` to dev deps |
| `examples/openai_server.py` | create | All OCES code: models, helpers, parser, driver, SSE/JSON paths, FastAPI app, uvicorn launcher |
| `tests/test_openai_server.py` | create | Fast unit + endpoint tests (fully mocked, no claude binary) |
| `tests/integration/__init__.py` | create | Marker file for the integration package |
| `tests/integration/test_openai_server_real.py` | create | Opt-in real-claude smoke test (`@pytest.mark.integration`) |
| `tests/fixtures/oces_stream_golden.txt` | create | Byte-for-byte SSE snapshot fixture |

The example file is intentionally one file (~600-800 LOC). No internal module split — matches existing examples' style and KISS preference.

---

## Task ordering principle

Tasks are vertical slices that each leave the codebase in a working state (tests pass). Order: deps first, then pure helpers (no FastAPI), then prompt builder, then parser, then driver, then HTTP plumbing, then error handling, then harness conformance, then integration.

Each task: red → green → commit. Per-task commits are explicit.

---

### Task 1: Add `examples` extras + dev deps

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current pyproject.toml**

Run: `cat pyproject.toml`

- [ ] **Step 2: Add the `[project.optional-dependencies]` table and extend dev deps**

Edit `pyproject.toml`. After the existing `dependencies = [...]` block (which ends around line 16), insert before `[build-system]`:

```toml
[project.optional-dependencies]
examples = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "jsonschema>=4.0",
]
```

In `[dependency-groups]`, modify `dev = [...]` to add `openai>=1.0` and `httpx>=0.27`:

```toml
[dependency-groups]
dev = [
    "pytest>=9.0.3",
    "pytest-asyncio>=1.3.0",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "jsonschema>=4.0",
    "openai>=1.0",
    "httpx>=0.27",
]
```

(Dev gets the examples deps too so tests can import them.)

- [ ] **Step 3: Resolve and install**

Run: `uv sync --all-extras`
Expected: succeeds; `uv.lock` updated.

- [ ] **Step 4: Sanity-check imports**

Run: `uv run python -c "import fastapi, uvicorn, jsonschema, openai, httpx; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add examples extras (fastapi/uvicorn/jsonschema) and dev deps (openai/httpx) for OCES"
```

---

### Task 2: Skeleton file with module docstring

**Files:**
- Create: `examples/openai_server.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_openai_server.py`:

```python
"""Tests for examples/openai_server.py (OCES)."""
import importlib


def test_module_imports():
    mod = importlib.import_module("examples.openai_server")
    assert hasattr(mod, "app"), "FastAPI app must be exported as `app`"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examples.openai_server'`.

- [ ] **Step 3: Create the skeleton file**

Create `examples/openai_server.py`:

```python
"""OCES — OpenAI-Compatible Example Server.

A FastAPI app that exposes a `POST /chat/completions` endpoint compatible with
the OpenAI API, backed by cckit driving the real `claude` CLI binary.

USAGE:
    export API_KEY=sk-test-1234567890
    export CLAUDE_BINARY=~/.local/bin/claude   # optional
    uv run python examples/openai_server.py    # listens on http://0.0.0.0:8000

VERIFY:
    curl -N http://localhost:8000/v1/chat/completions \\
      -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \\
      -d '{"model":"sonnet","messages":[{"role":"user","content":"hi"}],"stream":true}'

LIMITATIONS (documented in the design spec):
    - n>1, logprobs, sampling knobs (temperature/top_p/etc.) accepted but ignored.
    - max_tokens uses a char-budget heuristic (~4 chars/token), no tokenizer dep.
    - Tool-call arguments stream as one chunk (claude doesn't emit them progressively).
    - Mid-stream errors use `finish_reason="error"` (non-spec but standard practice).
    - response_format=json_schema is best-effort: instructions-only, no runtime validation.
      Clients should validate the response themselves.

Tested with openai>=1.0, raw httpx, and curl.
"""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="OCES", version="0.1.0")
```

Also create `examples/__init__.py` if it doesn't exist (so `tests/test_openai_server.py` can `import examples.openai_server`):

Run: `[ -f examples/__init__.py ] || touch examples/__init__.py`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py examples/__init__.py tests/test_openai_server.py
git commit -m "feat(examples): skeleton FastAPI app for OCES (OpenAI-compatible /chat/completions)"
```

---

### Task 3: Pydantic request/response models

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

The models are data-carriers only. We mirror the OpenAPI spec but only the fields we actually act on. Unknown fields tolerated (Pydantic v2 default).

- [ ] **Step 1: Write failing tests for model parsing**

Append to `tests/test_openai_server.py`:

```python
import pytest
from pydantic import ValidationError


def test_parse_minimal_chat_request():
    from examples.openai_server import ChatCompletionRequest
    req = ChatCompletionRequest.model_validate({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert req.model == "gpt-4o-mini"
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "hi"
    assert req.stream is False


def test_parse_chat_request_with_tools():
    from examples.openai_server import ChatCompletionRequest
    req = ChatCompletionRequest.model_validate({
        "model": "sonnet",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }],
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert req.tools[0].function.name == "get_weather"
    assert req.tool_choice == "auto"
    assert req.stream_options.include_usage is True


def test_parse_chat_message_multimodal_content():
    from examples.openai_server import ChatMessage
    msg = ChatMessage.model_validate({
        "role": "user",
        "content": [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ],
    })
    assert isinstance(msg.content, list)
    assert msg.content[0].type == "text"
    assert msg.content[1].type == "image_url"


def test_parse_assistant_message_with_tool_calls():
    from examples.openai_server import ChatMessage
    msg = ChatMessage.model_validate({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
        }],
    })
    assert msg.tool_calls[0].id == "call_abc"
    assert msg.tool_calls[0].function.name == "get_weather"


def test_parse_tool_role_message():
    from examples.openai_server import ChatMessage
    msg = ChatMessage.model_validate({
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "22°C, sunny",
    })
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_abc"


def test_invalid_role_rejected():
    from examples.openai_server import ChatMessage
    with pytest.raises(ValidationError):
        ChatMessage.model_validate({"role": "robot", "content": "hi"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: 6 fails, all `ImportError: cannot import name '<X>' from 'examples.openai_server'`.

- [ ] **Step 3: Add the Pydantic models**

In `examples/openai_server.py`, insert AFTER the imports and BEFORE `app = FastAPI(...)`:

```python
from typing import Any, Literal, Union
from pydantic import BaseModel, ConfigDict, Field


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Union[TextContentPart, ImageContentPart]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Union[str, list[ContentPart], None] = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool = False


class Tool(BaseModel):
    type: Literal["function"]
    function: FunctionDef


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: Literal["function"]
    function: ToolChoiceFunction


ToolChoice = Union[Literal["none", "auto", "required"], ToolChoiceObject]


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: Union[str, list[str], None] = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    user: str | None = None
    seed: int | None = None
    response_format: ResponseFormat | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES Pydantic models for OpenAI chat completion request schema"
```

---

### Task 4: Helpers — `_err`, `make_request_id`, `resolve_claude_model`, `map_usage`

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
def test_err_envelope_shape():
    from examples.openai_server import _err
    body = _err("invalid_request_error", "bad input", code="bad_field", param="messages")
    assert body == {
        "error": {
            "message": "bad input",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "bad_field",
        }
    }


def test_err_envelope_defaults():
    from examples.openai_server import _err
    body = _err("server_error", "boom")
    assert body["error"]["param"] is None
    assert body["error"]["code"] is None


def test_make_request_id_format():
    from examples.openai_server import make_request_id
    rid = make_request_id()
    assert rid.startswith("chatcmpl-")
    assert len(rid) == len("chatcmpl-") + 24


def test_make_request_id_uniqueness():
    from examples.openai_server import make_request_id
    ids = {make_request_id() for _ in range(50)}
    assert len(ids) == 50


@pytest.mark.parametrize("model_in,expected", [
    ("claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("Claude-Sonnet-4-6", "Claude-Sonnet-4-6"),
    ("sonnet", "sonnet"),
    ("OPUS", "OPUS"),
    ("haiku", "haiku"),
    ("gpt-4o-mini", None),
    ("text-davinci-003", None),
    ("", None),
])
def test_resolve_claude_model(model_in, expected):
    from examples.openai_server import resolve_claude_model
    assert resolve_claude_model(model_in) == expected


def test_map_usage_full():
    from examples.openai_server import map_usage
    raw = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 5,
    }
    u = map_usage(raw)
    assert u == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 20},
    }


def test_map_usage_empty():
    from examples.openai_server import map_usage
    u = map_usage({})
    assert u == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: 7 new fails (ImportError on each helper).

- [ ] **Step 3: Implement helpers**

In `examples/openai_server.py`, insert AFTER the Pydantic models and BEFORE `app = FastAPI(...)`:

```python
import re
import secrets

CLAUDE_MODEL_RE = re.compile(r"^(?:claude-|sonnet$|opus$|haiku$)", re.IGNORECASE)


def _err(err_type: str, message: str, *, code: str | None = None, param: str | None = None) -> dict:
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}


def make_request_id() -> str:
    return "chatcmpl-" + secrets.token_hex(12)  # 24 hex chars


def resolve_claude_model(model: str) -> str | None:
    """Pass-through-when-claude. Returns the original string if it looks like a claude
    model alias or full ID; otherwise None (caller omits --model and lets claude default)."""
    if not model:
        return None
    return model if CLAUDE_MODEL_RE.match(model) else None


def map_usage(raw: dict) -> dict:
    """Translate cckit-format usage dict (Anthropic-style keys) to OpenAI-format."""
    pt = raw.get("input_tokens", 0)
    ct = raw.get("output_tokens", 0)
    cached = raw.get("cache_read_input_tokens", 0)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES helpers (_err, make_request_id, resolve_claude_model, map_usage)"
```

---

### Task 5: `truncate_at_stop` and `truncate_max_tokens`

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
def test_truncate_at_stop_first_match():
    from examples.openai_server import truncate_at_stop
    text, truncated = truncate_at_stop("hello\nstop\nworld", ["stop"])
    assert text == "hello\n"
    assert truncated is True


def test_truncate_at_stop_multi_stop_picks_earliest():
    from examples.openai_server import truncate_at_stop
    text, truncated = truncate_at_stop("foo END bar STOP baz", ["STOP", "END"])
    assert text == "foo "
    assert truncated is True


def test_truncate_at_stop_no_match():
    from examples.openai_server import truncate_at_stop
    text, truncated = truncate_at_stop("hello world", ["xyz"])
    assert text == "hello world"
    assert truncated is False


def test_truncate_at_stop_str_param():
    from examples.openai_server import truncate_at_stop
    text, truncated = truncate_at_stop("a STOP b", "STOP")
    assert text == "a "
    assert truncated is True


def test_truncate_at_stop_none():
    from examples.openai_server import truncate_at_stop
    text, truncated = truncate_at_stop("anything", None)
    assert text == "anything"
    assert truncated is False


def test_truncate_max_tokens_under_budget():
    from examples.openai_server import truncate_max_tokens
    text, truncated = truncate_max_tokens("short text", max_tokens=100)
    assert text == "short text"
    assert truncated is False


def test_truncate_max_tokens_over_budget():
    from examples.openai_server import truncate_max_tokens
    long = "x" * 100
    text, truncated = truncate_max_tokens(long, max_tokens=10)  # ~40 chars
    assert len(text) == 40
    assert truncated is True


def test_truncate_max_tokens_none_passes_through():
    from examples.openai_server import truncate_max_tokens
    text, truncated = truncate_max_tokens("anything", max_tokens=None)
    assert text == "anything"
    assert truncated is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: new fails on `truncate_*` imports.

- [ ] **Step 3: Implement**

In `examples/openai_server.py`, append after `map_usage`:

```python
CHARS_PER_TOKEN = 4  # rough heuristic; no tokenizer dep


def truncate_at_stop(text: str, stop: Union[str, list[str], None]) -> tuple[str, bool]:
    """Find earliest occurrence of any stop string and truncate at it. Returns (text, truncated_flag)."""
    if not stop:
        return text, False
    stops = [stop] if isinstance(stop, str) else list(stop)
    earliest = -1
    for s in stops:
        idx = text.find(s)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    if earliest == -1:
        return text, False
    return text[:earliest], True


def truncate_max_tokens(text: str, max_tokens: int | None) -> tuple[str, bool]:
    """Char-budget truncate (~4 chars/token). Returns (text, truncated_flag)."""
    if max_tokens is None:
        return text, False
    budget = max_tokens * CHARS_PER_TOKEN
    if len(text) <= budget:
        return text, False
    return text[:budget], True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES truncation helpers (stop / max_tokens)"
```

---

### Task 6: `validate_request` — all 400 paths

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
def _req(**overrides):
    """Helper: build a minimal valid ChatCompletionRequest with overrides."""
    from examples.openai_server import ChatCompletionRequest
    base = {
        "model": "sonnet",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return ChatCompletionRequest.model_validate(base)


def test_validate_happy_path():
    from examples.openai_server import validate_request
    assert validate_request(_req()) is None


def test_validate_empty_messages():
    from examples.openai_server import validate_request, ChatCompletionRequest
    req = ChatCompletionRequest.model_validate({"model": "x", "messages": []})
    err = validate_request(req)
    assert err is not None and err.status_code == 400
    assert "messages" in err.detail["error"]["message"]


def test_validate_n_greater_than_one():
    from examples.openai_server import validate_request
    err = validate_request(_req(n=2))
    assert err is not None and err.status_code == 400
    assert "n>1" in err.detail["error"]["message"]


def test_validate_include_usage_without_stream():
    from examples.openai_server import validate_request
    err = validate_request(_req(stream=False, stream_options={"include_usage": True}))
    assert err is not None and err.status_code == 400
    assert "stream_options" in err.detail["error"]["message"]


def test_validate_include_usage_with_stream_ok():
    from examples.openai_server import validate_request
    assert validate_request(_req(stream=True, stream_options={"include_usage": True})) is None


def test_validate_tool_choice_unknown_function():
    from examples.openai_server import validate_request
    req = _req(
        tools=[{"type": "function", "function": {"name": "a", "parameters": {}}}],
        tool_choice={"type": "function", "function": {"name": "b"}},
    )
    err = validate_request(req)
    assert err is not None and err.status_code == 400
    assert "tool_choice" in err.detail["error"]["message"]


def test_validate_tool_choice_required_without_tools():
    from examples.openai_server import validate_request
    err = validate_request(_req(tool_choice="required"))
    assert err is not None and err.status_code == 400


def test_validate_json_schema_missing_schema():
    from examples.openai_server import validate_request
    err = validate_request(_req(response_format={"type": "json_schema"}))
    assert err is not None and err.status_code == 400
    assert "json_schema" in err.detail["error"]["message"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v -k validate`
Expected: 8 fails.

- [ ] **Step 3: Implement**

In `examples/openai_server.py`, append:

```python
from fastapi import HTTPException


def validate_request(req: "ChatCompletionRequest") -> HTTPException | None:
    if not req.messages:
        return HTTPException(400, _err("invalid_request_error",
                                       "messages must be a non-empty array",
                                       param="messages"))
    if req.n > 1:
        return HTTPException(400, _err("invalid_request_error",
                                       "n>1 not supported (claude CLI exposes no equivalent)",
                                       param="n"))
    if req.stream_options and req.stream_options.include_usage and not req.stream:
        return HTTPException(400, _err("invalid_request_error",
                                       "stream_options.include_usage requires stream=true",
                                       param="stream_options.include_usage"))
    if req.tool_choice == "required" and not req.tools:
        return HTTPException(400, _err("invalid_request_error",
                                       "tool_choice=required requires tools",
                                       param="tool_choice"))
    if isinstance(req.tool_choice, ToolChoiceObject):
        names = {t.function.name for t in (req.tools or [])}
        if req.tool_choice.function.name not in names:
            return HTTPException(400, _err("invalid_request_error",
                                           "tool_choice.function.name not in tools",
                                           param="tool_choice.function.name"))
    if req.response_format and req.response_format.type == "json_schema":
        if not req.response_format.json_schema or "schema" not in req.response_format.json_schema:
            return HTTPException(400, _err("invalid_request_error",
                                           "json_schema.schema is required when response_format.type=json_schema",
                                           param="response_format.json_schema.schema"))
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v -k validate`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES validate_request — 400 envelopes for all hard rejects"
```

---

### Task 7: `verify_bearer` auth dependency

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
def test_verify_bearer_no_api_key_set(monkeypatch):
    from examples.openai_server import verify_bearer, _OCESConfig
    monkeypatch.setattr(_OCESConfig, "api_key", "")
    with pytest.raises(HTTPException) as exc:
        verify_bearer("Bearer xxx")
    assert exc.value.status_code == 500
    assert "API_KEY not configured" in exc.value.detail["error"]["message"]


def test_verify_bearer_missing(monkeypatch):
    from examples.openai_server import verify_bearer, _OCESConfig
    monkeypatch.setattr(_OCESConfig, "api_key", "secret")
    with pytest.raises(HTTPException) as exc:
        verify_bearer("")
    assert exc.value.status_code == 401


def test_verify_bearer_wrong_format(monkeypatch):
    from examples.openai_server import verify_bearer, _OCESConfig
    monkeypatch.setattr(_OCESConfig, "api_key", "secret")
    with pytest.raises(HTTPException) as exc:
        verify_bearer("secret")  # missing "Bearer " prefix
    assert exc.value.status_code == 401


def test_verify_bearer_wrong_value(monkeypatch):
    from examples.openai_server import verify_bearer, _OCESConfig
    monkeypatch.setattr(_OCESConfig, "api_key", "secret")
    with pytest.raises(HTTPException) as exc:
        verify_bearer("Bearer wrong")
    assert exc.value.status_code == 401


def test_verify_bearer_correct(monkeypatch):
    from examples.openai_server import verify_bearer, _OCESConfig
    monkeypatch.setattr(_OCESConfig, "api_key", "secret")
    assert verify_bearer("Bearer secret") is None


def test_verify_bearer_correct_with_whitespace(monkeypatch):
    from examples.openai_server import verify_bearer, _OCESConfig
    monkeypatch.setattr(_OCESConfig, "api_key", "secret")
    assert verify_bearer("Bearer  secret  ") is None  # tolerant of trailing/leading ws on token
```

Replace top-of-file imports in the test file to also include `HTTPException`:
```python
from fastapi import HTTPException
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v -k bearer`
Expected: 6 fails.

- [ ] **Step 3: Implement**

In `examples/openai_server.py`, near the imports add:

```python
import hmac
import os
from fastapi import Header
```

After helpers, add a tiny config object that tests can monkeypatch (avoids re-reading `os.environ` at every request):

```python
class _OCESConfig:
    api_key: str = os.environ.get("API_KEY", "")
    binary_path: str = os.environ.get("CLAUDE_BINARY", "~/.local/bin/claude")
```

Then:

```python
def verify_bearer(authorization: str = Header(default="")) -> None:
    if not _OCESConfig.api_key:
        raise HTTPException(500, _err("server_error", "API_KEY not configured"))
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, _err("invalid_request_error", "Missing bearer token"))
    token = authorization[7:].strip()
    if not hmac.compare_digest(token, _OCESConfig.api_key):
        raise HTTPException(401, _err("invalid_request_error", "Invalid API key"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v -k bearer`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES bearer auth dependency"
```

---

### Task 8: `build_prompt` — message flattening + envelope instructions

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
def test_build_prompt_simple():
    from examples.openai_server import build_prompt
    sys, user = build_prompt(_req(messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]))
    assert sys == "You are helpful."
    assert "[User]: hi" in user
    assert user.rstrip().endswith("[Assistant]:")


def test_build_prompt_developer_role_joined_to_system():
    from examples.openai_server import build_prompt
    sys, _ = build_prompt(_req(messages=[
        {"role": "developer", "content": "Be concise."},
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]))
    assert "Be concise." in sys
    assert "You are helpful." in sys


def test_build_prompt_with_tools_includes_envelope_instructions():
    from examples.openai_server import build_prompt
    sys, _ = build_prompt(_req(
        messages=[{"role": "user", "content": "weather?"}],
        tools=[{
            "type": "function",
            "function": {"name": "get_weather", "description": "Gets weather",
                         "parameters": {"type": "object"}},
        }],
        tool_choice="auto",
    ))
    assert "<<<TOOL_CALLS>>>" in sys
    assert "<<<CONTENT>>>" in sys
    assert "get_weather" in sys
    assert "Gets weather" in sys
    assert "tool_choice" in sys.lower()


def test_build_prompt_no_tools_no_envelope():
    from examples.openai_server import build_prompt
    sys, _ = build_prompt(_req())
    assert "<<<TOOL_CALLS>>>" not in sys


def test_build_prompt_response_format_json_object():
    from examples.openai_server import build_prompt
    sys, _ = build_prompt(_req(response_format={"type": "json_object"}))
    assert "JSON" in sys.upper()


def test_build_prompt_response_format_json_schema_includes_schema():
    from examples.openai_server import build_prompt
    sys, _ = build_prompt(_req(response_format={
        "type": "json_schema",
        "json_schema": {"schema": {"type": "object", "properties": {"x": {"type": "string"}}}},
    }))
    assert "JSON" in sys.upper()
    assert '"type":"object"' in sys.replace(" ", "") or '"type": "object"' in sys


def test_build_prompt_assistant_history_rendered():
    from examples.openai_server import build_prompt
    _, user = build_prompt(_req(messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
        {"role": "user", "content": "how are you?"},
    ]))
    assert "[User]: hi" in user
    assert "[Assistant]: hello!" in user
    assert "[User]: how are you?" in user
    assert user.rstrip().endswith("[Assistant]:")


def test_build_prompt_tool_role_rendered():
    from examples.openai_server import build_prompt
    _, user = build_prompt(_req(messages=[
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": "{\"city\":\"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "22°C, sunny"},
        {"role": "user", "content": "thanks"},
    ]))
    assert "[Tool call_1 result]: 22°C, sunny" in user


def test_build_prompt_multimodal_image_elided():
    from examples.openai_server import build_prompt
    _, user = build_prompt(_req(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ],
    }]))
    assert "what is this?" in user
    assert "[image:" in user  # placeholder for image
    assert "cat.png" in user or "https://example.com" in user  # url snippet preserved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v -k build_prompt`
Expected: 9 fails.

- [ ] **Step 3: Implement**

In `examples/openai_server.py`, append:

```python
import json as _json


def _render_content(content: Union[str, list, None]) -> str:
    """Render a message's content (str | list of parts | None) as plain text. Images elided."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for p in content:
        if isinstance(p, TextContentPart):
            parts.append(p.text)
        elif isinstance(p, ImageContentPart):
            url = p.image_url.url
            short = url[:80] + ("…" if len(url) > 80 else "")
            parts.append(f"[image: {short}]")
    return "\n".join(parts)


def _envelope_instructions(req: "ChatCompletionRequest") -> str:
    tools_block = []
    for t in req.tools or []:
        f = t.function
        params = _json.dumps(f.parameters or {}, indent=2)
        tools_block.append(f"- name: {f.name}\n  description: {f.description or ''}\n  parameters: {params}")
    tools_str = "\n".join(tools_block)

    tc = req.tool_choice
    if tc == "none":
        tc_rule = 'tool_choice="none": do NOT emit any tool calls; tool_calls must be [].'
    elif tc == "required":
        tc_rule = 'tool_choice="required": emit at least one tool call.'
    elif isinstance(tc, ToolChoiceObject):
        tc_rule = f'tool_choice=function "{tc.function.name}": emit exactly one tool call to that function.'
    else:
        tc_rule = 'tool_choice="auto" (default): use a tool when helpful, otherwise reply directly.'

    parallel_rule = (
        "parallel_tool_calls=true: you may emit multiple tool calls in one turn."
        if req.parallel_tool_calls
        else "parallel_tool_calls=false: emit at most one tool call per turn."
    )

    return f"""
You will reply in EXACTLY this format. No prose before or after.

<<<TOOL_CALLS>>>
<JSON array of tool calls, or [] if you are not calling any tool>
<<</TOOL_CALLS>>>
<<<CONTENT>>>
<your natural-language reply, possibly empty if a tool was called>
<<</CONTENT>>>

Each tool call is: {{"id":"call_<short-hex>","name":"<tool_name>","arguments":<JSON object>}}
Use double quotes everywhere. Do not wrap in markdown fences.

Available tools:
{tools_str}

Tool selection rules:
- {tc_rule}
- {parallel_rule}
""".strip()


def _response_format_instructions(rf: ResponseFormat | None) -> str:
    if not rf or rf.type == "text":
        return ""
    if rf.type == "json_object":
        return "Reply with a single valid JSON object only. No prose, no code fences."
    # json_schema
    schema = (rf.json_schema or {}).get("schema", {})
    return (
        "Reply with a single valid JSON object only, conforming to this JSON Schema:\n"
        + _json.dumps(schema, indent=2)
        + "\nNo prose, no code fences."
    )


def build_prompt(req: "ChatCompletionRequest") -> tuple[str, str]:
    """Pure: flatten messages[] into (system_prompt, user_prompt) for claude."""
    system_parts: list[str] = []
    transcript_parts: list[str] = []

    for msg in req.messages:
        rendered = _render_content(msg.content)
        if msg.role in ("system", "developer"):
            if rendered:
                system_parts.append(rendered)
        elif msg.role == "user":
            transcript_parts.append(f"[User]: {rendered}")
        elif msg.role == "assistant":
            line = f"[Assistant]: {rendered}"
            if msg.tool_calls:
                tc_str = _json.dumps([
                    {"id": tc.id, "name": tc.function.name,
                     "arguments": _json.loads(tc.function.arguments) if tc.function.arguments else {}}
                    for tc in msg.tool_calls
                ])
                line = f"[Assistant]: <<<TOOL_CALLS>>>{tc_str}<<</TOOL_CALLS>><<<CONTENT>>>{rendered}<<</CONTENT>>>"
            transcript_parts.append(line)
        elif msg.role == "tool":
            transcript_parts.append(f"[Tool {msg.tool_call_id} result]: {rendered}")

    sys_prompt = "\n\n".join(system_parts)
    if req.tools:
        sys_prompt = (sys_prompt + "\n\n" if sys_prompt else "") + _envelope_instructions(req)
    rf_instr = _response_format_instructions(req.response_format)
    if rf_instr:
        sys_prompt = (sys_prompt + "\n\n" if sys_prompt else "") + rf_instr

    user_prompt = "\n".join(transcript_parts) + "\n[Assistant]: "
    return sys_prompt, user_prompt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v -k build_prompt`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES build_prompt — message flattening + envelope/response-format instructions"
```

---

### Task 9: `parse_envelope_stream` — sentinel state machine

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
async def _drain(gen):
    out = []
    async for ev in gen:
        out.append(ev)
    return out


async def _from_chunks(chunks):
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_parse_passthrough_when_tools_off():
    from examples.openai_server import parse_envelope_stream
    events = await _drain(parse_envelope_stream(_from_chunks(["hello ", "world"]), tools_present=False))
    kinds = [e["kind"] for e in events]
    assert kinds == ["text_delta", "text_delta", "finish"]
    assert events[0]["text"] == "hello "
    assert events[1]["text"] == "world"
    assert events[-1]["reason"] == "stop"


@pytest.mark.asyncio
async def test_parse_envelope_empty_tool_calls_then_content():
    from examples.openai_server import parse_envelope_stream
    text = "<<<TOOL_CALLS>>>[]<<</TOOL_CALLS>>><<<CONTENT>>>hi there<<</CONTENT>>>"
    events = await _drain(parse_envelope_stream(_from_chunks([text]), tools_present=True))
    kinds = [e["kind"] for e in events]
    assert "text_delta" in kinds
    assert events[-1]["kind"] == "finish"
    assert events[-1]["reason"] == "stop"
    text_emitted = "".join(e["text"] for e in events if e["kind"] == "text_delta")
    assert text_emitted == "hi there"


@pytest.mark.asyncio
async def test_parse_envelope_one_tool_call():
    from examples.openai_server import parse_envelope_stream
    text = ('<<<TOOL_CALLS>>>[{"id":"call_1","name":"get_weather","arguments":{"city":"Paris"}}]'
            '<<</TOOL_CALLS>>><<<CONTENT>>><<</CONTENT>>>')
    events = await _drain(parse_envelope_stream(_from_chunks([text]), tools_present=True))
    tc_events = [e for e in events if e["kind"] == "tool_calls"]
    assert len(tc_events) == 1
    assert tc_events[0]["calls"] == [{"id": "call_1", "name": "get_weather",
                                       "arguments": {"city": "Paris"}}]
    assert events[-1]["reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_parse_envelope_tag_split_across_chunks():
    from examples.openai_server import parse_envelope_stream
    chunks = ["<<<TOOL", "_CALLS>>>[]<<", "</TOOL_CALLS>>>", "<<<CONTENT>>>ok<<</CONTENT>>>"]
    events = await _drain(parse_envelope_stream(_from_chunks(chunks), tools_present=True))
    text_emitted = "".join(e["text"] for e in events if e["kind"] == "text_delta")
    assert text_emitted == "ok"
    assert events[-1]["reason"] == "stop"


@pytest.mark.asyncio
async def test_parse_envelope_malformed_json_emits_error():
    from examples.openai_server import parse_envelope_stream
    text = "<<<TOOL_CALLS>>>[not json]<<</TOOL_CALLS>>><<<CONTENT>>>x<<</CONTENT>>>"
    events = await _drain(parse_envelope_stream(_from_chunks([text]), tools_present=True))
    error = next(e for e in events if e["kind"] == "error")
    assert error["code"] == "bridge_parse_error"


@pytest.mark.asyncio
async def test_parse_envelope_missing_envelope_emits_error():
    from examples.openai_server import parse_envelope_stream
    chunks = ["just plain text without any tags"]
    events = await _drain(parse_envelope_stream(_from_chunks(chunks), tools_present=True))
    error = next(e for e in events if e["kind"] == "error")
    assert error["code"] == "bridge_envelope_missing"


@pytest.mark.asyncio
async def test_parse_envelope_strips_markdown_fence():
    from examples.openai_server import parse_envelope_stream
    text = "<<<TOOL_CALLS>>>```json\n[]\n```<<</TOOL_CALLS>>><<<CONTENT>>>x<<</CONTENT>>>"
    events = await _drain(parse_envelope_stream(_from_chunks([text]), tools_present=True))
    assert not any(e["kind"] == "error" for e in events)


@pytest.mark.asyncio
async def test_parse_envelope_multiple_parallel_tool_calls():
    from examples.openai_server import parse_envelope_stream
    text = ('<<<TOOL_CALLS>>>['
            '{"id":"call_1","name":"a","arguments":{}},'
            '{"id":"call_2","name":"b","arguments":{"x":1}}'
            ']<<</TOOL_CALLS>>><<<CONTENT>>><<</CONTENT>>>')
    events = await _drain(parse_envelope_stream(_from_chunks([text]), tools_present=True))
    tc = next(e for e in events if e["kind"] == "tool_calls")
    assert len(tc["calls"]) == 2
    assert tc["calls"][0]["name"] == "a"
    assert tc["calls"][1]["name"] == "b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v -k parse_envelope`
Expected: 8 fails.

- [ ] **Step 3: Implement**

In `examples/openai_server.py`, near the imports add:

```python
from collections.abc import AsyncIterator
```

Append:

```python
TOOL_OPEN = "<<<TOOL_CALLS>>>"
TOOL_CLOSE = "<<</TOOL_CALLS>>>"
CONT_OPEN = "<<<CONTENT>>>"
CONT_CLOSE = "<<</CONTENT>>>"
TAGS = (TOOL_OPEN, TOOL_CLOSE, CONT_OPEN, CONT_CLOSE)
TAG_MAX = max(len(t) for t in TAGS)


def _strip_md_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```json"):
        s = s[len("```json"):].lstrip("\n")
    elif s.startswith("```"):
        s = s[3:].lstrip("\n")
    if s.endswith("```"):
        s = s[:-3].rstrip("\n")
    return s


async def parse_envelope_stream(
    chunks: AsyncIterator[str], tools_present: bool
) -> AsyncIterator[dict]:
    """Pure-ish state machine that consumes claude text chunks and emits parser events.

    Events are dicts of one of these shapes:
      {"kind": "text_delta", "text": "..."}
      {"kind": "tool_calls", "calls": [{"id":..., "name":..., "arguments": {...}}]}
      {"kind": "finish", "reason": "stop" | "tool_calls" | "length"}
      {"kind": "error", "code": "bridge_parse_error" | "bridge_envelope_missing", "message": "..."}
    """
    if not tools_present:
        async for chunk in chunks:
            if chunk:
                yield {"kind": "text_delta", "text": chunk}
        yield {"kind": "finish", "reason": "stop"}
        return

    state = 0
    buf = ""
    tool_buf = ""
    saw_any_tag = False
    finish_reason = "stop"
    tool_calls: list[dict] = []

    async for chunk in chunks:
        buf += chunk
        progressed = True
        while progressed:
            progressed = False
            if state == 0:
                idx = buf.find(TOOL_OPEN)
                if idx != -1:
                    buf = buf[idx + len(TOOL_OPEN):]
                    state = 1
                    saw_any_tag = True
                    progressed = True
                else:
                    # retain only the tail that could start a tag
                    if len(buf) > TAG_MAX:
                        buf = buf[-(TAG_MAX - 1):]
                    break
            elif state == 1:
                idx = buf.find(TOOL_CLOSE)
                if idx != -1:
                    tool_buf += buf[:idx]
                    buf = buf[idx + len(TOOL_CLOSE):]
                    state = 2
                    progressed = True
                    # parse tool_buf
                    raw = _strip_md_fence(tool_buf)
                    try:
                        parsed = _json.loads(raw) if raw else []
                        if not isinstance(parsed, list):
                            raise ValueError("tool_calls must be a JSON array")
                        tool_calls = parsed
                    except Exception as exc:
                        yield {"kind": "error",
                               "code": "bridge_parse_error",
                               "message": f"Malformed tool_calls JSON: {exc}"}
                        return
                else:
                    # retain everything except the tail that could be a partial close-tag
                    if len(buf) > TAG_MAX:
                        tool_buf += buf[:-(TAG_MAX - 1)]
                        buf = buf[-(TAG_MAX - 1):]
                    break
            elif state == 2:
                idx = buf.find(CONT_OPEN)
                if idx != -1:
                    buf = buf[idx + len(CONT_OPEN):]
                    state = 3
                    progressed = True
                else:
                    if len(buf) > TAG_MAX:
                        buf = buf[-(TAG_MAX - 1):]
                    break
            elif state == 3:
                idx = buf.find(CONT_CLOSE)
                if idx != -1:
                    if idx > 0:
                        yield {"kind": "text_delta", "text": buf[:idx]}
                    buf = buf[idx + len(CONT_CLOSE):]
                    state = 4
                    progressed = True
                else:
                    if len(buf) > TAG_MAX:
                        emit = buf[:-(TAG_MAX - 1)]
                        buf = buf[-(TAG_MAX - 1):]
                        if emit:
                            yield {"kind": "text_delta", "text": emit}
                    break
            else:  # state 4: done, drain
                buf = ""
                break

    # End of stream cleanup
    if state == 3 and buf:
        # CONTENT close never arrived — emit remaining buffer as content
        yield {"kind": "text_delta", "text": buf}

    if not saw_any_tag:
        yield {"kind": "error",
               "code": "bridge_envelope_missing",
               "message": "Upstream model did not emit the required <<<TOOL_CALLS>>> envelope"}
        return

    # Emit tool_calls if any
    if tool_calls:
        # Normalize: ensure each call has id/name/arguments
        normalized = []
        for c in tool_calls:
            if not isinstance(c, dict) or "name" not in c:
                yield {"kind": "error",
                       "code": "bridge_parse_error",
                       "message": "tool_call missing required field: name"}
                return
            normalized.append({
                "id": c.get("id") or f"call_{secrets.token_hex(6)}",
                "name": c["name"],
                "arguments": c.get("arguments", {}),
            })
        yield {"kind": "tool_calls", "calls": normalized}
        finish_reason = "tool_calls"

    yield {"kind": "finish", "reason": finish_reason}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v -k parse_envelope`
Expected: 8 passed.

- [ ] **Step 5: Run the entire test file to ensure no regression**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: all green so far.

- [ ] **Step 6: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES sentinel-envelope parser (state machine over claude text deltas)"
```

---

### Task 10: `drive_claude` — cckit driver with injectable factory

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

The driver is a thin async generator wrapping `cckit.CustomAgent.stream_execute`. We expose an injection seam so tests can plug a fake.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
class _FakeAgent:
    """Fake CustomAgent for tests: yields the given chunks then a fake ResultEvent."""

    def __init__(self, chunks, usage_raw=None):
        self._chunks = chunks
        self._usage = usage_raw or {"input_tokens": 10, "output_tokens": 20,
                                    "cache_read_input_tokens": 0,
                                    "cache_creation_input_tokens": 0}

    async def stream_execute(self, prompt):
        from cckit import TextChunkEvent, ResultEvent
        for c in self._chunks:
            yield TextChunkEvent(text=c)
        yield ResultEvent(raw={"usage": self._usage}, result="", session_id="fake")


def _make_factory(chunks, usage_raw=None):
    def _factory(req):
        return _FakeAgent(chunks, usage_raw)
    return _factory


@pytest.mark.asyncio
async def test_drive_claude_yields_text_and_resolves_usage():
    import asyncio
    from examples.openai_server import drive_claude, set_agent_factory, _OCESConfig
    set_agent_factory(_make_factory(["hello ", "world"]))
    final_usage = asyncio.get_event_loop().create_future()
    out = []
    async for chunk in drive_claude(_req(), final_usage):
        out.append(chunk)
    assert out == ["hello ", "world"]
    assert final_usage.done()
    assert final_usage.result()["prompt_tokens"] == 10
    set_agent_factory(None)  # reset


@pytest.mark.asyncio
async def test_drive_claude_default_factory_constructs_real_agent(monkeypatch):
    """The default factory should construct a real cckit.CustomAgent — we just verify it doesn't crash to import."""
    from examples.openai_server import _default_agent_factory
    factory = _default_agent_factory
    agent = factory(_req())
    # Don't call stream_execute (would spawn claude). Just check the type.
    from cckit import CustomAgent
    assert isinstance(agent, CustomAgent)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v -k drive_claude`
Expected: 2 fails.

- [ ] **Step 3: Implement**

In `examples/openai_server.py`, near the top imports add:

```python
from cckit import CLI, CLIConfig, CustomAgent, ResultEvent, TextChunkEvent
```

Append:

```python
def _default_agent_factory(req: "ChatCompletionRequest") -> CustomAgent:
    cli = CLI(config=CLIConfig(
        binary_path=_OCESConfig.binary_path,
        extra_flags=["--tools", "", "--include-partial-messages"],
    ))
    sys_prompt, _ = build_prompt(req)
    return CustomAgent(
        cli=cli,
        system_prompt=sys_prompt,
        bare=True,
        model=resolve_claude_model(req.model),
    )


_agent_factory = _default_agent_factory


def set_agent_factory(factory):
    """Test seam: pass a callable (req) -> agent-with-stream_execute. Pass None to reset."""
    global _agent_factory
    _agent_factory = factory if factory is not None else _default_agent_factory


async def drive_claude(req: "ChatCompletionRequest", final_usage) -> AsyncIterator[str]:
    """Yield raw text chunks from claude; resolve final_usage on ResultEvent."""
    _, user_prompt = build_prompt(req)
    agent = _agent_factory(req)
    async for event in agent.stream_execute(user_prompt):
        if isinstance(event, TextChunkEvent):
            if event.text:
                yield event.text
        elif isinstance(event, ResultEvent):
            if not final_usage.done():
                final_usage.set_result(map_usage(event.raw.get("usage", {})))
    if not final_usage.done():
        final_usage.set_result(map_usage({}))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v -k drive_claude`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES drive_claude — cckit driver with injectable agent factory"
```

---

### Task 11: `stream_openai` — SSE chunk emitter

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
def _parse_sse(text: str) -> list[dict]:
    """Parse an SSE response into list of decoded JSON chunks (excluding [DONE])."""
    out = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        body = line[len("data:"):].strip()
        if body == "[DONE]":
            continue
        out.append(_json.loads(body))
    return out


@pytest.mark.asyncio
async def test_stream_openai_text_only():
    import asyncio
    from examples.openai_server import stream_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 1, "completion_tokens": 2,
                            "total_tokens": 3,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    parser = parse_envelope_stream(_from_chunks(["hi ", "there"]), tools_present=False)
    sse = ""
    async for line in stream_openai(parser, "chatcmpl-x", "sonnet", 1700000000, False, final_usage):
        sse += line
    chunks = _parse_sse(sse)
    # First chunk: role assistant
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    # Middle chunks: content deltas
    contents = [c["choices"][0]["delta"].get("content")
                for c in chunks[1:] if "content" in c["choices"][0]["delta"]]
    assert "".join(c for c in contents if c) == "hi there"
    # Final chunk: finish_reason stop
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert sse.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_stream_openai_with_usage():
    import asyncio
    from examples.openai_server import stream_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 5, "completion_tokens": 7,
                            "total_tokens": 12,
                            "prompt_tokens_details": {"cached_tokens": 1}})
    parser = parse_envelope_stream(_from_chunks(["x"]), tools_present=False)
    sse = ""
    async for line in stream_openai(parser, "chatcmpl-x", "sonnet", 1700000000, True, final_usage):
        sse += line
    chunks = _parse_sse(sse)
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"]["prompt_tokens"] == 5
    assert usage_chunks[0]["usage"]["total_tokens"] == 12


@pytest.mark.asyncio
async def test_stream_openai_tool_calls_emit_two_chunks_per_call():
    import asyncio
    from examples.openai_server import stream_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    text = ('<<<TOOL_CALLS>>>[{"id":"call_1","name":"f","arguments":{"a":1}}]<<</TOOL_CALLS>>>'
            '<<<CONTENT>>><<</CONTENT>>>')
    parser = parse_envelope_stream(_from_chunks([text]), tools_present=True)
    sse = ""
    async for line in stream_openai(parser, "chatcmpl-x", "sonnet", 1700000000, False, final_usage):
        sse += line
    chunks = _parse_sse(sse)
    tc_deltas = [c for c in chunks if "tool_calls" in c["choices"][0].get("delta", {})]
    assert len(tc_deltas) == 2  # header chunk + args chunk
    assert tc_deltas[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"
    assert tc_deltas[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "f"
    assert tc_deltas[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == ""
    assert tc_deltas[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_stream_openai_error_event_terminates_with_error_chunk():
    import asyncio
    from examples.openai_server import stream_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    parser = parse_envelope_stream(_from_chunks(["plain text no envelope"]), tools_present=True)
    sse = ""
    async for line in stream_openai(parser, "chatcmpl-x", "sonnet", 1700000000, False, final_usage):
        sse += line
    chunks = _parse_sse(sse)
    # Find the error chunk
    err_chunks = [c for c in chunks if c.get("error")]
    assert len(err_chunks) == 1
    assert err_chunks[0]["choices"][0]["finish_reason"] == "error"
    assert err_chunks[0]["error"]["code"] == "bridge_envelope_missing"
    assert sse.endswith("data: [DONE]\n\n")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v -k stream_openai`
Expected: 4 fails.

- [ ] **Step 3: Implement**

In `examples/openai_server.py`, append:

```python
def _sse_chunk(payload: dict) -> str:
    return "data: " + _json.dumps(payload, separators=(",", ":")) + "\n\n"


def _base_chunk(req_id: str, model: str, created: int) -> dict:
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }


async def stream_openai(
    parser_events: AsyncIterator[dict],
    req_id: str,
    model: str,
    created: int,
    include_usage: bool,
    final_usage,
) -> AsyncIterator[str]:
    """Yield SSE 'data: ...\\n\\n' lines, terminating with 'data: [DONE]\\n\\n'."""
    # Opening role chunk
    first = _base_chunk(req_id, model, created)
    first["choices"][0]["delta"] = {"role": "assistant"}
    yield _sse_chunk(first)

    finish_reason: str | None = None
    error: dict | None = None

    async for ev in parser_events:
        if ev["kind"] == "text_delta":
            chunk = _base_chunk(req_id, model, created)
            chunk["choices"][0]["delta"] = {"content": ev["text"]}
            yield _sse_chunk(chunk)
        elif ev["kind"] == "tool_calls":
            for i, call in enumerate(ev["calls"]):
                # Header chunk
                header = _base_chunk(req_id, model, created)
                header["choices"][0]["delta"] = {"tool_calls": [{
                    "index": i,
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": ""},
                }]}
                yield _sse_chunk(header)
                # Args chunk
                args_chunk = _base_chunk(req_id, model, created)
                args_chunk["choices"][0]["delta"] = {"tool_calls": [{
                    "index": i,
                    "function": {"arguments": _json.dumps(call["arguments"])},
                }]}
                yield _sse_chunk(args_chunk)
        elif ev["kind"] == "finish":
            finish_reason = ev["reason"]
        elif ev["kind"] == "error":
            error = ev
            break

    # Final delta chunk: either finish_reason or error
    if error is not None:
        err_chunk = _base_chunk(req_id, model, created)
        err_chunk["choices"][0]["finish_reason"] = "error"
        err_chunk["error"] = {"message": error["message"], "type": "server_error",
                              "code": error["code"]}
        yield _sse_chunk(err_chunk)
    else:
        final = _base_chunk(req_id, model, created)
        final["choices"][0]["finish_reason"] = finish_reason or "stop"
        yield _sse_chunk(final)

        if include_usage:
            usage = await final_usage if not final_usage.done() else final_usage.result()
            usage_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            }
            yield _sse_chunk(usage_chunk)

    yield "data: [DONE]\n\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v -k stream_openai`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES stream_openai — SSE chunk emitter (text/tool_calls/error/usage)"
```

---

### Task 12: `collect_openai` — non-streaming response builder

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
@pytest.mark.asyncio
async def test_collect_openai_text():
    import asyncio
    from examples.openai_server import collect_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    parser = parse_envelope_stream(_from_chunks(["hi ", "there"]), tools_present=False)
    body = await collect_openai(parser, "chatcmpl-x", "sonnet", 1700000000, final_usage)
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "hi there"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"].get("tool_calls") is None
    assert body["usage"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_collect_openai_tool_calls_omits_content():
    import asyncio
    from examples.openai_server import collect_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    text = ('<<<TOOL_CALLS>>>[{"id":"c1","name":"f","arguments":{"x":1}}]<<</TOOL_CALLS>>>'
            '<<<CONTENT>>><<</CONTENT>>>')
    parser = parse_envelope_stream(_from_chunks([text]), tools_present=True)
    body = await collect_openai(parser, "chatcmpl-x", "sonnet", 1700000000, final_usage)
    msg = body["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["id"] == "c1"
    assert msg["tool_calls"][0]["function"]["name"] == "f"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    assert body["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_collect_openai_error_raises_http_exception():
    import asyncio
    from examples.openai_server import collect_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    parser = parse_envelope_stream(_from_chunks(["no envelope"]), tools_present=True)
    with pytest.raises(HTTPException) as exc:
        await collect_openai(parser, "chatcmpl-x", "sonnet", 1700000000, final_usage)
    assert exc.value.status_code == 502
    assert exc.value.detail["error"]["code"] == "bridge_envelope_missing"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v -k collect_openai`
Expected: 3 fails.

- [ ] **Step 3: Implement**

In `examples/openai_server.py`, append:

```python
SYSTEM_FINGERPRINT = "cckit-oces-0.1"


async def collect_openai(
    parser_events: AsyncIterator[dict],
    req_id: str,
    model: str,
    created: int,
    final_usage,
) -> dict:
    """Walk parser_events to completion and build a non-streaming ChatCompletionResponse."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = "stop"

    async for ev in parser_events:
        if ev["kind"] == "text_delta":
            text_parts.append(ev["text"])
        elif ev["kind"] == "tool_calls":
            tool_calls = ev["calls"]
        elif ev["kind"] == "finish":
            finish_reason = ev["reason"]
        elif ev["kind"] == "error":
            raise HTTPException(502, _err("server_error", ev["message"], code=ev["code"]))

    usage = await final_usage if not final_usage.done() else final_usage.result()
    content_text = "".join(text_parts)

    message: dict = {"role": "assistant"}
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": _json.dumps(c["arguments"])}}
            for c in tool_calls
        ]
    else:
        message["content"] = content_text

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v -k collect_openai`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES collect_openai — non-streaming response builder"
```

---

### Task 13: FastAPI route + global exception handler

**Files:**
- Modify: `examples/openai_server.py`
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
def _client(api_key="testkey"):
    from fastapi.testclient import TestClient
    from examples.openai_server import app, _OCESConfig
    _OCESConfig.api_key = api_key
    return TestClient(app)


def _auth(key="testkey"):
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def test_endpoint_happy_non_streaming(monkeypatch):
    from examples.openai_server import set_agent_factory
    set_agent_factory(_make_factory(["hello ", "world"]))
    try:
        client = _client()
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "hello world"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["completion_tokens"] == 20
    finally:
        set_agent_factory(None)


def test_endpoint_alias_route():
    from examples.openai_server import set_agent_factory
    set_agent_factory(_make_factory(["x"]))
    try:
        client = _client()
        r = client.post("/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
    finally:
        set_agent_factory(None)


def test_endpoint_streaming(monkeypatch):
    from examples.openai_server import set_agent_factory
    set_agent_factory(_make_factory(["hi ", "there"]))
    try:
        client = _client()
        with client.stream("POST", "/v1/chat/completions", headers=_auth(),
                           json={"model": "sonnet",
                                 "messages": [{"role": "user", "content": "x"}],
                                 "stream": True}) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            body = b"".join(r.iter_bytes()).decode()
            chunks = _parse_sse(body)
            assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
            assert body.endswith("data: [DONE]\n\n")
    finally:
        set_agent_factory(None)


def test_endpoint_missing_bearer():
    client = _client()
    r = client.post("/v1/chat/completions", headers={"Content-Type": "application/json"},
                    json={"model": "sonnet",
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_endpoint_wrong_bearer():
    client = _client()
    r = client.post("/v1/chat/completions", headers=_auth("wrong"),
                    json={"model": "sonnet",
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401


def test_endpoint_validation_400():
    client = _client()
    r = client.post("/v1/chat/completions", headers=_auth(),
                    json={"model": "sonnet", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_endpoint_pydantic_validation_returns_openai_envelope():
    client = _client()
    r = client.post("/v1/chat/completions", headers=_auth(),
                    json={"model": "sonnet"})  # missing messages
    assert r.status_code in (400, 422)
    assert "error" in r.json()
    assert r.json()["error"]["type"] == "invalid_request_error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_server.py -v -k endpoint`
Expected: 7 fails.

- [ ] **Step 3: Implement route, error handler, and uvicorn launcher**

In `examples/openai_server.py`, near the top imports add:

```python
import asyncio
import logging
import time
from fastapi import Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from cckit.utils.errors import AuthError, CLIError, ParseError, TimeoutError as CckitTimeout

logger = logging.getLogger("oces")

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
```

Replace the existing `app = FastAPI(title="OCES", version="0.1.0")` with:

```python
app = FastAPI(title="OCES", version="0.1.0")


@app.exception_handler(RequestValidationError)
async def _pydantic_handler(_request: Request, exc: RequestValidationError):
    msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    return JSONResponse(_err("invalid_request_error", msg or "validation failed"), status_code=400)


@app.exception_handler(HTTPException)
async def _http_handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(detail, status_code=exc.status_code)
    return JSONResponse(_err("server_error", str(detail)), status_code=exc.status_code)


@app.exception_handler(Exception)
async def _global_handler(_request: Request, exc: Exception):
    if isinstance(exc, AuthError):
        return JSONResponse(_err("server_error", str(exc), code="claude_auth_unavailable"), 503)
    if isinstance(exc, CckitTimeout):
        return JSONResponse(_err("server_error", str(exc), code="claude_timeout"), 504)
    if isinstance(exc, ParseError):
        return JSONResponse(_err("server_error", str(exc), code="claude_parse_error"), 502)
    if isinstance(exc, CLIError):
        return JSONResponse(_err("server_error", str(exc), code="claude_cli_failed"), 502)
    logger.exception("Unhandled exception in OCES request")
    return JSONResponse(_err("server_error", "Internal server error", code="unexpected"), 500)


async def _handle_chat(req: ChatCompletionRequest):
    err = validate_request(req)
    if err:
        raise err
    final_usage = asyncio.get_event_loop().create_future()
    chunks = drive_claude(req, final_usage)
    parser = parse_envelope_stream(chunks, tools_present=bool(req.tools))
    rid = make_request_id()
    created = int(time.time())
    if req.stream:
        include_usage = bool(req.stream_options and req.stream_options.include_usage)
        return StreamingResponse(
            stream_openai(parser, rid, req.model, created, include_usage, final_usage),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    body = await collect_openai(parser, rid, req.model, created, final_usage)
    return JSONResponse(body)


@app.post("/chat/completions")
async def chat_completions_root(req: ChatCompletionRequest, _=Depends(verify_bearer)):
    return await _handle_chat(req)


@app.post("/v1/chat/completions")
async def chat_completions_v1(req: ChatCompletionRequest, _=Depends(verify_bearer)):
    return await _handle_chat(req)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v -k endpoint`
Expected: 7 passed.

- [ ] **Step 5: Run the full test file**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add examples/openai_server.py tests/test_openai_server.py
git commit -m "feat(examples): OCES FastAPI route + global exception handler (cckit→OpenAI error mapping)"
```

---

### Task 14: cckit-error mapping endpoint tests + stream-options usage chunk

**Files:**
- Modify: `tests/test_openai_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_openai_server.py`:

```python
class _RaisingAgent:
    def __init__(self, exc): self._exc = exc
    async def stream_execute(self, prompt):
        raise self._exc
        yield  # unreachable, makes it an async gen


def _raising_factory(exc):
    def _f(req): return _RaisingAgent(exc)
    return _f


def test_endpoint_cckit_timeout_returns_504():
    from examples.openai_server import set_agent_factory
    from cckit.utils.errors import TimeoutError as CckitTimeout
    set_agent_factory(_raising_factory(CckitTimeout("timed out after 30s")))
    try:
        client = _client()
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 504
        assert r.json()["error"]["code"] == "claude_timeout"
    finally:
        set_agent_factory(None)


def test_endpoint_cckit_auth_error_returns_503():
    from examples.openai_server import set_agent_factory
    from cckit.utils.errors import AuthError
    set_agent_factory(_raising_factory(AuthError("not logged in")))
    try:
        client = _client()
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "claude_auth_unavailable"
    finally:
        set_agent_factory(None)


def test_endpoint_cckit_cli_error_returns_502():
    from examples.openai_server import set_agent_factory
    from cckit.utils.errors import CLIError
    set_agent_factory(_raising_factory(CLIError("exit 1", exit_code=1)))
    try:
        client = _client()
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "claude_cli_failed"
    finally:
        set_agent_factory(None)


def test_endpoint_streaming_with_include_usage():
    from examples.openai_server import set_agent_factory
    set_agent_factory(_make_factory(["x"]))
    try:
        client = _client()
        with client.stream("POST", "/v1/chat/completions", headers=_auth(),
                           json={"model": "sonnet",
                                 "messages": [{"role": "user", "content": "hi"}],
                                 "stream": True,
                                 "stream_options": {"include_usage": True}}) as r:
            body = b"".join(r.iter_bytes()).decode()
            chunks = _parse_sse(body)
            usage_chunks = [c for c in chunks if c.get("usage")]
            assert len(usage_chunks) == 1
            assert usage_chunks[0]["usage"]["completion_tokens"] == 20
    finally:
        set_agent_factory(None)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_server.py -v -k "endpoint and (timeout or auth_error or cli_error or include_usage)"`
Expected: 4 passed (the existing impl from Task 13 should already cover these — if any fail, fix in `examples/openai_server.py`).

- [ ] **Step 3: Run the full test file**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_openai_server.py
git commit -m "test(examples): OCES endpoint tests for cckit error mapping + streaming usage chunk"
```

---

### Task 15: uvicorn launcher block

**Files:**
- Modify: `examples/openai_server.py`

- [ ] **Step 1: Add the `if __name__ == "__main__"` block at the bottom of the file**

Append:

```python
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    if not _OCESConfig.api_key:
        print("ERROR: API_KEY env var not set. Refusing to start.", flush=True)
        raise SystemExit(2)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("OCES starting on http://%s:%d (claude binary: %s)",
                host, port, _OCESConfig.binary_path)
    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("LOGLEVEL", "info").lower())
```

- [ ] **Step 2: Smoke-test the launcher**

Run: `API_KEY=sk-test-xxx PORT=18765 uv run python -c "
import os, threading, time, urllib.request
os.environ['API_KEY']='sk-test-xxx'
os.environ['PORT']='18765'
import uvicorn
from examples.openai_server import app
def serve(): uvicorn.run(app, host='127.0.0.1', port=18765, log_level='warning')
t = threading.Thread(target=serve, daemon=True); t.start(); time.sleep(1)
req = urllib.request.Request('http://127.0.0.1:18765/v1/chat/completions',
    data=b'{\"model\":\"x\",\"messages\":[]}',
    headers={'Authorization':'Bearer sk-test-xxx','Content-Type':'application/json'})
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print('OK status=', e.code, 'body=', e.read()[:200])
"`

Expected: prints `OK status= 400 body= ...invalid_request_error...` (proves the launcher boots and routes).

- [ ] **Step 3: Commit**

```bash
git add examples/openai_server.py
git commit -m "feat(examples): OCES uvicorn launcher (HOST/PORT/API_KEY env vars)"
```

---

### Task 16: Harness conformance — openai SDK roundtrip + httpx + golden snapshot

**Files:**
- Modify: `tests/test_openai_server.py`
- Create: `tests/fixtures/oces_stream_golden.txt`

This is the **load-bearing** test category per the spec — wire compatibility with real OpenAI clients.

- [ ] **Step 1: Write failing openai-SDK roundtrip tests**

Append to `tests/test_openai_server.py`:

```python
def test_harness_openai_sdk_non_streaming():
    """Verify the official openai>=1.0 SDK can call our endpoint without errors."""
    import httpx
    from openai import OpenAI
    from examples.openai_server import app, set_agent_factory
    set_agent_factory(_make_factory(["the answer is 42"]))
    try:
        from fastapi.testclient import TestClient
        tc = TestClient(app)
        # Build an httpx transport that pipes to our TestClient
        transport = httpx.WSGITransport(app=app) if False else None
        # Easier: use TestClient as the http_client base_url proxy
        client = OpenAI(base_url=str(tc.base_url) + "/v1",
                        api_key="testkey",
                        http_client=httpx.Client(transport=httpx.ASGITransport(app=app),
                                                 base_url="http://testserver"))
        resp = client.chat.completions.create(
            model="sonnet",
            messages=[{"role": "user", "content": "what is the answer?"}],
        )
        assert resp.choices[0].message.content == "the answer is 42"
        assert resp.choices[0].finish_reason == "stop"
    finally:
        set_agent_factory(None)


def test_harness_openai_sdk_streaming():
    import httpx
    from openai import OpenAI
    from examples.openai_server import app, set_agent_factory, _OCESConfig
    _OCESConfig.api_key = "testkey"
    set_agent_factory(_make_factory(["foo ", "bar ", "baz"]))
    try:
        client = OpenAI(base_url="http://testserver/v1",
                        api_key="testkey",
                        http_client=httpx.Client(transport=httpx.ASGITransport(app=app),
                                                 base_url="http://testserver"))
        stream = client.chat.completions.create(
            model="sonnet",
            messages=[{"role": "user", "content": "x"}],
            stream=True,
        )
        chunks = list(stream)
        text = "".join((c.choices[0].delta.content or "") for c in chunks if c.choices)
        assert text == "foo bar baz"
        # Last chunk should have finish_reason
        assert any(c.choices and c.choices[0].finish_reason == "stop" for c in chunks)
    finally:
        set_agent_factory(None)


def test_harness_openai_sdk_tool_calls():
    import httpx
    from openai import OpenAI
    from examples.openai_server import app, set_agent_factory, _OCESConfig
    _OCESConfig.api_key = "testkey"
    text = ('<<<TOOL_CALLS>>>[{"id":"call_1","name":"get_weather",'
            '"arguments":{"city":"Paris"}}]<<</TOOL_CALLS>>>'
            '<<<CONTENT>>><<</CONTENT>>>')
    set_agent_factory(_make_factory([text]))
    try:
        client = OpenAI(base_url="http://testserver/v1",
                        api_key="testkey",
                        http_client=httpx.Client(transport=httpx.ASGITransport(app=app),
                                                 base_url="http://testserver"))
        resp = client.chat.completions.create(
            model="sonnet",
            messages=[{"role": "user", "content": "weather?"}],
            tools=[{"type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        )
        assert resp.choices[0].finish_reason == "tool_calls"
        tc = resp.choices[0].message.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert _json.loads(tc.function.arguments) == {"city": "Paris"}
    finally:
        set_agent_factory(None)
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_openai_server.py -v -k harness`
Expected: 3 passed (no implementation changes needed — exercises existing endpoint).

- [ ] **Step 3: Generate the golden SSE snapshot**

Create the fixture by running the streaming endpoint with a fixed seed and writing output. Add this script as a doctest-style file or just generate once and freeze:

Run:
```bash
uv run python -c "
from fastapi.testclient import TestClient
from examples.openai_server import app, set_agent_factory, _OCESConfig

class _FA:
    def __init__(self, c): self.c = c
    async def stream_execute(self, p):
        from cckit import TextChunkEvent, ResultEvent
        for x in self.c:
            yield TextChunkEvent(text=x)
        yield ResultEvent(raw={'usage':{'input_tokens':1,'output_tokens':2,'cache_read_input_tokens':0,'cache_creation_input_tokens':0}}, result='', session_id='s')

_OCESConfig.api_key = 'testkey'
set_agent_factory(lambda req: _FA(['hello ', 'world']))
client = TestClient(app)
with client.stream('POST', '/v1/chat/completions',
                   headers={'Authorization':'Bearer testkey','Content-Type':'application/json'},
                   json={'model':'sonnet',
                         'messages':[{'role':'user','content':'hi'}],
                         'stream':True}) as r:
    body = b''.join(r.iter_bytes()).decode()

# Mask non-deterministic fields
import re, json as J
masked = re.sub(r'\"id\":\"chatcmpl-[0-9a-f]{24}\"', '\"id\":\"chatcmpl-FIXED\"', body)
masked = re.sub(r'\"created\":\d+', '\"created\":1700000000', masked)
with open('tests/fixtures/oces_stream_golden.txt', 'w') as f:
    f.write(masked)
print('Wrote golden fixture, length=', len(masked))
"
```

Expected: prints `Wrote golden fixture, length= ...`.

- [ ] **Step 4: Add the golden test**

Append to `tests/test_openai_server.py`:

```python
def test_harness_golden_sse_snapshot():
    """Byte-for-byte SSE snapshot — fails on any drift."""
    import re
    from pathlib import Path
    from examples.openai_server import set_agent_factory, _OCESConfig
    _OCESConfig.api_key = "testkey"
    set_agent_factory(_make_factory(["hello ", "world"],
                                     usage_raw={"input_tokens": 1, "output_tokens": 2,
                                                "cache_read_input_tokens": 0,
                                                "cache_creation_input_tokens": 0}))
    try:
        client = _client()
        with client.stream("POST", "/v1/chat/completions", headers=_auth(),
                           json={"model": "sonnet",
                                 "messages": [{"role": "user", "content": "hi"}],
                                 "stream": True}) as r:
            body = b"".join(r.iter_bytes()).decode()
        masked = re.sub(r'"id":"chatcmpl-[0-9a-f]{24}"', '"id":"chatcmpl-FIXED"', body)
        masked = re.sub(r'"created":\d+', '"created":1700000000', masked)
        golden = Path("tests/fixtures/oces_stream_golden.txt").read_text()
        assert masked == golden, (
            "SSE output drift detected. To accept new output as canonical, regenerate "
            "tests/fixtures/oces_stream_golden.txt with the snippet in plan Task 16 step 3."
        )
    finally:
        set_agent_factory(None)
```

- [ ] **Step 5: Run all harness tests**

Run: `uv run pytest tests/test_openai_server.py -v -k harness`
Expected: 4 passed.

- [ ] **Step 6: Run the full test file**

Run: `uv run pytest tests/test_openai_server.py -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add tests/test_openai_server.py tests/fixtures/oces_stream_golden.txt
git commit -m "test(examples): OCES harness conformance — openai SDK roundtrip + golden SSE snapshot"
```

---

### Task 17: Integration test (opt-in real claude)

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_openai_server_real.py`

- [ ] **Step 1: Create the package marker**

Run: `touch tests/integration/__init__.py`

- [ ] **Step 2: Write the integration test**

Create `tests/integration/test_openai_server_real.py`:

```python
"""Opt-in integration test for OCES against a real claude binary.

Run with: pytest -m integration tests/integration/test_openai_server_real.py
Skipped by default (excluded via addopts in pyproject.toml).

Requires:
- claude CLI installed at ~/.local/bin/claude (or override CLAUDE_BINARY env var)
- claude logged in (oauth or ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import os
import threading
import time

import httpx
import pytest
import uvicorn
from openai import OpenAI


@pytest.fixture
def server():
    """Start OCES on a random port in a background thread; yield (host, port, api_key)."""
    api_key = "test-integration-key-" + os.urandom(8).hex()
    os.environ["API_KEY"] = api_key
    # Re-import to pick up the env var
    import importlib
    from examples import openai_server as oces
    importlib.reload(oces)

    port = 18765
    host = "127.0.0.1"
    cfg = uvicorn.Config(oces.app, host=host, port=port, log_level="warning")
    server_obj = uvicorn.Server(cfg)
    t = threading.Thread(target=server_obj.run, daemon=True)
    t.start()
    # Wait for readiness
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=1.0) as c:
                c.get(f"http://{host}:{port}/")
                break
        except httpx.ConnectError:
            time.sleep(0.1)
    yield host, port, api_key
    server_obj.should_exit = True


@pytest.mark.integration
def test_real_claude_says_pong(server):
    host, port, api_key = server
    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)
    resp = client.chat.completions.create(
        model="sonnet",
        messages=[
            {"role": "system",
             "content": "Reply with exactly one word: pong. No punctuation, no other text."},
            {"role": "user", "content": "ready?"},
        ],
        max_tokens=20,
    )
    content = (resp.choices[0].message.content or "").lower()
    assert "pong" in content, f"Unexpected reply: {content!r}"
    assert resp.choices[0].finish_reason in ("stop", "length")


@pytest.mark.integration
def test_real_claude_streaming(server):
    host, port, api_key = server
    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)
    stream = client.chat.completions.create(
        model="sonnet",
        messages=[{"role": "user", "content": "Count: one, two, three. Just those three words."}],
        stream=True,
    )
    text = "".join((c.choices[0].delta.content or "") for c in stream if c.choices).lower()
    assert "one" in text and "two" in text and "three" in text
```

- [ ] **Step 3: Verify the test is excluded by default**

Run: `uv run pytest tests/integration/ -v`
Expected: 2 deselected (excluded by `addopts = "-m 'not integration'"`).

- [ ] **Step 4: Run it explicitly (only if claude is logged in — otherwise skip this step)**

Run: `uv run pytest -m integration tests/integration/test_openai_server_real.py -v`
Expected (when claude is logged in): 2 passed. Skip if claude isn't installed/logged in — that's fine, the gate is for CI-equivalent verification only.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_openai_server_real.py
git commit -m "test(examples): OCES opt-in integration test against real claude binary"
```

---

### Task 18: Final smoke — manual curl + openai SDK against running server

**Files:** none modified

- [ ] **Step 1: Start OCES on a known port**

In one terminal: `API_KEY=sk-smoke-1 PORT=18000 uv run python examples/openai_server.py`
Expected: prints `OCES starting on http://0.0.0.0:18000 ...`.

- [ ] **Step 2: Send a curl request (non-streaming)**

In another terminal:
```bash
curl -sS http://localhost:18000/v1/chat/completions \
  -H "Authorization: Bearer sk-smoke-1" -H "Content-Type: application/json" \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"say hi in one word"}]}' \
  | python -m json.tool
```
Expected: 200 JSON response with `choices[0].message.content` containing some word like "Hi" / "Hello".

- [ ] **Step 3: Send a curl request (streaming)**

```bash
curl -N http://localhost:18000/v1/chat/completions \
  -H "Authorization: Bearer sk-smoke-1" -H "Content-Type: application/json" \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"count 1 2 3"}],"stream":true}'
```
Expected: SSE stream ending with `data: [DONE]`.

- [ ] **Step 4: openai SDK against running server**

```bash
uv run python -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:18000/v1', api_key='sk-smoke-1')
print(c.chat.completions.create(model='sonnet',
    messages=[{'role':'user','content':'one-word reply: ok'}]).choices[0].message.content)
"
```
Expected: prints a one-word reply.

- [ ] **Step 5: Stop the server**

`Ctrl-C` in the first terminal.

- [ ] **Step 6: No commit** — manual verification only.

---

### Task 19: Final test sweep + done check

**Files:** none modified

- [ ] **Step 1: Run the full default test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass; integration tests are deselected (per pyproject `addopts`).

- [ ] **Step 2: Confirm no `claude_agent_sdk` in source**

Run: `! grep -r "claude_agent_sdk" examples/ tests/ cckit/`
Expected: prints nothing (negation makes exit-0 mean "no matches").

- [ ] **Step 3: Confirm spec done-criteria**

Per `docs/superpowers/specs/2026-05-07-openai-fastapi-example-design.md` §10:
- `uv pip install "cckit[examples]"` works → `uv sync --all-extras` succeeded in Task 1.
- `examples/openai_server.py` exists, single file, ≤900 LOC: `wc -l examples/openai_server.py` should report ≤900.
- All `tests/test_openai_server.py` pass: confirmed in Step 1.
- Golden snapshot matches: covered by `test_harness_golden_sse_snapshot`.
- Integration test passes against logged-in claude: confirmed in Task 17.4 (if claude was available).
- curl + openai-SDK smoke works: confirmed in Task 18.
- No `claude_agent_sdk`: confirmed in Step 2.

Run: `wc -l examples/openai_server.py`
Expected: line count ≤ 900. If over, the implementation drifted — review for over-abstraction; KISS principle.

- [ ] **Step 4: Final commit if anything was tweaked**

If steps 1–3 surfaced any minor fixes:

```bash
git add -A
git commit -m "chore(examples): final OCES tidy after smoke verification"
```

Otherwise: nothing to commit, plan is complete.

---

## Summary

19 tasks total. Each produces a green test suite and a focused commit. The implementation is fundamentally:
- 1 file (`examples/openai_server.py`)
- 1 main test file (`tests/test_openai_server.py`)
- 1 opt-in integration test (`tests/integration/test_openai_server_real.py`)
- 1 golden fixture (`tests/fixtures/oces_stream_golden.txt`)

Mission-critical wire compatibility is enforced by Task 16's three harness tests (openai SDK non-stream, openai SDK stream, openai SDK tool-calls) plus the byte-for-byte golden SSE snapshot. Any future change that breaks these will fail loudly.
