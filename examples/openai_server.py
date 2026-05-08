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

import hmac
import os
import re
import secrets
from typing import Any, Literal, Union

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict


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


class _OCESConfig:
    api_key: str = os.environ.get("API_KEY", "")
    binary_path: str = os.environ.get("CLAUDE_BINARY", "~/.local/bin/claude")


def verify_bearer(authorization: str = Header(default="")) -> None:
    if not _OCESConfig.api_key:
        raise HTTPException(500, _err("server_error", "API_KEY not configured"))
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, _err("invalid_request_error", "Missing bearer token"))
    token = authorization[7:].strip()
    if not hmac.compare_digest(token, _OCESConfig.api_key):
        raise HTTPException(401, _err("invalid_request_error", "Invalid API key"))


app = FastAPI(title="OCES", version="0.1.0")
