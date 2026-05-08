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

import re
import secrets
from typing import Any, Literal, Union

from fastapi import FastAPI
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


app = FastAPI(title="OCES", version="0.1.0")
