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
    - Tool argument values containing the literal string `<<</TOOL_CALLS>>>` will trigger
      a parse error. This sentinel collision is rare; a JSON-aware parser would be needed
      to fully resolve.

Tested with openai>=1.0, raw httpx, and curl.
"""
from __future__ import annotations

import hmac
import os
import re
import secrets
from collections.abc import AsyncIterator
from typing import Any, Literal, Union

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from cckit import CLI, CLIConfig, CustomAgent, ResultEvent, TextChunkEvent


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
                line = f"[Assistant]: <<<TOOL_CALLS>>>{tc_str}<<</TOOL_CALLS>>><<<CONTENT>>>{rendered}<<</CONTENT>>>"
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


app = FastAPI(title="OCES", version="0.1.0")


TOOL_OPEN = "<<<TOOL_CALLS>>>"
TOOL_CLOSE = "<<</TOOL_CALLS>>>"
CONT_OPEN = "<<<CONTENT>>>"
CONT_CLOSE = "<<</CONTENT>>>"
TAGS = (TOOL_OPEN, TOOL_CLOSE, CONT_OPEN, CONT_CLOSE)
TAG_MAX = max(len(t) for t in TAGS)
MAX_TOOL_BUF = 64 * 1024  # 64 KiB cap on accumulated tool_calls JSON before bailing


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
                        if len(tool_buf) > MAX_TOOL_BUF:
                            yield {"kind": "error",
                                   "code": "bridge_parse_error",
                                   "message": f"tool_calls JSON exceeded {MAX_TOOL_BUF} bytes without closing tag"}
                            return
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

    if state == 1:
        yield {"kind": "error",
               "code": "bridge_parse_error",
               "message": "Stream ended inside <<<TOOL_CALLS>>> block before <<</TOOL_CALLS>>>"}
        return

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
