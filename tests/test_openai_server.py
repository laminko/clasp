"""Tests for examples/openai_server.py (OCES)."""
import importlib

import pytest
from pydantic import ValidationError


def test_module_imports():
    mod = importlib.import_module("examples.openai_server")
    assert hasattr(mod, "app"), "FastAPI app must be exported as `app`"


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


from fastapi import HTTPException


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


@pytest.mark.asyncio
async def test_parse_envelope_truncated_in_tool_section_emits_error():
    """Stream ends inside <<<TOOL_CALLS>>> before <<</TOOL_CALLS>>> — must emit error."""
    from examples.openai_server import parse_envelope_stream
    chunks = ['<<<TOOL_CALLS>>>[{"name":"get_weather","arguments":{']
    events = await _drain(parse_envelope_stream(_from_chunks(chunks), tools_present=True))
    error = next((e for e in events if e["kind"] == "error"), None)
    assert error is not None
    assert error["code"] == "bridge_parse_error"
    assert "TOOL_CALLS" in error["message"]


@pytest.mark.asyncio
async def test_parse_envelope_oversized_tool_buf_caps_with_error():
    """Tool calls JSON section exceeding MAX_TOOL_BUF (64 KiB) must emit error, not OOM."""
    from examples.openai_server import parse_envelope_stream
    # 100 KiB of garbage inside tool section, no close tag
    chunks = ["<<<TOOL_CALLS>>>", "x" * (100 * 1024)]
    events = await _drain(parse_envelope_stream(_from_chunks(chunks), tools_present=True))
    error = next((e for e in events if e["kind"] == "error"), None)
    assert error is not None
    assert error["code"] == "bridge_parse_error"
    assert "exceeded" in error["message"] or "64" in error["message"]


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


def _parse_sse(text: str) -> list[dict]:
    """Parse an SSE response into list of decoded JSON chunks (excluding [DONE])."""
    import json as _json
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


class _RaisingAgent:
    def __init__(self, exc): self._exc = exc
    async def stream_execute(self, prompt):
        raise self._exc
        yield  # unreachable, makes it an async gen


def _raising_factory(exc):
    def _f(req): return _RaisingAgent(exc)
    return _f


def test_endpoint_cckit_timeout_returns_504():
    from fastapi.testclient import TestClient
    from examples.openai_server import app, _OCESConfig, set_agent_factory
    from cckit.utils.errors import TimeoutError as CckitTimeout
    set_agent_factory(_raising_factory(CckitTimeout("timed out after 30s")))
    try:
        _OCESConfig.api_key = "testkey"
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 504
        assert r.json()["error"]["code"] == "claude_timeout"
    finally:
        set_agent_factory(None)


def test_endpoint_cckit_auth_error_returns_503():
    from fastapi.testclient import TestClient
    from examples.openai_server import app, _OCESConfig, set_agent_factory
    from cckit.utils.errors import AuthError
    set_agent_factory(_raising_factory(AuthError("not logged in")))
    try:
        _OCESConfig.api_key = "testkey"
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "claude_auth_unavailable"
    finally:
        set_agent_factory(None)


def test_endpoint_cckit_cli_error_returns_502():
    from fastapi.testclient import TestClient
    from examples.openai_server import app, _OCESConfig, set_agent_factory
    from cckit.utils.errors import CLIError
    set_agent_factory(_raising_factory(CLIError("exit 1", exit_code=1)))
    try:
        _OCESConfig.api_key = "testkey"
        client = TestClient(app, raise_server_exceptions=False)
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
