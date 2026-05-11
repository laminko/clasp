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
async def test_parse_envelope_missing_falls_back_to_plain_text():
    """When tools_present=True but model replies with prose (no envelope), treat as plain text."""
    from examples.openai_server import parse_envelope_stream
    chunks = ["just plain text without any tags"]
    events = await _drain(parse_envelope_stream(_from_chunks(chunks), tools_present=True))
    assert not any(e["kind"] == "error" for e in events)
    text = "".join(e["text"] for e in events if e["kind"] == "text_delta")
    assert text == "just plain text without any tags"
    assert events[-1] == {"kind": "finish", "reason": "stop"}


@pytest.mark.asyncio
async def test_parse_envelope_missing_streamed_falls_back():
    """Same as above but text arrives across multiple chunks — still treated as plain."""
    from examples.openai_server import parse_envelope_stream
    chunks = ["The answer ", "is forty-", "two."]
    events = await _drain(parse_envelope_stream(_from_chunks(chunks), tools_present=True))
    assert not any(e["kind"] == "error" for e in events)
    text = "".join(e["text"] for e in events if e["kind"] == "text_delta")
    assert text == "The answer is forty-two."
    assert events[-1] == {"kind": "finish", "reason": "stop"}


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
    """Fake CustomAgent for tests: yields the given chunks then a fake ResultEvent.

    Records which entry point was used (stream_execute vs stream_execute_messages)
    and the args, so tests can assert routing.
    """

    def __init__(self, chunks, usage_raw=None):
        self._chunks = chunks
        self._usage = usage_raw or {"input_tokens": 10, "output_tokens": 20,
                                    "cache_read_input_tokens": 0,
                                    "cache_creation_input_tokens": 0}
        self.calls: list[tuple[str, object]] = []

    async def stream_execute(self, prompt):
        from cckit import TextChunkEvent, ResultEvent
        self.calls.append(("stream_execute", prompt))
        for c in self._chunks:
            yield TextChunkEvent(text=c)
        yield ResultEvent(raw={"usage": self._usage}, result="", session_id="fake")

    async def stream_execute_messages(self, messages):
        from cckit import TextChunkEvent, ResultEvent
        self.calls.append(("stream_execute_messages", messages))
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
async def test_drive_claude_raises_authentication_error_on_text_pattern():
    """When claude emits 'Not logged in...' as text, drive_claude raises AuthError early."""
    import asyncio
    from cckit.utils.errors import AuthError
    from examples.openai_server import drive_claude, set_agent_factory
    set_agent_factory(_make_factory(["Not logged in · Please run /login"]))
    final_usage = asyncio.get_event_loop().create_future()
    try:
        with pytest.raises(AuthError):
            async for _chunk in drive_claude(_req(), final_usage):
                pass
    finally:
        set_agent_factory(None)


@pytest.mark.asyncio
async def test_drive_claude_raises_authentication_error_on_result_is_error():
    """When ResultEvent.is_error=True with auth message, drive_claude raises AuthError."""
    import asyncio
    from cckit import ResultEvent, TextChunkEvent
    from cckit.utils.errors import AuthError
    from examples.openai_server import drive_claude, set_agent_factory

    class _ErrorResultAgent:
        async def stream_execute(self, prompt):
            yield ResultEvent(raw={"result": "Authentication failed",
                                    "is_error": True, "usage": {}},
                               result="Authentication failed", session_id="x",
                               is_error=True)

    set_agent_factory(lambda req: _ErrorResultAgent())
    final_usage = asyncio.get_event_loop().create_future()
    try:
        with pytest.raises(AuthError):
            async for _chunk in drive_claude(_req(), final_usage):
                pass
    finally:
        set_agent_factory(None)


@pytest.mark.asyncio
async def test_drive_claude_raises_cli_error_on_non_auth_result_is_error():
    """ResultEvent.is_error=True without auth markers raises CLIError (mapped to 502)."""
    import asyncio
    from cckit import ResultEvent
    from cckit.utils.errors import CLIError
    from examples.openai_server import drive_claude, set_agent_factory

    class _ErrorResultAgent:
        async def stream_execute(self, prompt):
            yield ResultEvent(raw={"result": "Rate limit exceeded",
                                    "is_error": True, "usage": {}},
                               result="Rate limit exceeded", session_id="x",
                               is_error=True)

    set_agent_factory(lambda req: _ErrorResultAgent())
    final_usage = asyncio.get_event_loop().create_future()
    try:
        with pytest.raises(CLIError):
            async for _chunk in drive_claude(_req(), final_usage):
                pass
    finally:
        set_agent_factory(None)


def test_endpoint_claude_not_authenticated_returns_503():
    """End-to-end: when claude binary is unauthenticated, OCES returns 503 (not 200)."""
    from fastapi.testclient import TestClient
    from examples.openai_server import app, _OCESConfig, set_agent_factory
    _OCESConfig.api_key = "testkey"
    set_agent_factory(_make_factory(["Not logged in · Please run /login"]))
    try:
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "claude_auth_unavailable"
        assert "logged in" in body["error"]["message"].lower()
    finally:
        set_agent_factory(None)


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
    """Real protocol violations (malformed JSON inside envelope) emit terminating error chunk."""
    import asyncio
    from examples.openai_server import stream_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    bad = "<<<TOOL_CALLS>>>[not valid json]<<</TOOL_CALLS>>><<<CONTENT>>>x<<</CONTENT>>>"
    parser = parse_envelope_stream(_from_chunks([bad]), tools_present=True)
    sse = ""
    async for line in stream_openai(parser, "chatcmpl-x", "sonnet", 1700000000, False, final_usage):
        sse += line
    chunks = _parse_sse(sse)
    err_chunks = [c for c in chunks if c.get("error")]
    assert len(err_chunks) == 1
    assert err_chunks[0]["choices"][0]["finish_reason"] == "error"
    assert err_chunks[0]["error"]["code"] == "bridge_parse_error"
    assert sse.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_stream_openai_plain_response_when_tools_present():
    """tools_present=True but model replies with prose: stream as plain text + stop."""
    import asyncio
    from examples.openai_server import stream_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    parser = parse_envelope_stream(_from_chunks(["Hi! ", "How can I help?"]), tools_present=True)
    sse = ""
    async for line in stream_openai(parser, "chatcmpl-x", "sonnet", 1700000000, False, final_usage):
        sse += line
    chunks = _parse_sse(sse)
    assert not any(c.get("error") for c in chunks)
    contents = [c["choices"][0]["delta"].get("content")
                for c in chunks if "content" in c["choices"][0].get("delta", {})]
    text = "".join(c for c in contents if c)
    assert text == "Hi! How can I help?"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
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
    """Real protocol violation (malformed envelope JSON) raises 502."""
    import asyncio
    from examples.openai_server import collect_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    bad = "<<<TOOL_CALLS>>>[not valid json]<<</TOOL_CALLS>>><<<CONTENT>>>x<<</CONTENT>>>"
    parser = parse_envelope_stream(_from_chunks([bad]), tools_present=True)
    with pytest.raises(HTTPException) as exc:
        await collect_openai(parser, "chatcmpl-x", "sonnet", 1700000000, final_usage)
    assert exc.value.status_code == 502
    assert exc.value.detail["error"]["code"] == "bridge_parse_error"


@pytest.mark.asyncio
async def test_collect_openai_plain_response_when_tools_present():
    """tools_present=True + plain prose reply: returned as content with finish=stop, no error."""
    import asyncio
    from examples.openai_server import collect_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    parser = parse_envelope_stream(_from_chunks(["I don't need a tool. Hello!"]), tools_present=True)
    body = await collect_openai(parser, "chatcmpl-x", "sonnet", 1700000000, final_usage)
    assert body["choices"][0]["message"]["content"] == "I don't need a tool. Hello!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"].get("tool_calls") is None


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


import json as _json


@pytest.mark.asyncio
async def test_harness_openai_sdk_non_streaming():
    """Verify the official openai>=1.0 SDK can call our endpoint without errors."""
    import httpx
    from openai import AsyncOpenAI
    from examples.openai_server import app, set_agent_factory, _OCESConfig
    _OCESConfig.api_key = "testkey"
    set_agent_factory(_make_factory(["the answer is 42"]))
    try:
        client = AsyncOpenAI(
            base_url="http://testserver/v1",
            api_key="testkey",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                          base_url="http://testserver"),
        )
        resp = await client.chat.completions.create(
            model="sonnet",
            messages=[{"role": "user", "content": "what is the answer?"}],
        )
        assert resp.choices[0].message.content == "the answer is 42"
        assert resp.choices[0].finish_reason == "stop"
    finally:
        await client.close()
        set_agent_factory(None)


@pytest.mark.asyncio
async def test_harness_openai_sdk_streaming():
    import httpx
    from openai import AsyncOpenAI
    from examples.openai_server import app, set_agent_factory, _OCESConfig
    _OCESConfig.api_key = "testkey"
    set_agent_factory(_make_factory(["foo ", "bar ", "baz"]))
    try:
        client = AsyncOpenAI(
            base_url="http://testserver/v1",
            api_key="testkey",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                          base_url="http://testserver"),
        )
        stream = await client.chat.completions.create(
            model="sonnet",
            messages=[{"role": "user", "content": "x"}],
            stream=True,
        )
        chunks = []
        async for c in stream:
            chunks.append(c)
        text = "".join((c.choices[0].delta.content or "") for c in chunks if c.choices)
        assert text == "foo bar baz"
        assert any(c.choices and c.choices[0].finish_reason == "stop" for c in chunks)
    finally:
        await client.close()
        set_agent_factory(None)


@pytest.mark.asyncio
async def test_harness_openai_sdk_tool_calls():
    import httpx
    from openai import AsyncOpenAI
    from examples.openai_server import app, set_agent_factory, _OCESConfig
    _OCESConfig.api_key = "testkey"
    text = ('<<<TOOL_CALLS>>>[{"id":"call_1","name":"get_weather",'
            '"arguments":{"city":"Paris"}}]<<</TOOL_CALLS>>>'
            '<<<CONTENT>>><<</CONTENT>>>')
    set_agent_factory(_make_factory([text]))
    try:
        client = AsyncOpenAI(
            base_url="http://testserver/v1",
            api_key="testkey",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                          base_url="http://testserver"),
        )
        resp = await client.chat.completions.create(
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
        await client.close()
        set_agent_factory(None)


@pytest.mark.asyncio
async def test_collect_openai_applies_max_tokens():
    import asyncio
    from examples.openai_server import collect_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    parser = parse_envelope_stream(_from_chunks(["x" * 100]), tools_present=False)
    body = await collect_openai(parser, "chatcmpl-x", "sonnet", 0, final_usage,
                                 stop=None, max_tokens=10)
    # 10 tokens * 4 chars/token = 40 chars
    assert len(body["choices"][0]["message"]["content"]) == 40
    assert body["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_collect_openai_applies_stop():
    import asyncio
    from examples.openai_server import collect_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    parser = parse_envelope_stream(_from_chunks(["hello STOP world"]), tools_present=False)
    body = await collect_openai(parser, "chatcmpl-x", "sonnet", 0, final_usage,
                                 stop=["STOP"], max_tokens=None)
    assert body["choices"][0]["message"]["content"] == "hello "
    assert body["choices"][0]["finish_reason"] == "stop"


def test_endpoint_max_tokens_truncates_response():
    """End-to-end: max_tokens enforcement in non-streaming mode."""
    from examples.openai_server import set_agent_factory
    set_agent_factory(_make_factory(["x" * 200]))  # 200 chars
    try:
        client = _client()
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 5})  # 5 * 4 = 20 chars
        body = r.json()
        assert len(body["choices"][0]["message"]["content"]) == 20
        assert body["choices"][0]["finish_reason"] == "length"
    finally:
        set_agent_factory(None)


def test_endpoint_stop_truncates_response():
    """End-to-end: stop string enforcement."""
    from examples.openai_server import set_agent_factory
    set_agent_factory(_make_factory(["before HALT after"]))
    try:
        client = _client()
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": "hi"}],
                              "stop": "HALT"})
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "before "
        assert body["choices"][0]["finish_reason"] == "stop"
    finally:
        set_agent_factory(None)


@pytest.mark.asyncio
async def test_stream_openai_applies_max_tokens():
    """Streaming path: max_tokens enforcement."""
    import asyncio
    from examples.openai_server import stream_openai, parse_envelope_stream
    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    parser = parse_envelope_stream(_from_chunks(["a" * 50, "b" * 50]), tools_present=False)
    sse = ""
    async for line in stream_openai(parser, "chatcmpl-x", "sonnet", 0, False, final_usage,
                                     stop=None, max_tokens=10):  # 40 char budget
        sse += line
    chunks = _parse_sse(sse)
    contents = [c["choices"][0]["delta"].get("content", "")
                for c in chunks if "content" in c["choices"][0].get("delta", {})]
    total = "".join(c for c in contents if c)
    assert len(total) == 40
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_stream_openai_catches_mid_stream_cckit_error():
    """If cckit raises after streaming starts, error chunk + [DONE] are emitted (not silent abort)."""
    import asyncio
    from examples.openai_server import stream_openai
    from cckit.utils.errors import TimeoutError as CckitTimeout

    async def _bad_parser():
        yield {"kind": "text_delta", "text": "partial response..."}
        raise CckitTimeout("simulated timeout mid-stream")

    final_usage = asyncio.get_event_loop().create_future()
    final_usage.set_result({"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_tokens_details": {"cached_tokens": 0}})
    sse = ""
    async for line in stream_openai(_bad_parser(), "chatcmpl-x", "sonnet", 0, False, final_usage,
                                     stop=None, max_tokens=None):
        sse += line
    chunks = _parse_sse(sse)
    err_chunks = [c for c in chunks if c.get("error")]
    assert len(err_chunks) == 1
    assert err_chunks[0]["error"]["code"] == "claude_timeout"
    assert err_chunks[0]["choices"][0]["finish_reason"] == "error"
    assert sse.endswith("data: [DONE]\n\n")


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


# ── image-input path ────────────────────────────────────────────────────────

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/"
    "w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
)


def test_image_part_to_block_data_url_ok():
    from examples.openai_server import _image_part_to_block, ImageContentPart, ImageURL
    part = ImageContentPart(type="image_url",
                            image_url=ImageURL(url=f"data:image/png;base64,{_TINY_PNG_B64}"))
    block = _image_part_to_block(part)
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["data"] == _TINY_PNG_B64


def test_image_part_to_block_data_url_bad_media_type():
    from examples.openai_server import _image_part_to_block, ImageContentPart, ImageURL
    part = ImageContentPart(type="image_url",
                            image_url=ImageURL(url=f"data:image/bmp;base64,{_TINY_PNG_B64}"))
    with pytest.raises(HTTPException) as exc:
        _image_part_to_block(part)
    assert exc.value.status_code == 400
    assert "media_type" in exc.value.detail["error"]["message"]


def test_image_part_to_block_data_url_malformed():
    from examples.openai_server import _image_part_to_block, ImageContentPart, ImageURL
    # Missing base64 portion
    part = ImageContentPart(type="image_url",
                            image_url=ImageURL(url="data:image/png,not-base64"))
    with pytest.raises(HTTPException) as exc:
        _image_part_to_block(part)
    assert exc.value.status_code == 400
    assert "malformed data URL" in exc.value.detail["error"]["message"]


def test_image_part_to_block_oversized():
    from examples.openai_server import _image_part_to_block, ImageContentPart, ImageURL, MAX_IMAGE_B64_LEN
    huge = "A" * (MAX_IMAGE_B64_LEN + 1)
    part = ImageContentPart(type="image_url",
                            image_url=ImageURL(url=f"data:image/png;base64,{huge}"))
    with pytest.raises(HTTPException) as exc:
        _image_part_to_block(part)
    assert exc.value.status_code == 400
    assert "too large" in exc.value.detail["error"]["message"]


def test_image_part_to_block_https_passthrough():
    from examples.openai_server import _image_part_to_block, ImageContentPart, ImageURL
    part = ImageContentPart(type="image_url",
                            image_url=ImageURL(url="https://example.com/cat.png"))
    block = _image_part_to_block(part)
    assert block == {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}


def test_image_part_to_block_file_url_rejected():
    from examples.openai_server import _image_part_to_block, ImageContentPart, ImageURL
    part = ImageContentPart(type="image_url",
                            image_url=ImageURL(url="file:///etc/passwd"))
    with pytest.raises(HTTPException) as exc:
        _image_part_to_block(part)
    assert exc.value.status_code == 400
    assert "scheme" in exc.value.detail["error"]["message"]


def test_last_user_msg_has_images_no_images():
    from examples.openai_server import _last_user_msg_has_images
    req = _req(messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "thanks"},
    ])
    assert _last_user_msg_has_images(req) is False


def test_last_user_msg_has_images_when_present_in_last_turn():
    from examples.openai_server import _last_user_msg_has_images
    req = _req(messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": [
            {"type": "text", "text": "what's this?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
        ]},
    ])
    assert _last_user_msg_has_images(req) is True


def test_last_user_msg_has_images_only_in_prior_turn_returns_false():
    """Per design: prior-turn images degrade to placeholders → text path."""
    from examples.openai_server import _last_user_msg_has_images
    req = _req(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "follow-up"},
    ])
    assert _last_user_msg_has_images(req) is False


def test_last_user_msg_has_images_returns_false_when_final_is_tool_result():
    """Regression: vision + tool-call flow. The final message is a tool result, so
    the image (in an earlier user turn) must NOT trigger the stream-json path —
    sending it as 'current input' would invert conversation chronology."""
    from examples.openai_server import _last_user_msg_has_images, build_prompt_with_blocks
    req = _req(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "describe and call the noop tool"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
        ]},
        {"role": "assistant", "content": "calling tool", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "noop", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
    ])
    # Trigger says: text path (final message is "tool", not "user")
    assert _last_user_msg_has_images(req) is False


def test_last_user_msg_has_images_returns_false_when_final_is_assistant():
    """Same family: trigger requires the final message to be a user message."""
    from examples.openai_server import _last_user_msg_has_images
    req = _req(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
        ]},
        {"role": "assistant", "content": "ok"},
    ])
    assert _last_user_msg_has_images(req) is False


# ── /v1/models endpoint ─────────────────────────────────────────────────────
#
# Discovery lives in cckit (cckit.core.models). These tests verify OCES wraps
# its result in the OpenAI list shape — the discovery logic itself is covered
# by tests/test_models.py.


@pytest.fixture
def _models_cache_clear():
    """Each /v1/models test starts with an empty discovery cache."""
    from cckit import discover_claude_models
    discover_claude_models.cache_clear()
    yield
    discover_claude_models.cache_clear()


def _stub_models(monkeypatch, ids: tuple[str, ...]) -> None:
    """Replace cckit discovery with a fixed list for the test."""
    import examples.openai_server as mod
    monkeypatch.setattr(mod, "discover_claude_models", lambda _path: ids)


def test_models_endpoint_requires_auth(_models_cache_clear):
    client = _client()
    r = client.get("/v1/models")
    assert r.status_code == 401


def test_models_endpoint_rejects_wrong_key(_models_cache_clear):
    client = _client()
    r = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_models_endpoint_returns_discovered_list(_models_cache_clear, monkeypatch):
    """Endpoint forwards whatever cckit discovers, wrapped in OpenAI shape."""
    _stub_models(monkeypatch,
                 ("claude-opus-4-7", "claude-sonnet-4-6", "haiku", "opus", "sonnet"))
    client = _client()
    r = client.get("/v1/models", headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids == ["claude-opus-4-7", "claude-sonnet-4-6", "haiku", "opus", "sonnet"]
    for m in body["data"]:
        assert m["object"] == "model"
        assert m["owned_by"] == "anthropic"
        assert isinstance(m["created"], int)


def test_models_endpoint_root_path_works_too(_models_cache_clear, monkeypatch):
    """Parity with /chat/completions also being mounted at root."""
    _stub_models(monkeypatch, ("sonnet", "opus", "haiku"))
    client = _client()
    r_root = client.get("/models", headers=_auth())
    r_v1 = client.get("/v1/models", headers=_auth())
    assert r_root.status_code == 200
    assert r_v1.status_code == 200
    assert r_root.json() == r_v1.json()


def test_models_endpoint_response_is_deterministic(_models_cache_clear, monkeypatch):
    """Static created timestamp keeps responses cacheable; spot-check it doesn't drift."""
    _stub_models(monkeypatch, ("sonnet", "opus", "haiku"))
    client = _client()
    r1 = client.get("/v1/models", headers=_auth()).json()
    r2 = client.get("/v1/models", headers=_auth()).json()
    assert r1 == r2


def test_build_prompt_with_blocks_preserves_interleave_order():
    """Advisor-flagged bug to avoid: [text, image, text, image] must stay in order."""
    from examples.openai_server import build_prompt_with_blocks
    req = _req(messages=[{"role": "user", "content": [
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
        {"type": "text", "text": "second"},
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
    ]}])
    _, blocks = build_prompt_with_blocks(req)
    assert [b["type"] for b in blocks] == ["text", "image", "text", "image"]
    assert blocks[0]["text"] == "first"
    assert blocks[2]["text"] == "second"
    assert blocks[1]["source"]["type"] == "base64"
    assert blocks[3]["source"]["type"] == "url"


def test_build_prompt_with_blocks_prior_transcript_as_leading_text():
    from examples.openai_server import build_prompt_with_blocks
    req = _req(messages=[
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "remember the color blue"},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": [
            {"type": "text", "text": "what color did I mention?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
        ]},
    ])
    sys_prompt, blocks = build_prompt_with_blocks(req)
    # System turns go to sys_prompt, not blocks
    assert "you are helpful" in sys_prompt
    # Prior transcript becomes a leading text block
    assert blocks[0]["type"] == "text"
    assert "[User]: remember the color blue" in blocks[0]["text"]
    assert "[Assistant]: noted" in blocks[0]["text"]
    # Then the live user's text and image
    assert blocks[1]["type"] == "text"
    assert blocks[1]["text"] == "what color did I mention?"
    assert blocks[2]["type"] == "image"


def test_build_prompt_with_blocks_no_prior_transcript_when_only_last_user():
    """Single-turn vision: no leading transcript block, just the live blocks."""
    from examples.openai_server import build_prompt_with_blocks
    req = _req(messages=[{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
    ]}])
    _, blocks = build_prompt_with_blocks(req)
    assert len(blocks) == 2
    assert blocks[0] == {"type": "text", "text": "describe"}
    assert blocks[1]["type"] == "image"


def test_build_prompt_with_blocks_validates_eagerly():
    """Image parsing errors must raise synchronously (before StreamingResponse)."""
    from examples.openai_server import build_prompt_with_blocks
    req = _req(messages=[{"role": "user", "content": [
        {"type": "text", "text": "x"},
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
    ]}])
    with pytest.raises(HTTPException) as exc:
        build_prompt_with_blocks(req)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_drive_claude_routes_to_messages_path_when_blocks_supplied():
    """When _handle_chat passes content_blocks, drive_claude calls stream_execute_messages."""
    import asyncio
    from examples.openai_server import drive_claude, set_agent_factory
    fake = _FakeAgent(["got image"])
    set_agent_factory(lambda req: fake)
    blocks = [{"type": "text", "text": "what?"},
              {"type": "image", "source": {"type": "base64",
                                            "media_type": "image/png", "data": "AAAA"}}]
    final_usage = asyncio.get_event_loop().create_future()
    out = []
    try:
        async for chunk in drive_claude(_req(), final_usage, content_blocks=blocks):
            out.append(chunk)
    finally:
        set_agent_factory(None)
    assert out == ["got image"]
    # Routed to the messages method, not stream_execute
    assert fake.calls[0][0] == "stream_execute_messages"
    msgs = fake.calls[0][1]
    assert msgs[0]["type"] == "user"
    assert msgs[0]["message"]["content"] == blocks


@pytest.mark.asyncio
async def test_drive_claude_text_path_unchanged_when_no_blocks():
    """Regression: existing text path still calls stream_execute, not the new method."""
    import asyncio
    from examples.openai_server import drive_claude, set_agent_factory
    fake = _FakeAgent(["hello"])
    set_agent_factory(lambda req: fake)
    final_usage = asyncio.get_event_loop().create_future()
    out = []
    try:
        async for chunk in drive_claude(_req(), final_usage):
            out.append(chunk)
    finally:
        set_agent_factory(None)
    assert out == ["hello"]
    assert fake.calls[0][0] == "stream_execute"


def test_endpoint_rejects_bad_image_with_400_before_streaming():
    """E2E: malformed image returns 400 (not 200 OK with in-band error chunk)."""
    from fastapi.testclient import TestClient
    from examples.openai_server import app, _OCESConfig, set_agent_factory
    _OCESConfig.api_key = "testkey"
    # Set a benign factory so we'd succeed if validation didn't catch us
    set_agent_factory(_make_factory(["ok"]))
    try:
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "stream": True,
                              "messages": [{"role": "user", "content": [
                                  {"type": "text", "text": "look"},
                                  {"type": "image_url",
                                   "image_url": {"url": "file:///etc/passwd"}},
                              ]}]})
        assert r.status_code == 400
        assert "scheme" in r.json()["error"]["message"]
    finally:
        set_agent_factory(None)


def test_endpoint_image_request_routes_to_messages_path():
    """E2E: a valid image request goes through stream_execute_messages."""
    from fastapi.testclient import TestClient
    from examples.openai_server import app, _OCESConfig, set_agent_factory

    captured: dict = {}

    class _CaptureAgent:
        async def stream_execute_messages(self, messages):
            from cckit import ResultEvent, TextChunkEvent
            captured["messages"] = messages
            yield TextChunkEvent(text="seen")
            yield ResultEvent(raw={"usage": {"input_tokens": 1, "output_tokens": 1}},
                              result="", session_id="x")

    _OCESConfig.api_key = "testkey"
    set_agent_factory(lambda req: _CaptureAgent())
    try:
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/v1/chat/completions", headers=_auth(),
                        json={"model": "sonnet",
                              "messages": [{"role": "user", "content": [
                                  {"type": "text", "text": "describe"},
                                  {"type": "image_url",
                                   "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
                              ]}]})
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "seen"
        # Verify the agent saw a structured stream-json input message
        msg = captured["messages"][0]
        assert msg["type"] == "user"
        content = msg["message"]["content"]
        assert content[0]["type"] == "text" and content[0]["text"] == "describe"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["media_type"] == "image/png"
    finally:
        set_agent_factory(None)
