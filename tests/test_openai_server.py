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
