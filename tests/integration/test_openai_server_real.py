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
