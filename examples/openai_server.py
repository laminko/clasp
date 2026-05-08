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
