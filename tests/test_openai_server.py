"""Tests for examples/openai_server.py (OCES)."""
import importlib


def test_module_imports():
    mod = importlib.import_module("examples.openai_server")
    assert hasattr(mod, "app"), "FastAPI app must be exported as `app`"
