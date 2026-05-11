"""Tests for cckit.core.models.discover_claude_models."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from cckit import CLAUDE_BARE_ALIASES, discover_claude_models


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a clean discovery cache."""
    discover_claude_models.cache_clear()
    yield
    discover_claude_models.cache_clear()


def _synth_binary(tmp_path: Path, contents: bytes) -> Path:
    """Write a synthetic 'binary' (just bytes) and return its path."""
    p = tmp_path / "fake-claude"
    p.write_bytes(contents)
    p.chmod(0o755)
    return p


class TestDiscoverClaudeModels:
    def test_extracts_clean_ids_from_synthetic_binary(self, tmp_path: Path) -> None:
        """The regex finds whole model IDs separated by non-printable bytes."""
        data = (
            b"\x00\x00claude-opus-4-7\x00\x00"
            b"\x01claude-sonnet-4-6\x00"
            b"prefix:claude-haiku-4-5-20251001;suffix"
        )
        path = _synth_binary(tmp_path, data)
        models = discover_claude_models(str(path))
        assert "claude-opus-4-7" in models
        assert "claude-sonnet-4-6" in models
        assert "claude-haiku-4-5-20251001" in models

    def test_bare_aliases_always_included(self, tmp_path: Path) -> None:
        """sonnet/opus/haiku are documented bare aliases — prepend regardless."""
        path = _synth_binary(tmp_path, b"no model ids in here, just text")
        models = discover_claude_models(str(path))
        for alias in CLAUDE_BARE_ALIASES:
            assert alias in models

    def test_result_is_sorted_and_deduplicated(self, tmp_path: Path) -> None:
        data = b"\x00claude-opus-4-7\x00claude-opus-4-7\x00claude-sonnet-4-6\x00"
        path = _synth_binary(tmp_path, data)
        models = discover_claude_models(str(path))
        # Sorted ascending; duplicates removed
        assert list(models) == sorted(set(models))

    def test_ignores_substrings_inside_word_chars(self, tmp_path: Path) -> None:
        """Boundary lookahead/lookbehind prevents picking up embedded substrings."""
        # 'xxxclaude-opus-4-7zzz' should NOT match — surrounded by word chars.
        # 'claude-opus-4-7' alone should.
        data = b"xxxclaude-opus-4-7zzz\x00claude-opus-4-7\x00"
        path = _synth_binary(tmp_path, data)
        models = discover_claude_models(str(path))
        # Only one occurrence ends up in the set (the standalone one)
        assert "claude-opus-4-7" in models
        # And we didn't mistakenly capture the prefixed form
        assert "xxxclaude-opus-4-7zzz" not in models

    def test_only_known_families_match(self, tmp_path: Path) -> None:
        """Plugin/feature-flag prefixes like 'claude-plugin-foo' don't appear."""
        data = (
            b"\x00claude-plugin-foo-bar\x00"
            b"claude-code-marketplace\x00"
            b"claude-ai-external-token\x00"
            b"claude-opus-4-7\x00"
        )
        path = _synth_binary(tmp_path, data)
        models = discover_claude_models(str(path))
        assert "claude-opus-4-7" in models
        assert "claude-plugin-foo-bar" not in models
        assert "claude-code-marketplace" not in models
        assert "claude-ai-external-token" not in models

    def test_falls_back_to_bare_aliases_when_binary_missing(self, tmp_path: Path) -> None:
        models = discover_claude_models(str(tmp_path / "does-not-exist"))
        # Fallback: just the bare aliases (sorted)
        assert set(models) == set(CLAUDE_BARE_ALIASES)

    def test_returns_tuple(self, tmp_path: Path) -> None:
        """Tuple return is required for lru_cache compatibility."""
        path = _synth_binary(tmp_path, b"\x00claude-opus-4-7\x00")
        models = discover_claude_models(str(path))
        assert isinstance(models, tuple)

    def test_caches_by_binary_path(self, tmp_path: Path) -> None:
        """Repeat calls don't re-read the file (lru_cache hit)."""
        path = _synth_binary(tmp_path, b"\x00claude-opus-4-7\x00")
        first = discover_claude_models(str(path))
        # Overwrite the file with different content — cache should hide this
        path.write_bytes(b"\x00claude-sonnet-4-6\x00")
        second = discover_claude_models(str(path))
        assert first == second  # cache hit, same result despite changed file
        # After clearing cache the new content is visible
        discover_claude_models.cache_clear()
        third = discover_claude_models(str(path))
        assert "claude-sonnet-4-6" in third
        assert "claude-opus-4-7" not in third or "claude-opus-4-7" in CLAUDE_BARE_ALIASES

    def test_expands_user_path(self, tmp_path: Path, monkeypatch) -> None:
        """Tilde paths are expanded against $HOME."""
        # Put a synthetic binary at $HOME/fake-claude
        monkeypatch.setenv("HOME", str(tmp_path))
        _synth_binary(tmp_path, b"\x00claude-opus-4-7\x00")
        models = discover_claude_models("~/fake-claude")
        assert "claude-opus-4-7" in models

    def test_real_binary_smoke(self) -> None:
        """Smoke test against the actual installed claude binary, if present.

        Skipped when no claude binary is on disk — keeps the test suite portable.
        """
        candidate = os.path.expanduser("~/.local/bin/claude")
        if not os.path.exists(candidate):
            pytest.skip("no claude binary at ~/.local/bin/claude")
        models = discover_claude_models(candidate)
        # Sanity: includes bare aliases and at least one current-gen ID
        for alias in CLAUDE_BARE_ALIASES:
            assert alias in models
        # At least one of the well-known current-gen IDs is in the binary
        current_gen_hit = any(
            m.startswith("claude-opus-4-") or m.startswith("claude-sonnet-4-")
            or m.startswith("claude-haiku-4-")
            for m in models
        )
        assert current_gen_hit, f"expected at least one claude-4-* ID, got: {models}"
