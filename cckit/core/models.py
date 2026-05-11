"""Discover model IDs the claude CLI binary recognizes.

The claude binary has no list-models subcommand, but its compiled-in model
resolution table embeds the model identifiers as ASCII strings inside the
executable. ``discover_claude_models`` reads the binary once and regex-scans
for IDs matching the documented Anthropic naming grammar, returning a sorted
tuple suitable for advertising via OpenAI-style ``/v1/models`` endpoints or
similar use cases.

Maintenance: if Anthropic ships a new model FAMILY (e.g., a hypothetical
"ultra" line), update ``_MODEL_ID_PATTERN`` here to include it. Within an
existing family, no code change is needed — new IDs ship inside the binary
and the scan finds them automatically on next call (after cache reset).
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

from ..utils.helpers import expand_path, get_logger

logger = get_logger(__name__)

# Bare aliases — accepted by the binary per ``claude --help``. These don't
# carry the ``claude-`` prefix so the strings scan won't pick them up; we
# prepend them explicitly. Stable across versions.
CLAUDE_BARE_ALIASES: tuple[str, ...] = ("sonnet", "opus", "haiku")

# Strict ID grammar. Boundaries (lookahead/lookbehind) ensure we only match
# whole IDs, not substrings of unrelated text that happens to contain
# ``claude-opus-...`` (e.g. plugin or feature-flag identifiers that start
# with the same prefix).
_MODEL_ID_PATTERN = re.compile(
    rb"(?<![a-zA-Z0-9_-])claude-(?:opus|sonnet|haiku|3)-[a-z0-9-]+(?![a-zA-Z0-9_-])"
)


@functools.lru_cache(maxsize=4)
def discover_claude_models(binary_path: str = "~/.local/bin/claude") -> tuple[str, ...]:
    """Return the sorted tuple of model IDs the claude binary recognizes.

    Resolves symlinks, reads the binary, regex-scans for IDs matching the
    documented Anthropic naming grammar, and prepends the bare aliases
    (``sonnet``, ``opus``, ``haiku``). Result is LRU-cached per binary path —
    the binary doesn't change between requests during a process lifetime.

    On read failure (missing file, permission denied, etc.) falls back to
    just the bare aliases (always valid, always accepted by the CLI). The
    fallback is logged at WARNING.

    Tests should call ``discover_claude_models.cache_clear()`` between cases
    that exercise different binaries.

    Args:
        binary_path: path to the claude binary. Tilde expansion is applied.
    """
    path = Path(expand_path(binary_path))
    if path.is_symlink():
        path = path.resolve()
    try:
        data = path.read_bytes()
    except OSError as exc:
        logger.warning(
            "Could not read claude binary for model discovery (%s): %s — "
            "falling back to bare aliases",
            path, exc,
        )
        return tuple(sorted(CLAUDE_BARE_ALIASES))

    found: set[str] = set(CLAUDE_BARE_ALIASES)
    for raw in _MODEL_ID_PATTERN.findall(data):
        found.add(raw.decode("ascii"))
    return tuple(sorted(found))
