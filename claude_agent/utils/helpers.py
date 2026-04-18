from __future__ import annotations

import json
import logging
import os
from pathlib import Path


def expand_path(path: str) -> str:
    """Expand ~ and environment variables in a path string."""
    return str(Path(os.path.expandvars(os.path.expanduser(path))).resolve())


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


def safe_json_loads(text: str) -> dict | None:
    """Parse JSON, returning None on failure instead of raising."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
