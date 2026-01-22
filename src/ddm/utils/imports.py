from __future__ import annotations

from typing import Any


def optional_import(name: str) -> Any | None:
    """
    Import a module if available; return None if not installed.
    """
    try:
        return __import__(name)
    except Exception:
        return None
