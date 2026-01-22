from __future__ import annotations

from typing import Any
import importlib


def optional_import(name: str) -> Any | None:
    """
    Import a module if available; return None if not installed.
    """
    try:
        return importlib.import_module(name)
    except Exception:
        return None
