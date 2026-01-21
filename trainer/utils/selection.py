"""Helpers for selecting values from structured objects."""
from __future__ import annotations

from typing import Any


def select(obj: Any, key: str | int | None) -> Any:
    """Select a value by key/index from dict/tuple/object; None returns obj."""
    if key is None:
        return obj
    if isinstance(key, int):
        return obj[key]
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)
