"""Helpers for safe dynamic imports and instantiation."""
from __future__ import annotations

import importlib
import inspect
from typing import Any


def import_module(path: str) -> Any:
    """Import a module by dotted path."""
    return importlib.import_module(path)


def instantiate(obj: Any, params: dict[str, Any] | None = None, **extra: Any) -> Any:
    """Instantiate a class or call a factory with filtered kwargs.

    Extra kwargs are passed only if accepted by the target signature or **kwargs exists.
    """
    if params is None:
        params = {}

    if inspect.isclass(obj):
        ctor = obj
        kwargs = _filter_kwargs(ctor, {**params, **extra})
        return ctor(**kwargs)

    if callable(obj):
        kwargs = _filter_kwargs(obj, {**params, **extra})
        return obj(**kwargs)

    return obj


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    allowed = {name for name in params.keys()}
    return {k: v for k, v in kwargs.items() if k in allowed}
