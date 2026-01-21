"""YAML config loader with composition and deep merge."""
from __future__ import annotations

import os
from typing import Any

import yaml


class ConfigLoadError(ValueError):
    """Raised when loading or composing configs fails."""


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config with support for defaults composition."""
    abs_path = os.path.abspath(path)
    return _load_with_defaults(abs_path, stack=[])


def _load_with_defaults(path: str, stack: list[str]) -> dict[str, Any]:
    if path in stack:
        cycle = " -> ".join(stack + [path])
        raise ConfigLoadError(f"Config defaults cycle detected: {cycle}")

    stack.append(path)
    data = _read_yaml(path)
    defaults = data.get("defaults", []) if isinstance(data, dict) else []
    merged: dict[str, Any] = {}

    if defaults:
        if not isinstance(defaults, list):
            raise ConfigLoadError(f"defaults must be a list in {path}")
        for entry in defaults:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ConfigLoadError(
                    f"defaults entries must be single-key mappings in {path}: {entry}"
                )
            _, rel_path = next(iter(entry.items()))
            if not isinstance(rel_path, str):
                raise ConfigLoadError(f"defaults entry path must be a string in {path}: {entry}")
            child_path = _resolve_path(rel_path, base=path)
            child_cfg = _load_with_defaults(child_path, stack)
            merged = _deep_merge(merged, child_cfg)

    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if k != "defaults"}
        merged = _deep_merge(merged, data)

    stack.pop()
    return merged


def _read_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise ConfigLoadError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigLoadError(f"Config root must be a mapping: {path}")
    return data


def _resolve_path(rel_path: str, base: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    base_dir = os.path.dirname(base)
    return os.path.abspath(os.path.join(base_dir, rel_path))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
