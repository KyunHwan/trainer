from __future__ import annotations

from pathlib import Path
from typing import Any

from offline_trainer.deps.model_constructor import ConfigError, SourceMap


def enrich_config_error(exc: ConfigError, *, source_map: SourceMap) -> ConfigError:
    if exc.location is not None or exc.config_path is None:
        return exc
    loc = source_map.get(exc.config_path)
    if loc is None:
        return exc
    return ConfigError(
        exc.message,
        config_path=exc.config_path,
        location=loc,
        include_stack=list(exc.include_stack),
        suggestions=list(exc.suggestions),
    )


def format_error_with_context(exc: ConfigError, *, context_lines: int = 2) -> str:
    parts = [str(exc)]
    if exc.location is None:
        return "\n".join(parts)

    try:
        p = Path(exc.location.file)
        if not p.exists():
            return "\n".join(parts)
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "\n".join(parts)

    line_idx = exc.location.line - 1
    start = max(0, line_idx - context_lines)
    end = min(len(lines), line_idx + context_lines + 1)

    width = len(str(end))
    snippet: list[str] = []
    for i in range(start, end):
        prefix = ">" if i == line_idx else " "
        snippet.append(f"{prefix} {str(i+1).rjust(width)} | {lines[i]}")
    parts.append("\n".join(snippet))
    return "\n".join(parts)


def raise_config_error(
    message: str,
    *,
    config_path: tuple[Any, ...],
    source_map: SourceMap,
    suggestions: list[str] | None = None,
) -> None:
    raise ConfigError(message, config_path=config_path, location=source_map.get(config_path), suggestions=suggestions)
