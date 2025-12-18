from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from offline_trainer.deps.model_constructor import ConfigError


@dataclass(frozen=True)
class ArgsKwargs:
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


Path = list[str | int]


@dataclass
class MappingIO:
    model_kwargs: dict[str, Path]
    targets: dict[str, Path]

    def split(self, batch: Any, *, stage: str) -> tuple[ArgsKwargs, Any, dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        for name, path in self.model_kwargs.items():
            try:
                kwargs[name] = _extract(batch, path)
            except Exception as exc:
                raise ConfigError(
                    f"Failed to extract model input {name!r} at path {path}: {exc}",
                    config_path=("io", "model_kwargs", name),
                ) from exc

        tgt: dict[str, Any] = {}
        for name, path in self.targets.items():
            try:
                tgt[name] = _extract(batch, path)
            except Exception as exc:
                raise ConfigError(
                    f"Failed to extract target {name!r} at path {path}: {exc}",
                    config_path=("io", "targets", name),
                ) from exc

        return ArgsKwargs(args=(), kwargs=kwargs), tgt, {"stage": stage}


def _extract(obj: Any, path: Path) -> Any:
    cur = obj
    for seg in path:
        if isinstance(seg, str):
            if not isinstance(cur, dict) or seg not in cur:
                raise KeyError(seg)
            cur = cur[seg]
        elif isinstance(seg, int):
            if not isinstance(cur, (list, tuple)) or seg < 0 or seg >= len(cur):
                raise IndexError(seg)
            cur = cur[seg]
        else:
            raise TypeError(f"Invalid path segment type: {type(seg).__name__}")
    return cur

