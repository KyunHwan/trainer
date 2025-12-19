"""Logger interface and built-ins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    """Minimal logger interface."""

    def log(self, metrics: dict[str, Any], step: int) -> None: ...

    def flush(self) -> None: ...


@dataclass
class StdoutLogger:
    prefix: str = "train"

    def log(self, metrics: dict[str, Any], step: int) -> None:
        parts = ", ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"[{self.prefix}] step={step} {parts}")

    def flush(self) -> None:
        return None


class NoOpLogger:
    def log(self, metrics: dict[str, Any], step: int) -> None:  # noqa: ARG002
        return None

    def flush(self) -> None:
        return None
