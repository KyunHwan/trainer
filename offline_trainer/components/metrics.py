from __future__ import annotations

from typing import Any


class NoOpMetric:
    def update(self, *, outputs: Any, targets: Any, batch: Any, stage: str) -> None:
        return

    def compute(self) -> dict[str, float]:
        return {}

    def reset(self) -> None:
        return

