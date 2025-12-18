from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StdoutLogger:
    every_n_steps: int = 50

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        return

    def log_metrics(self, metrics: dict[str, float], *, step: int, stage: str) -> None:
        if self.every_n_steps and step % self.every_n_steps != 0:
            return
        keys = ", ".join(f"{k}={v:.6g}" for k, v in sorted(metrics.items()))
        print(f"[{stage}] step={step} {keys}")

    def close(self) -> None:
        return

