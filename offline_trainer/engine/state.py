from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    micro_step_in_epoch: int = 0


@dataclass
class RunContext:
    run_name: str
    out_dir: str
    state: TrainState

    model: Any
    optimizer: Any
    scheduler: Any
    scheduler_bundle: Any
    accelerator: Any

    datamodule: Any
    io: Any
    loss: Any
    metrics: list[Any] = field(default_factory=list)
    callbacks: list[Any] = field(default_factory=list)
    loggers: list[Any] = field(default_factory=list)

    last_train_logs: dict[str, float] = field(default_factory=dict)
    last_val_logs: dict[str, float] = field(default_factory=dict)
