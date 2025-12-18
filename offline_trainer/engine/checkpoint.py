from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from offline_trainer.engine.accelerator import AcceleratorState
from offline_trainer.engine.seed import get_rng_state, set_rng_state
from offline_trainer.engine.state import TrainState


@dataclass(frozen=True)
class Checkpoint:
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None
    accelerator: dict[str, Any]
    callbacks: list[dict[str, Any]]
    train_state: dict[str, Any]
    rng_state: dict[str, Any]
    meta: dict[str, Any]


def capture_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    accelerator_state: AcceleratorState,
    callbacks: list[Any],
    train_state: TrainState,
    include_cuda: bool,
    meta: dict[str, Any] | None = None,
) -> Checkpoint:
    cb_states: list[dict[str, Any]] = []
    for cb in callbacks:
        state_dict = cb.state_dict() if hasattr(cb, "state_dict") else {}
        cb_states.append({"type": f"{cb.__class__.__module__}.{cb.__class__.__qualname__}", "state": state_dict})

    scheduler_state = None
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        scheduler_state = scheduler.state_dict()

    return Checkpoint(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler_state,
        accelerator=asdict(accelerator_state),
        callbacks=cb_states,
        train_state=asdict(train_state),
        rng_state=get_rng_state(include_cuda=include_cuda),
        meta=dict(meta or {}),
    )


def save_checkpoint(path: str | Path, checkpoint: Checkpoint) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(checkpoint), p)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def restore_checkpoint(
    raw: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    accelerator_state: AcceleratorState,
    callbacks: list[Any],
    include_cuda: bool,
) -> TrainState:
    model.load_state_dict(raw["model"])
    optimizer.load_state_dict(raw["optimizer"])

    if raw.get("scheduler") is not None:
        if scheduler is None or not hasattr(scheduler, "load_state_dict"):
            raise ValueError("Checkpoint contains scheduler state but no scheduler is configured")
        scheduler.load_state_dict(raw["scheduler"])

    accel = raw.get("accelerator", {})
    if isinstance(accel, dict) and accel.get("scaler") is not None:
        accelerator_state.scaler = accel.get("scaler")

    cb_states = raw.get("callbacks", [])
    if not isinstance(cb_states, list) or len(cb_states) != len(callbacks):
        raise ValueError("Checkpoint callbacks do not match current config")
    for cb, cb_raw in zip(callbacks, cb_states, strict=False):
        state = cb_raw.get("state", {}) if isinstance(cb_raw, dict) else {}
        if hasattr(cb, "load_state_dict"):
            cb.load_state_dict(state)

    rng_state = raw.get("rng_state")
    if isinstance(rng_state, dict):
        set_rng_state(rng_state, include_cuda=include_cuda)

    ts_raw = raw.get("train_state", {})
    if not isinstance(ts_raw, dict):
        raise ValueError("Invalid train_state in checkpoint")
    return TrainState(
        epoch=int(ts_raw.get("epoch", 0)),
        global_step=int(ts_raw.get("global_step", 0)),
        micro_step_in_epoch=int(ts_raw.get("micro_step_in_epoch", 0)),
    )
