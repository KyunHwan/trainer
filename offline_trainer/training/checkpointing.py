"""Checkpoint save/load utilities."""
from __future__ import annotations

import os
from typing import Any

import torch

from offline_trainer.training.state import TrainState


class CheckpointManager:
    """Manages saving and loading training checkpoints."""

    def __init__(self, save_dir: str, save_every_n_steps: int | None = None, save_last: bool = True) -> None:
        self.save_dir = save_dir
        self.save_every_n_steps = save_every_n_steps
        self.save_last = save_last
        os.makedirs(self.save_dir, exist_ok=True)

    def maybe_save(
        self,
        *,
        step: int,
        models: dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any,
        scaler: torch.cuda.amp.GradScaler | None,
        state: TrainState,
        callbacks: list[Any],
    ) -> None:
        if self.save_every_n_steps is None:
            return
        if step % self.save_every_n_steps == 0:
            self.save(
                path=self._step_path(step),
                models=models,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                state=state,
                callbacks=callbacks,
            )
            if self.save_last:
                self.save(
                    path=self._last_path(),
                    models=models,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    state=state,
                    callbacks=callbacks,
                )

    def save(
        self,
        *,
        path: str,
        models: dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any,
        scaler: torch.cuda.amp.GradScaler | None,
        state: TrainState,
        callbacks: list[Any],
    ) -> None:
        payload = {
            "models": {name: model.state_dict() for name, model in models.items()},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "train_state": state.to_dict(),
            "callbacks": {str(i): cb.state_dict() for i, cb in enumerate(callbacks)},
        }
        torch.save(payload, path)

    def load(self, path: str) -> dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu")

    def _step_path(self, step: int) -> str:
        return os.path.join(self.save_dir, f"step_{step}.pt")

    def _last_path(self) -> str:
        return os.path.join(self.save_dir, "last.pt")
