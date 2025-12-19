"""Callback interface and built-ins."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from offline_trainer.training.state import TrainState


@runtime_checkable
class Callback(Protocol):
    """Hook-based callbacks with state persistence."""

    def on_train_start(self, state: TrainState) -> None: ...

    def on_step_end(self, state: TrainState) -> None: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


class NoOpCallback:
    def on_train_start(self, state: TrainState) -> None:  # noqa: ARG002
        return None

    def on_step_end(self, state: TrainState) -> None:  # noqa: ARG002
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:  # noqa: ARG002
        return None
