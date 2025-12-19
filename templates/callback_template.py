"""Template for a custom callback."""
from __future__ import annotations

from dataclasses import dataclass

from offline_trainer.registry import CALLBACK_REGISTRY
from offline_trainer.training.state import TrainState


@CALLBACK_REGISTRY.register("my_callback")
@dataclass
class MyCallback:
    def on_train_start(self, state: TrainState) -> None:  # noqa: ARG002
        return None

    def on_step_end(self, state: TrainState) -> None:  # noqa: ARG002
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:  # noqa: ARG002
        return None
