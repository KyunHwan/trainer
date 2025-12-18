from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from offline_trainer.engine.checkpoint import capture_checkpoint, save_checkpoint
from offline_trainer.engine.state import RunContext


@dataclass
class ModelCheckpoint:
    save_dir: str
    save_last: bool = True
    every_n_steps: int | None = None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return

    def on_fit_start(self, ctx: RunContext) -> None:
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def on_train_step_end(self, ctx: RunContext) -> None:
        if not self.save_last and not self.every_n_steps:
            return
        step = ctx.state.global_step
        if self.every_n_steps is not None and step % self.every_n_steps == 0:
            self._save(ctx, name=f"step_{step}.pt")
        if self.save_last:
            self._save(ctx, name="last.pt")

    def _save(self, ctx: RunContext, *, name: str) -> None:
        ckpt = capture_checkpoint(
            model=ctx.model,
            optimizer=ctx.optimizer,
            scheduler=ctx.scheduler,
            accelerator_state=ctx.accelerator.state_dict(),
            callbacks=ctx.callbacks,
            train_state=ctx.state,
            include_cuda=(ctx.accelerator.device.type == "cuda"),
            meta={"run_name": ctx.run_name},
        )
        save_checkpoint(Path(self.save_dir) / name, ckpt)
