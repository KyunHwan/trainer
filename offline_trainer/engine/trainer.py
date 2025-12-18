from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from offline_trainer.config.validate import RunConfig
from offline_trainer.deps.model_constructor import ConfigError, Registry
from offline_trainer.engine.accelerator import SingleDeviceAccelerator
from offline_trainer.engine.checkpoint import load_checkpoint, restore_checkpoint
from offline_trainer.engine.state import RunContext, TrainState
from offline_trainer.engine.tree import move_to_device, to_float


@dataclass
class DefaultTrainer:
    max_epochs: int = 1
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = None
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1

    def fit(
        self,
        *,
        model: torch.nn.Module,
        datamodule: Any,
        io: Any,
        loss: Any,
        optimizer_factory: Any,
        scheduler_factory: Any,
        metrics: list[Any],
        callbacks: list[Any],
        loggers: list[Any],
        registry: Registry,
        run: RunConfig,
    ) -> None:
        if self.max_epochs < 1:
            raise ConfigError("trainer.max_epochs must be >= 1", config_path=("trainer", "max_epochs"))
        if self.grad_accum_steps < 1:
            raise ConfigError("trainer.grad_accum_steps must be >= 1", config_path=("trainer", "grad_accum_steps"))
        if self.log_every_n_steps < 1:
            raise ConfigError("trainer.log_every_n_steps must be >= 1", config_path=("trainer", "log_every_n_steps"))
        if self.val_every_n_epochs < 1:
            raise ConfigError("trainer.val_every_n_epochs must be >= 1", config_path=("trainer", "val_every_n_epochs"))

        out_dir = Path(run.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        accelerator = SingleDeviceAccelerator(device=run.device, precision=run.precision)

        datamodule.setup("fit")
        train_state = TrainState()

        model.to(accelerator.device)
        optimizer = optimizer_factory.build(model, registry=registry)
        scheduler_bundle = scheduler_factory.build(optimizer, registry=registry) if scheduler_factory is not None else None
        scheduler = scheduler_bundle.scheduler if scheduler_bundle is not None else None

        if run.resume_from:
            accel_state = accelerator.state_dict()
            try:
                raw = load_checkpoint(run.resume_from)
                restored = restore_checkpoint(
                    raw,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    accelerator_state=accel_state,
                    callbacks=callbacks,
                    include_cuda=(accelerator.device.type == "cuda"),
                )
                accelerator.load_state_dict(accel_state)
                train_state = restored
                _optimizer_state_to_device(optimizer, accelerator.device)
            except Exception as exc:
                raise ConfigError(f"Failed to resume from checkpoint: {exc}", config_path=("run", "resume_from")) from exc

        ctx = RunContext(
            run_name=run.name,
            out_dir=str(out_dir),
            state=train_state,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_bundle=scheduler_bundle,
            accelerator=accelerator,
            datamodule=datamodule,
            io=io,
            loss=loss,
            metrics=metrics,
            callbacks=callbacks,
            loggers=loggers,
        )

        for lg in loggers:
            if hasattr(lg, "log_hparams"):
                lg.log_hparams({"run": run.__dict__, "trainer": self.__dict__})

        for cb in callbacks:
            if hasattr(cb, "on_fit_start"):
                cb.on_fit_start(ctx)

        try:
            for epoch in range(train_state.epoch, self.max_epochs):
                train_state.epoch = epoch
                self._train_epoch(ctx)

                ran_val = False
                if self.val_every_n_epochs and ((epoch + 1) % self.val_every_n_epochs == 0):
                    self._validate_epoch(ctx)
                    ran_val = True

                if scheduler_bundle is not None:
                    if scheduler_bundle.interval == "epoch":
                        if (epoch + 1) % scheduler_bundle.frequency == 0:
                            _step_scheduler(scheduler_bundle, ctx, metrics=ctx.last_val_logs)
                    if scheduler_bundle.interval == "metric":
                        if not ran_val:
                            raise ConfigError(
                                "scheduler.interval=metric requires validation to run on the same epoch",
                                config_path=("scheduler", "interval"),
                            )
                        if (epoch + 1) % scheduler_bundle.frequency == 0:
                            _step_scheduler(scheduler_bundle, ctx, metrics=ctx.last_val_logs)

                for cb in callbacks:
                    if hasattr(cb, "on_epoch_end"):
                        cb.on_epoch_end(ctx)

                train_state.micro_step_in_epoch = 0

            train_state.epoch = self.max_epochs
            train_state.micro_step_in_epoch = 0
        except BaseException as exc:
            for cb in callbacks:
                if hasattr(cb, "on_exception"):
                    cb.on_exception(ctx, exc)
            raise
        finally:
            for cb in callbacks:
                if hasattr(cb, "on_fit_end"):
                    cb.on_fit_end(ctx)
            for lg in loggers:
                if hasattr(lg, "close"):
                    lg.close()

    def _train_epoch(self, ctx: RunContext) -> None:
        model: torch.nn.Module = ctx.model
        model.train()
        ctx.optimizer.zero_grad(set_to_none=True)
        ctx.last_train_logs = {}

        train_loader = ctx.datamodule.train_dataloader()
        it = iter(train_loader)

        skip = ctx.state.micro_step_in_epoch
        for _ in range(skip):
            try:
                next(it)
            except StopIteration:
                return

        last_loss_tensor: torch.Tensor | None = None
        last_loss_logs: dict[str, Any] = {}

        for batch in it:
            batch_on_device = move_to_device(batch, ctx.accelerator.device)
            model_inputs, targets, _meta = ctx.io.split(batch_on_device, stage="train")

            with ctx.accelerator.autocast():
                outputs = model(*model_inputs.args, **model_inputs.kwargs)
                loss_out = ctx.loss(outputs=outputs, targets=targets, batch=batch_on_device, stage="train")
                loss_tensor = loss_out.loss
                if not torch.is_tensor(loss_tensor) or loss_tensor.ndim != 0:
                    raise ConfigError("loss must be a scalar torch.Tensor", config_path=("loss",))
                loss_scaled = loss_tensor / float(self.grad_accum_steps)

            ctx.accelerator.backward(loss_scaled)
            ctx.state.micro_step_in_epoch += 1
            last_loss_tensor = loss_tensor
            last_loss_logs = dict(getattr(loss_out, "logs", {}) or {})

            for m in ctx.metrics:
                m.update(outputs=outputs, targets=targets, batch=batch_on_device, stage="train")

            boundary = (ctx.state.micro_step_in_epoch % self.grad_accum_steps) == 0
            if boundary:
                self._optimizer_step(ctx, last_loss_tensor, last_loss_logs)

        remainder = ctx.state.micro_step_in_epoch % self.grad_accum_steps
        if remainder and last_loss_tensor is not None:
            self._optimizer_step(ctx, last_loss_tensor, last_loss_logs)

    def _optimizer_step(self, ctx: RunContext, loss_tensor: torch.Tensor, loss_logs: dict[str, Any]) -> None:
        if self.grad_clip_norm is not None:
            ctx.accelerator.unscale_(ctx.optimizer)
            ctx.accelerator.clip_grad_norm_(ctx.model.parameters(), self.grad_clip_norm)

        ctx.accelerator.optimizer_step(ctx.optimizer)
        ctx.accelerator.zero_grad(ctx.optimizer)
        ctx.state.global_step += 1

        step_logs = {"loss": to_float(loss_tensor)}
        for k, v in (loss_logs or {}).items():
            if isinstance(v, (float, int)):
                step_logs[str(k)] = float(v)
            elif torch.is_tensor(v):
                step_logs[str(k)] = to_float(v)

        ctx.last_train_logs = step_logs

        if ctx.scheduler_bundle is not None and ctx.scheduler_bundle.interval == "step":
            if ctx.state.global_step % ctx.scheduler_bundle.frequency == 0:
                _step_scheduler(ctx.scheduler_bundle, ctx, metrics=ctx.last_val_logs)

        if ctx.state.global_step % self.log_every_n_steps == 0:
            for lg in ctx.loggers:
                lg.log_metrics(step_logs, step=ctx.state.global_step, stage="train")

        for cb in ctx.callbacks:
            if hasattr(cb, "on_train_step_end"):
                cb.on_train_step_end(ctx)

    def _validate_epoch(self, ctx: RunContext) -> None:
        val_loader = ctx.datamodule.val_dataloader()
        if val_loader is None:
            return
        model: torch.nn.Module = ctx.model
        model.eval()
        ctx.last_val_logs = {}

        with torch.no_grad():
            for batch in val_loader:
                batch_on_device = move_to_device(batch, ctx.accelerator.device)
                model_inputs, targets, _meta = ctx.io.split(batch_on_device, stage="val")
                with ctx.accelerator.autocast():
                    outputs = model(*model_inputs.args, **model_inputs.kwargs)
                    loss_out = ctx.loss(outputs=outputs, targets=targets, batch=batch_on_device, stage="val")
                    loss_tensor = loss_out.loss
                    if not torch.is_tensor(loss_tensor) or loss_tensor.ndim != 0:
                        raise ConfigError("loss must be a scalar torch.Tensor", config_path=("loss",))
                for m in ctx.metrics:
                    m.update(outputs=outputs, targets=targets, batch=batch_on_device, stage="val")
                ctx.last_val_logs = {"val_loss": to_float(loss_tensor)}

        if ctx.state.global_step:
            for lg in ctx.loggers:
                lg.log_metrics(ctx.last_val_logs, step=ctx.state.global_step, stage="val")


def _step_scheduler(bundle: Any, ctx: RunContext, *, metrics: dict[str, float]) -> None:
    sched = bundle.scheduler
    if bundle.interval == "metric":
        if not bundle.monitor:
            raise ConfigError("scheduler.monitor is required when interval=metric", config_path=("scheduler", "monitor"))
        if bundle.monitor not in metrics:
            raise ConfigError(
                f"monitor key {bundle.monitor!r} not found in metrics",
                config_path=("scheduler", "monitor"),
            )
        val = metrics[bundle.monitor]
        sched.step(val)
        return
    sched.step()


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)
