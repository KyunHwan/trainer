"""Training loop implementation used by DefaultTrainer."""
from __future__ import annotations

import contextlib
import itertools
from typing import Any

import torch

from offline_trainer.components.callbacks import Callback
from offline_trainer.components.loggers import Logger
from offline_trainer.components.loss import Loss
from offline_trainer.components.metrics import Metrics
from offline_trainer.components.optim import OptimizerFactory
from offline_trainer.components.sched import SchedulerFactory
from offline_trainer.config.schemas import ExperimentConfig
from offline_trainer.training.amp import get_grad_scaler
from offline_trainer.training.checkpointing import CheckpointManager
from offline_trainer.training.state import TrainState
from offline_trainer.utils.device import move_to_device, select_device
from offline_trainer.utils.selection import select
from offline_trainer.utils.seed import set_global_seed


class _EMA:
    def __init__(self, params: list[torch.nn.Parameter], decay: float) -> None:
        self.decay = decay
        self.params = [p for p in params if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]

    def update(self) -> None:
        for shadow, param in zip(self.shadow, self.params):
            shadow.mul_(self.decay).add_(param.data, alpha=1 - self.decay)


def fit_loop(
    *,
    models: dict[str, torch.nn.Module],
    datamodule: Any,
    config: ExperimentConfig,
    optimizer_factory: OptimizerFactory,
    scheduler_factory: SchedulerFactory | None,
    loss_fn: Loss,
    metrics: list[Metrics],
    callbacks: list[Callback],
    loggers: list[Logger],
) -> None:
    device = select_device(config.device)
    if config.seed is not None:
        set_global_seed(config.seed, deterministic=config.deterministic)

    for model in models.values():
        model.to(device)

    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()

    all_params = list(itertools.chain.from_iterable(model.parameters() for model in models.values()))
    optimizer = optimizer_factory.build(all_params)
    scheduler = scheduler_factory.build(optimizer) if scheduler_factory is not None else None

    scaler = get_grad_scaler(config.train.amp, device)
    checkpoint_mgr = CheckpointManager(
        save_dir=config.train.checkpoint.save_dir,
        save_every_n_steps=config.train.checkpoint.save_every_n_steps,
        save_last=config.train.checkpoint.save_last,
    )

    state = TrainState(
        step=0,
        epoch=0,
        max_steps=config.train.max_steps,
        max_epochs=config.train.max_epochs,
    )

    if config.train.checkpoint.resume_from:
        payload = checkpoint_mgr.load(config.train.checkpoint.resume_from)
        for name, model in models.items():
            if name in payload.get("models", {}):
                model.load_state_dict(payload["models"][name])
        if payload.get("optimizer") and optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") and scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler") and scaler is not None:
            scaler.load_state_dict(payload["scaler"])
        state = TrainState.from_dict(payload.get("train_state", {}))
        state.max_steps = config.train.max_steps
        state.max_epochs = config.train.max_epochs
        cb_states = payload.get("callbacks", {})
        for idx, cb in enumerate(callbacks):
            key = str(idx)
            if key in cb_states:
                cb.load_state_dict(cb_states[key])

    for metric in metrics:
        metric.reset()

    for cb in callbacks:
        cb.on_train_start(state)

    accum = config.train.accumulate_grad_batches
    autocast_ctx = (
        torch.cuda.amp.autocast if device.type == "cuda" else contextlib.nullcontext
    )

    ema = None
    if config.train.ema.enabled:
        ema = _EMA(list(itertools.chain.from_iterable(m.parameters() for m in models.values())), config.train.ema.decay)

    optimizer.zero_grad(set_to_none=True)
    last_loss = None
    micro_step = 0

    stop_training = False
    for epoch in range(state.epoch, config.train.max_epochs):
        state.epoch = epoch
        for model in models.values():
            model.train()
        for batch in train_loader:
            if state.max_steps is not None and state.step >= state.max_steps:
                stop_training = True
                break

            batch = move_to_device(batch, device)
            model_input = select(batch, config.train.model_input)

            with autocast_ctx():
                outputs = models["main"](model_input)
                loss = loss_fn(batch, outputs)
                loss_to_backprop = loss / accum

            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            last_loss = float(loss.detach().cpu().item())

            for metric in metrics:
                metric.update(batch, outputs)

            micro_step += 1
            if micro_step % accum == 0:
                if config.train.gradient_clip_val is not None and config.train.gradient_clip_val > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        all_params, config.train.gradient_clip_val
                    )

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                if ema is not None:
                    ema.update()

                state.step += 1

                if state.step % config.train.log_every_n_steps == 0:
                    metrics_dict: dict[str, Any] = {"train/loss": last_loss}
                    for metric in metrics:
                        metrics_dict.update(metric.compute())
                    state.metrics = {k: float(v) for k, v in metrics_dict.items()}
                    for logger in loggers:
                        logger.log(metrics_dict, step=state.step)

                for cb in callbacks:
                    cb.on_step_end(state)

                checkpoint_mgr.maybe_save(
                    step=state.step,
                    models=models,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    state=state,
                    callbacks=callbacks,
                )

                if state.max_steps is not None and state.step >= state.max_steps:
                    stop_training = True
                    break

        if stop_training:
            break

    if config.train.checkpoint.save_last:
        checkpoint_mgr.save(
            path=checkpoint_mgr._last_path(),
            models=models,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            callbacks=callbacks,
        )

    for logger in loggers:
        logger.flush()
