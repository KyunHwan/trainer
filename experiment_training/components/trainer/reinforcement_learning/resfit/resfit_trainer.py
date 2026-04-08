"""Resfit trainer: residual actor-critic for online RL.

Data flow:
  1. Critic update: Q(s, a_gt) trained with TD target from target Q-network.
     Target Q uses Polyak-averaged weights. Reward comes from replay buffer.
  2. Actor update: residual_actor(state, base_action) → delta.
     Combined action = base_action + delta.
     Actor loss = -Q(s, combined_action).mean() (maximize Q-value).
     Gradients flow through actor only (Q params frozen for this step).

Register via plugins in the experiment YAML:
  - "experiment_training.components.trainer.reinforcement_learning.resfit.resfit_trainer"
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils.actor_trainer import Actor_Trainer
from .utils.critic_trainer import Critic_Trainer

from trainer.trainer.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("resfit_trainer")
class ResfitTrainer(nn.Module):
    """Actor-critic trainer for residual policy learning.

    Models expected in ``models`` dict:
      - ``resfit_residual_actor``: GraphModel wrapping Residual_Actor
      - ``resfit_q_function``:     GraphModel wrapping resfit Q_Function

    The Q-function is monolithic (includes its own Resnet34Group + preprocessor).
    The actor is monolithic (includes its own Resnet34Group + preprocessor + MLP).
    """

    def __init__(
        self,
        *,
        models: nn.ModuleDict,
        optimizers: dict[str, torch.optim.Optimizer],
        loss: nn.Module | None,
        device: torch.device,
    ):
        super().__init__()
        self.models = models
        self.optimizers = optimizers
        self.loss = loss  # not used — losses are computed inline
        self.device = device

        self.actor_trainer = Actor_Trainer(
            self.models,
            self.device,
        )

        self.Q_trainer = Critic_Trainer(
            self.models,
            self.device,
        )

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(
        self,
        data: dict[str, Any],
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = {}
        loss_Q_trainer = {}
        loss_actor_trainer = {}

        self._ready_train()
        self._zero_grad()
        
        # Critic Update
        loss_Q_trainer['Residual Q Loss'] = self.Q_trainer(data, stats)
        self._backward(loss_Q_trainer)
        detached_loss_action_Q_trainer = self._clip_get_grad_norm(loss_Q_trainer, clip_val=1.0)
        self.optimizers['resfit_q_function'].step()
        self._zero_grad()
        self.Q_trainer.update_target()
        detached_loss_action_Q_trainer = self._detached_loss(detached_loss_action_Q_trainer)

        metrics.update(detached_loss_action_Q_trainer)

        if data['iter'] != 0 and data['iter'] % 10 == 0:
            self._ready_train()
            self._zero_grad()
            loss_actor_trainer['Residual Q Value'] = self.actor_trainer(data, stats)
            self._backward(loss_actor_trainer)
            detached_loss_actor_trainer = self._clip_get_grad_norm(loss_actor_trainer, clip_val=1.0)
            self.optimizers['resfit_residual_actor'].step()
            self._zero_grad()
            detached_loss_actor_trainer = self._detached_loss(detached_loss_actor_trainer)

            metrics.update(detached_loss_actor_trainer)
        return metrics

    # ------------------------------------------------------------------
    # Boilerplate (same pattern as DSRLOpenPITrainer)
    # ------------------------------------------------------------------

    def _ready_train(self):
        for key in self.optimizers:
            self.models[key].train()
            if hasattr(self.optimizers[key], "train"):
                self.optimizers[key].train()

    def _zero_grad(self):
        for key in self.optimizers:
            self.optimizers[key].zero_grad(set_to_none=True)

    def _backward(self, loss: dict[str, Any]):
        for v in loss.values():
            if isinstance(v, torch.Tensor):
                v.backward()

    def _step(self):
        for key in self.optimizers:
            self.optimizers[key].step()

    def _detached_loss(self, loss: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.detach().item() if isinstance(v, torch.Tensor) else v
            for k, v in loss.items()
        }

    def _clip_get_grad_norm(
        self, loss: dict[str, Any], clip_val: float = float("inf")
    ) -> dict[str, Any]:
        for name in self.models:
            if name in self.optimizers:
                loss[f"{name} grad_norm"] = (
                    torch.nn.utils.clip_grad_norm_(
                        self.models[name].parameters(), max_norm=clip_val
                    )
                    .detach()
                    .item()
                )
        return loss

    def _get_lr(self, loss: dict[str, Any]) -> dict[str, Any]:
        for name in self.models:
            if name in self.optimizers:
                loss[f"{name} lr"] = self.optimizers[name].param_groups[0]["lr"]
        return loss
