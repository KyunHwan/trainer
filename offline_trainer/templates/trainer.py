""" Trainer template """
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any

import torch
from torch import nn


@runtime_checkable
class Trainer(Protocol):
    """ Interface ensuring every trainer has a training loop step. """
    def __init__(self, 
                 models: nn.ModuleDict[str, nn.Module], 
                 optimizers: dict[str, torch.optim.Optimizer],
                 loss_fn: nn.Module): ...
    def train_step(self, data: dict[str, Any]) -> dict[str, Any]: 
        """ Should process data and return a dict of metrics/loss """
        
    def save_checkpoints(self, save_dir: str, epoch: int) -> None: ...