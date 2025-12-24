
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from offline_trainer.registry import TRAINER_REGISTRY

from typing import Any

@TRAINER_REGISTRY.register("il_flow_matching_trainer")
class FlowMatchingILTrainer(nn.Module):
    def __init__(self,
                 *
                 models: nn.ModuleDict[str, nn.Modules], 
                 optimizers: nn.ModuleDict[str, torch.optim.Optimizer],
                 loss: nn.Module,):
        super().__init__()
        self.models=models
        self.optimizers=optimizers
        self.loss=loss

    def train_step(self, data):
        return None
    
    def save_checkpoints(self, save_dir, epoch):
        """
            create a subfolder with epoch in its name
        """
        return None

    def load_checkpoints(self, load_dir):
        return None