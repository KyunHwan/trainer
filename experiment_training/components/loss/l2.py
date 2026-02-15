import torch
import torch.nn as nn
from trainer.trainer.registry import LOSS_BUILDER_REGISTRY

@LOSS_BUILDER_REGISTRY.register("l2_loss")
class L2LossFactory:
    def build(self, reduction: str = "sum") -> nn.Module:
        return L2Loss(reduction=reduction)



class L2Loss(nn.Module):
    def __init__(self, reduction: str = "none"):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)

