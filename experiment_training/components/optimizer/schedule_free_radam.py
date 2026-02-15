import torch
from torch import nn
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from trainer.trainer.registry import OPTIMIZER_BUILDER_REGISTRY

@OPTIMIZER_BUILDER_REGISTRY.register("schedule_free_radam")
class RAdamScheduleFreeFactory(nn.Module):
    def __init__(self, lr, betas, silent_sgd_phase):
        self.lr = lr
        self.betas = tuple(betas)
        self.silent_sgd_phase = silent_sgd_phase
    
    def build(self, 
              params
             ) -> torch.optim.Optimizer:
        
        return radam_free(params, 
                          lr=self.lr, 
                          betas=self.betas, 
                          silent_sgd_phase=self.silent_sgd_phase)