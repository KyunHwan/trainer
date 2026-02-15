import math
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from trainer.trainer.registry import OPTIMIZER_BUILDER_REGISTRY

class AdamWWithOneCycle(AdamW):
    """
    AdamW optimizer that owns a OneCycleLR scheduler and saves/loads both states
    via optimizer.state_dict() / optimizer.load_state_dict().
    """

    def __init__(
        self,
        params,
        lr: float,
        betas: list[float, float] = [0.9, 0.999],
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        # OneCycleLR args
        max_lr: float,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",  # "cos" or "linear"
        cycle_momentum: bool = False,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1,
    ):
        # Initialize the optimizer "properly" (this sets up param_groups, state, etc.)
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # OneCycleLR requires either total_steps OR (epochs + steps_per_epoch)
        if total_steps is None and (epochs is None or steps_per_epoch is None):
            raise ValueError(
                "OneCycleLR needs either total_steps, or both (epochs and steps_per_epoch)."
            )

        self.scheduler = OneCycleLR(
            optimizer=self,  # IMPORTANT: argument name is 'optimizer', not 'optimize'
            max_lr=max_lr,
            total_steps=total_steps,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )

    @torch.no_grad()
    def step(self, closure=None):
        """
        Step optimizer AND scheduler together.
        For OneCycleLR this is typically what you want (scheduler per optimizer-step).
        """
        loss = super().step(closure=closure)
        self.scheduler.step()
        return loss

    def state_dict(self):
        """
        Return the standard optimizer state dict PLUS scheduler state.
        Keeps 'state' + 'param_groups' keys intact for compatibility.
        """
        sd = super().state_dict()
        sd["scheduler"] = self.scheduler.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        """
        Load optimizer state and scheduler state from a combined dict.
        """
        sched_sd = state_dict.get("scheduler", None)

        # Make a dict containing ONLY optimizer keys; don't mutate caller's dict.
        opt_sd = {k: v for k, v in state_dict.items() if k != "scheduler"}

        super().load_state_dict(opt_sd)

        if sched_sd is not None:
            self.scheduler.load_state_dict(sched_sd)

@OPTIMIZER_BUILDER_REGISTRY.register("adamw_cosine_schedule")
class AdamW_Cosine_Scheduler_Builder(nn.Module):
    def __init__(self, 
                 lr: float, 
                 betas: list[float], 
                 eps: float,
                 weight_decay: float,

                 max_lr: float,
                 total_steps: int | None,
                 epochs: int | None,
                 steps_per_epoch: int | None,
                 pct_start: float,
                 anneal_strategy: str,
                 cycle_momentum: bool,
                 base_momentum: float,
                 max_momentum: float,
                 div_factor:float, 
                 final_div_factor: float,
                 three_phase: bool,
                 last_epoch: int,
                 ):
        # Optimizer
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # CosineLR
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum 
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.last_epoch = last_epoch
    
    def build(self, 
              params
             ) -> AdamWWithOneCycle:
        
       return AdamWWithOneCycle(
            params=params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            max_lr=self.max_lr,
            total_steps=self.total_steps,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            cycle_momentum=self.cycle_momentum,
            base_momentum=self.base_momentum,
            max_momentum=self.max_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            three_phase=self.three_phase,
            last_epoch=self.last_epoch,
        )

