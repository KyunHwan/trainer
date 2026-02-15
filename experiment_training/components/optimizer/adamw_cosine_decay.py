# import math
# import torch
# import torch.nn as nn
# from typing import Optional
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import _LRScheduler
# from trainer.registry import OPTIMIZER_BUILDER_REGISTRY


# class WarmupCosineDecayLR(_LRScheduler):
#     r"""
#     Warmup (linear/cosine) followed by cosine decay (JAX-style) with alpha + exponent.

#     Let base_lr be each param group's "initial_lr" at scheduler creation time.

#     Step index t is taken as self.last_epoch (PyTorch sets last_epoch=0 during scheduler init).

#     Warmup:
#       for t <= warmup_steps:
#         lr(t) = base_lr * warmup_factor(t)
#       where warmup_factor goes from warmup_start_factor -> 1.0

#       linear:
#         warmup_factor(t) = s + (1 - s) * (t / W)
#       cosine:
#         warmup_factor(t) = s + (1 - s) * 0.5 * (1 - cos(pi * t / W))

#     Decay (after warmup):
#       let d = t - warmup_steps
#       clamp d to [0, decay_steps]
#       cosine_decay = 0.5 * (1 + cos(pi * d / decay_steps))
#       decayed = (1 - alpha) * (cosine_decay ** exponent) + alpha
#       lr(t) = base_lr * decayed

#     After warmup_steps + decay_steps, lr stays at alpha * base_lr.

#     Notes on step timing:
#       - In modern PyTorch, _LRScheduler calls step() once during __init__,
#         so the LR for t=0 is applied immediately (good for warmup: step 0 uses warmup_start_factor).
#       - With your pattern (calling scheduler.step() after optimizer.step()),
#         step k uses lr(t=k).
#     """

#     def __init__(
#         self,
#         optimizer: torch.optim.Optimizer,
#         *,
#         total_steps: int,
#         warmup_steps: int = 0,
#         warmup_strategy: str = "linear",   # "linear" or "cosine"
#         warmup_start_factor: float = 0.0, # multiplier of base_lr at t=0
#         alpha: float = 0.0,
#         exponent: float = 1.0,
#         last_epoch: int = -1,
#     ):
#         if total_steps <= 0:
#             raise ValueError(f"total_steps must be > 0, got {total_steps}.")
#         if warmup_steps < 0:
#             raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
#         if warmup_steps >= total_steps:
#             raise ValueError(
#                 f"warmup_steps must be < total_steps, got warmup_steps={warmup_steps}, total_steps={total_steps}."
#             )
#         if not (0.0 <= warmup_start_factor <= 1.0):
#             raise ValueError(f"warmup_start_factor should be in [0, 1], got {warmup_start_factor}.")
#         if not (0.0 <= alpha <= 1.0):
#             raise ValueError(f"alpha should be in [0, 1], got {alpha}.")
#         if exponent <= 0:
#             raise ValueError(f"exponent must be > 0, got {exponent}.")

#         warmup_strategy = warmup_strategy.lower().strip()
#         if warmup_strategy not in ("linear", "cosine"):
#             raise ValueError(f"warmup_strategy must be 'linear' or 'cosine', got {warmup_strategy!r}.")

#         self.total_steps = int(total_steps)
#         self.warmup_steps = int(warmup_steps)
#         self.decay_steps = int(total_steps - warmup_steps)

#         self.warmup_strategy = warmup_strategy
#         self.warmup_start_factor = float(warmup_start_factor)

#         self.alpha = float(alpha)
#         self.exponent = float(exponent)

#         super().__init__(optimizer, last_epoch=last_epoch)

#     def _warmup_factor(self, t: int) -> float:
#         # Warmup is defined for t in [0, warmup_steps]
#         if self.warmup_steps == 0:
#             return 1.0

#         # clamp t into [0, warmup_steps]
#         t = max(0, min(t, self.warmup_steps))
#         progress = t / float(self.warmup_steps)  # 0 -> 1

#         s = self.warmup_start_factor
#         if self.warmup_strategy == "linear":
#             return s + (1.0 - s) * progress
#         else:  # cosine warmup
#             return s + (1.0 - s) * 0.5 * (1.0 - math.cos(math.pi * progress))

#     def _decay_factor(self, d: int) -> float:
#         # Decay is defined for d in [0, decay_steps]
#         d = max(0, min(d, self.decay_steps))
#         cosine_decay = 0.5 * (1.0 + math.cos(math.pi * d / float(self.decay_steps)))
#         return (1.0 - self.alpha) * (cosine_decay ** self.exponent) + self.alpha

#     def get_lr(self):
#         t = int(self.last_epoch)

#         if self.warmup_steps > 0 and t <= self.warmup_steps:
#             factor = self._warmup_factor(t)
#             return [base_lr * factor for base_lr in self.base_lrs]

#         # After warmup, start cosine decay with d = t - warmup_steps
#         d = t - self.warmup_steps
#         factor = self._decay_factor(d)
#         return [base_lr * factor for base_lr in self.base_lrs]




# class AdamWWithWarmupCosineDecay(AdamW):
#     """
#     AdamW optimizer that owns a WarmupCosineDecayLR scheduler and saves/loads both states
#     via optimizer.state_dict() / optimizer.load_state_dict().
#     """

#     def __init__(
#         self,
#         params,
#         lr: float,
#         betas=(0.9, 0.999),
#         eps: float = 1e-8,
#         weight_decay: float = 0.0,
#         *,
#         total_steps: Optional[int] = None,
#         epochs: Optional[int] = None,
#         steps_per_epoch: Optional[int] = None,
#         warmup_steps: int = 0,
#         warmup_strategy: str = "linear",   # "linear" or "cosine"
#         warmup_start_factor: float = 0.0,  # e.g. 1/div_factor if mimicking OneCycle start
#         alpha: float = 0.0,                # min_lr = alpha * lr
#         exponent: float = 1.0,
#         last_epoch: int = -1,
#     ):
#         super().__init__(
#             params=params,
#             lr=lr,  # this is the peak/base lr the schedule scales from
#             betas=betas,
#             eps=eps,
#             weight_decay=weight_decay,
#         )

#         if total_steps is None:
#             if epochs is None or steps_per_epoch is None:
#                 raise ValueError(
#                     "Need either total_steps, or both (epochs and steps_per_epoch)."
#                 )
#             total_steps = int(epochs) * int(steps_per_epoch)

#         self.scheduler = WarmupCosineDecayLR(
#             optimizer=self,
#             total_steps=total_steps,
#             warmup_steps=warmup_steps,
#             warmup_strategy=warmup_strategy,
#             warmup_start_factor=warmup_start_factor,
#             alpha=alpha,
#             exponent=exponent,
#             last_epoch=last_epoch,
#         )

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = super().step(closure=closure)
#         self.scheduler.step()
#         return loss

#     def state_dict(self):
#         sd = super().state_dict()
#         sd["scheduler"] = self.scheduler.state_dict()
#         return sd

#     def load_state_dict(self, state_dict):
#         sched_sd = state_dict.get("scheduler", None)
#         opt_sd = {k: v for k, v in state_dict.items() if k != "scheduler"}

#         super().load_state_dict(opt_sd)

#         if sched_sd is not None:
#             self.scheduler.load_state_dict(sched_sd)

# @OPTIMIZER_BUILDER_REGISTRY.register("adamw_warmup_cosine_decay")
# class AdamW_Warmup_CosineDecay_Builder(nn.Module):
#     def __init__(
#         self,
#         lr: float,               # you can ignore this and use max_lr, or keep for backward compat
#         betas,
#         eps: float,
#         weight_decay: float,
#         *,
#         max_lr: float,
#         total_steps: int | None,
#         epochs: int | None,
#         steps_per_epoch: int | None,
#         pct_start: float = 0.1,          # warmup fraction
#         warmup_strategy: str = "linear", # "linear" or "cosine"
#         div_factor: float = 25.0,
#         final_div_factor: float = 1e4,
#         alpha: float | None = None,      # optional override
#         exponent: float = 1.0,
#         last_epoch: int = -1,
#     ):
#         super().__init__()
#         self.betas = betas
#         self.eps = eps
#         self.weight_decay = weight_decay

#         self.max_lr = max_lr
#         self.total_steps = total_steps
#         self.epochs = epochs
#         self.steps_per_epoch = steps_per_epoch

#         self.pct_start = pct_start
#         self.warmup_strategy = warmup_strategy
#         self.div_factor = div_factor
#         self.final_div_factor = final_div_factor
#         self.alpha = alpha
#         self.exponent = exponent
#         self.last_epoch = last_epoch

#     def build(self, params) -> AdamWWithWarmupCosineDecay:
#         # Derive total_steps if needed
#         total_steps = self.total_steps
#         if total_steps is None:
#             if self.epochs is None or self.steps_per_epoch is None:
#                 raise ValueError("Need either total_steps or (epochs and steps_per_epoch).")
#             total_steps = int(self.epochs) * int(self.steps_per_epoch)

#         warmup_steps = int(self.pct_start * total_steps)

#         warmup_start_factor = 1.0 / float(self.div_factor)

#         # If alpha not explicitly provided, mimic OneCycle-ish final LR magnitude
#         alpha = self.alpha
#         if alpha is None:
#             alpha = 1.0 / (float(self.div_factor) * float(self.final_div_factor))

#         return AdamWWithWarmupCosineDecay(
#             params=params,
#             lr=self.max_lr,  # peak/base lr
#             betas=self.betas,
#             eps=self.eps,
#             weight_decay=self.weight_decay,
#             total_steps=total_steps,
#             warmup_steps=warmup_steps,
#             warmup_strategy=self.warmup_strategy,
#             warmup_start_factor=warmup_start_factor,
#             alpha=float(alpha),
#             exponent=self.exponent,
#             last_epoch=self.last_epoch,
#         )

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from trainer.trainer.registry import OPTIMIZER_BUILDER_REGISTRY


# Prefer the public base class in modern PyTorch; fall back for older versions.
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # pragma: no cover
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def _lr_to_float(v) -> float:
    """Safely cast optimizer param_group['lr'] which may be a Tensor."""
    if isinstance(v, torch.Tensor):
        return float(v.item())
    return float(v)


class WarmupCosineLR(LRScheduler):
    """
    Linear warmup -> cosine decay, using absolute LR endpoints.

    Required semantics:
      - total_steps: total number of optimizer updates.
      - warmup_steps: number of updates used for linear warmup.
      - start_lr: LR for update step 0 (first optimizer update).
      - end_lr: LR for final scheduled step (step total_steps-1).

    Notes:
      - Step the scheduler once per optimizer update.
      - In modern PyTorch, scheduler __init__ performs an initial step() that sets the
        LR for step 0. This implementation is written to be correct under that behavior.
      - LR is clamped after schedule ends (stays at end_lr).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        start_lr: float,
        end_lr: float,
        last_epoch: int = -1,
    ):
        total_steps = int(total_steps)
        warmup_steps = int(warmup_steps)
        start_lr = float(start_lr)
        end_lr = float(end_lr)

        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}.")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
        if warmup_steps >= total_steps:
            # You asked for warmup then cosine decay; that requires at least 1 decay step.
            raise ValueError(
                f"warmup_steps must be < total_steps, got warmup_steps={warmup_steps}, total_steps={total_steps}."
            )

        if start_lr < 0.0:
            raise ValueError(f"start_lr must be >= 0, got {start_lr}.")
        if end_lr < 0.0:
            raise ValueError(f"end_lr must be >= 0, got {end_lr}.")

        # IMPORTANT:
        # LRScheduler.__init__ will call step() once, which calls get_lr().
        # So anything get_lr() touches must be initialized BEFORE super().__init__().
        peak_lrs = [_lr_to_float(g["lr"]) for g in optimizer.param_groups]
        if not peak_lrs:
            raise ValueError("Optimizer has no param_groups; cannot create LR scheduler.")

        peak0 = peak_lrs[0]
        if peak0 <= 0.0:
            raise ValueError(f"Optimizer peak lr must be > 0, got {peak0} for group 0.")

        # Validate against "peak" meaning: start and end should not exceed peak.
        # (You can relax this if you ever want non-monotonic schedules.)
        if start_lr > peak0 + 1e-12:
            raise ValueError(f"start_lr must be <= peak_lr; start_lr={start_lr}, peak_lr={peak0}.")
        if end_lr > peak0 + 1e-12:
            raise ValueError(f"end_lr must be <= peak_lr; end_lr={end_lr}, peak_lr={peak0}.")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._decay_steps = total_steps - warmup_steps  # >= 1 due to warmup_steps < total_steps

        # Scale start/end across param groups proportionally to each group's peak LR.
        # This preserves relative LR ratios if the user uses per-group peak lrs.
        start_ratio = start_lr / peak0
        end_ratio = end_lr / peak0
        self._start_lrs = [p * start_ratio for p in peak_lrs]
        self._end_lrs = [p * end_ratio for p in peak_lrs]

        # Precompute denominators (avoid div-by-zero).
        self._warmup_den = max(1, warmup_steps - 1)
        self._decay_den = max(1, self._decay_steps - 1)

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # Clamp so LR stays at end_lr after schedule end.
        step = int(self.last_epoch)
        step = max(0, min(step, self.total_steps - 1))

        lrs = []
        for peak_lr, start_lr, end_lr in zip(self.base_lrs, self._start_lrs, self._end_lrs):
            # Warmup region: steps [0, warmup_steps-1]
            if self.warmup_steps > 0 and step < self.warmup_steps:
                if self.warmup_steps == 1:
                    # Only one warmup step => can't interpolate to peak within warmup.
                    lr = start_lr
                else:
                    progress = step / float(self._warmup_den)  # 0 -> 1
                    lr = start_lr + (peak_lr - start_lr) * progress

            # Decay region: steps [warmup_steps, total_steps-1]
            else:
                # Degenerate case: only 1 decay step (the last step).
                # We choose to land exactly on end_lr there.
                if self._decay_steps == 1:
                    lr = end_lr
                else:
                    d = step - self.warmup_steps  # 0 .. decay_steps-1
                    d = max(0, min(d, self._decay_steps - 1))
                    progress = d / float(self._decay_den)  # 0 -> 1
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
                    lr = end_lr + (peak_lr - end_lr) * cosine

            lrs.append(lr)

        return lrs


class AdamWWithWarmupCosine(AdamW):
    """
    AdamW optimizer that owns a WarmupCosineLR scheduler:
      - Calls scheduler.step() inside optimizer.step() (after the update).
      - Saves/loads scheduler state via optimizer.state_dict() / load_state_dict().

    IMPORTANT: Do not also step a scheduler externally in your training loop.
    """

    def __init__(
        self,
        params,
        *,
        peak_lr: float,
        total_steps: int,
        warmup_steps: int,
        start_lr: float,
        end_lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        peak_lr = float(peak_lr)
        if peak_lr <= 0.0:
            raise ValueError(f"peak_lr must be > 0, got {peak_lr}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")
        if not (isinstance(betas, (tuple, list)) and len(betas) == 2):
            raise ValueError(f"betas must be a tuple/list of length 2, got {betas!r}.")

        super().__init__(
            params=params,
            lr=peak_lr,
            betas=tuple(betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )

        self.scheduler = WarmupCosineLR(
            optimizer=self,
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            start_lr=float(start_lr),
            end_lr=float(end_lr),
            last_epoch=-1,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure=closure)
        self.scheduler.step()
        return loss

    def state_dict(self):
        sd = super().state_dict()
        sd["scheduler"] = self.scheduler.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        sched_sd = state_dict.get("scheduler")
        opt_sd = {k: v for k, v in state_dict.items() if k != "scheduler"}

        super().load_state_dict(opt_sd)
        if sched_sd is not None:
            self.scheduler.load_state_dict(sched_sd)


@OPTIMIZER_BUILDER_REGISTRY.register("adamw_warmup_cosine_decay")
class AdamW_WarmupCosine_Builder(nn.Module):
    """
    UI exposes ONLY:
      - peak_lr, betas, eps, weight_decay, total_steps, warmup_steps, start_lr, end_lr

    Schedule:
      warmup: start_lr -> peak_lr over warmup_steps
      decay:  peak_lr  -> end_lr  over (total_steps - warmup_steps)
    """

    def __init__(
        self,
        peak_lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        total_steps: int,
        warmup_steps: int,
        start_lr: float,
        end_lr: float,
    ):
        super().__init__()

        peak_lr = float(peak_lr)
        start_lr = float(start_lr)
        end_lr = float(end_lr)

        if peak_lr <= 0.0:
            raise ValueError(f"peak_lr must be > 0, got {peak_lr}.")
        if start_lr < 0.0:
            raise ValueError(f"start_lr must be >= 0, got {start_lr}.")
        if end_lr < 0.0:
            raise ValueError(f"end_lr must be >= 0, got {end_lr}.")
        if start_lr > peak_lr + 1e-12:
            raise ValueError(f"start_lr must be <= peak_lr; start_lr={start_lr}, peak_lr={peak_lr}.")
        if end_lr > peak_lr + 1e-12:
            raise ValueError(f"end_lr must be <= peak_lr; end_lr={end_lr}, peak_lr={peak_lr}.")

        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")

        if int(total_steps) <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}.")
        if int(warmup_steps) < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
        if int(warmup_steps) >= int(total_steps):
            raise ValueError(
                f"warmup_steps must be < total_steps, got warmup_steps={warmup_steps}, total_steps={total_steps}."
            )

        if not (isinstance(betas, (tuple, list)) and len(betas) == 2):
            raise ValueError(f"betas must be a tuple/list of length 2, got {betas!r}.")

        self.peak_lr = peak_lr
        self.betas = tuple(betas)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.start_lr = start_lr
        self.end_lr = end_lr

    def build(self, params) -> AdamWWithWarmupCosine:
        return AdamWWithWarmupCosine(
            params=params,
            peak_lr=self.peak_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            end_lr=self.end_lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


