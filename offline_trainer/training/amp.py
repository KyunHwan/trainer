"""Automatic mixed precision helpers."""
from __future__ import annotations

import torch


def get_grad_scaler(enabled: bool, device: torch.device) -> torch.cuda.amp.GradScaler:
    """Return a GradScaler enabled only for CUDA."""
    use_amp = enabled and device.type == "cuda"
    return torch.cuda.amp.GradScaler(enabled=use_amp)
