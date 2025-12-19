"""Seeding utilities for deterministic runs."""
from __future__ import annotations

import os
import random
from typing import Any

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, and torch RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers based on the initial seed."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    if np is not None:
        np.random.seed(seed + worker_id)
