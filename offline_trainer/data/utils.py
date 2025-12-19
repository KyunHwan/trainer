"""Data utilities for dataloaders."""
from __future__ import annotations

from typing import Callable

from offline_trainer.utils.seed import seed_worker


def worker_init_fn() -> Callable[[int], None]:
    """Return a worker init fn that seeds each worker deterministically."""
    return seed_worker
