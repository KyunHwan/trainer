from __future__ import annotations

import os
import random
from typing import Any

import torch


def seed_everything(seed: int, *, deterministic: bool, include_cuda: bool) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if include_cuda:
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_state(*, include_cuda: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
    }
    if include_cuda:
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None
    try:
        import random as pyrandom

        state["python"] = pyrandom.getstate()
    except Exception:
        state["python"] = None
    return state


def set_rng_state(state: dict[str, Any], *, include_cuda: bool) -> None:
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if include_cuda and "cuda" in state and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("numpy") is not None:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(state["numpy"])
        except Exception:
            pass
    if state.get("python") is not None:
        try:
            import random as pyrandom

            pyrandom.setstate(state["python"])
        except Exception:
            pass
