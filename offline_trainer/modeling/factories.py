"""Model factory interfaces and policy_constructor adapter."""
from __future__ import annotations

import os
import sys
from typing import Any, Protocol

import torch

try:
    from model_constructor import build_model
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(repo_root, "policy_constructor")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)
    from model_constructor import build_model


class ModelFactory(Protocol):
    """Builds torch modules from a model config dict."""

    def build(self, model_cfg: dict[str, Any]) -> torch.nn.Module: ...


class PolicyConstructorModelFactory:
    """Adapter for policy_constructor's build_model API."""

    def build(self, model_cfg: dict[str, Any]) -> torch.nn.Module:
        if "config_path" in model_cfg:
            return build_model(model_cfg["config_path"])
        elif "config" in model_cfg:
            return build_model(model_cfg["config"])
        elif len(model_cfg.keys()) > 1:
            models = {}
            for k, v in model_cfg.items():
                models[k] = build_model(v)
            return models
        else:
            raise ValueError("Model building inside PolicyConstructorModelFactory not supported")
