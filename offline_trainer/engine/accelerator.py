from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Literal

import torch

Precision = Literal["fp32", "fp16-mixed", "bf16-mixed"]


@dataclass
class AcceleratorState:
    scaler: dict[str, Any] | None = None


class SingleDeviceAccelerator:
    def __init__(self, *, device: torch.device, precision: Precision) -> None:
        self._device = device
        self._precision = precision
        self._scaler: torch.cuda.amp.GradScaler | None = None
        if precision == "fp16-mixed":
            self._scaler = torch.cuda.amp.GradScaler(enabled=True)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_main_process(self) -> bool:
        return True

    @property
    def scaler(self) -> torch.cuda.amp.GradScaler | None:
        return self._scaler

    def autocast(self):
        if self._precision == "fp32":
            return contextlib.nullcontext()
        if self._precision == "fp16-mixed":
            return torch.autocast(device_type=self._device.type, dtype=torch.float16)
        return torch.autocast(device_type=self._device.type, dtype=torch.bfloat16)

    def backward(self, loss: torch.Tensor) -> None:
        if self._scaler is None:
            loss.backward()
            return
        self._scaler.scale(loss).backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self._scaler is None:
            return
        self._scaler.unscale_(optimizer)

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self._scaler is None:
            optimizer.step()
            return
        self._scaler.step(optimizer)
        self._scaler.update()

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def clip_grad_norm_(self, params, max_norm: float) -> float:
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))

    def state_dict(self) -> AcceleratorState:
        if self._scaler is None:
            return AcceleratorState(scaler=None)
        return AcceleratorState(scaler=self._scaler.state_dict())

    def load_state_dict(self, state: AcceleratorState) -> None:
        if self._scaler is None or state.scaler is None:
            return
        self._scaler.load_state_dict(state.scaler)

