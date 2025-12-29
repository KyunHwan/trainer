"""Pydantic schemas for experiment configuration."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator, model_validator

from offline_trainer.config.errors import ConfigError, ConfigValidationIssue


class ComponentSpec(BaseModel):
    """Common YAML component spec: {type, params}."""

    model_config = ConfigDict(extra="allow")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _type_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("type must be a non-empty string")
        return v


class OptimizerParams(BaseModel):
    """Optimizer params with optional lr validation and extra keys allowed."""

    model_config = ConfigDict(extra="allow")

    lr: float | None = None

    @field_validator("lr")
    @classmethod
    def _validate_lr(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("lr must be a float > 0")
        return v


class OptimizerSpec(ComponentSpec):
    """Optimizer component spec with minimal validation."""

    params: OptimizerParams = Field(default_factory=OptimizerParams)


class ComponentConfigPaths(RootModel[dict[str, str]]):
    """Mapping of component names to config file paths."""

    @model_validator(mode="after")
    def _validate_entries(self) -> "ComponentConfigPaths":
        if not self.root:
            raise ValueError("component_config_paths must contain at least one entry")
        for name, path in self.root.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("component_config_paths keys must be non-empty strings")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("component_config_paths values must be non-empty strings")
        return self

    def as_dict(self) -> dict[str, str]:
        return dict(self.root)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    find_unused_parameters: bool
    component_config_paths: ComponentConfigPaths


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    datamodule: ComponentSpec


class EMAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    decay: float = 0.999

    @field_validator("decay")
    @classmethod
    def _decay_range(cls, v: float) -> float:
        if v <= 0 or v >= 1:
            raise ValueError("decay must be in (0, 1)")
        return v


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_dir: str = "runs"
    save_every_n_steps: int | None = None
    save_last: bool = True
    resume_from: str | None = None

    @field_validator("save_every_n_steps")
    @classmethod
    def _save_every_valid(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("save_every_n_steps must be > 0")
        return v


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    trainer: ComponentSpec
    optimizer: OptimizerSpec
    scheduler: ComponentSpec = Field(default_factory=lambda: ComponentSpec(type="none"))
    loss: ComponentSpec
    metrics: list[ComponentSpec] = Field(default_factory=list)
    callbacks: list[ComponentSpec] = Field(default_factory=list)
    loggers: list[ComponentSpec] = Field(default_factory=lambda: [ComponentSpec(type="noop")])

    model_input: str | int | None = None
    max_epochs: int = 1
    max_steps: int | None = None
    accumulate_grad_batches: int = 1
    amp: bool = False
    gradient_clip_val: float | None = None
    log_every_n_steps: int = 1

    ema: EMAConfig = Field(default_factory=EMAConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @field_validator("max_epochs")
    @classmethod
    def _epochs_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_epochs must be > 0")
        return v

    @field_validator("max_steps")
    @classmethod
    def _steps_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_steps must be > 0")
        return v

    @field_validator("accumulate_grad_batches")
    @classmethod
    def _accum_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("accumulate_grad_batches must be > 0")
        return v

    @field_validator("gradient_clip_val")
    @classmethod
    def _clip_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("gradient_clip_val must be >= 0")
        return v

    @field_validator("log_every_n_steps")
    @classmethod
    def _log_steps_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("log_every_n_steps must be > 0")
        return v


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    plugins: list[str]
    seed: int | None = 123
    deterministic: bool = False
    device: str = "auto"

    model: ModelConfig
    data: DataConfig
    train: TrainConfig


def validate_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Validate raw config dict into ExperimentConfig or raise ConfigError."""
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        issues = []
        for err in exc.errors():
            path = _loc_to_path(err.get("loc", []))
            issues.append(
                ConfigValidationIssue(
                    error_path=path or "<root>",
                    error_message=err.get("msg", "Invalid value"),
                )
            )
        raise ConfigError(issues) from exc


def _loc_to_path(loc: tuple[Any, ...] | list[Any]) -> str:
    path = ""
    for item in loc:
        if isinstance(item, int):
            path += f"[{item}]"
        else:
            if path:
                path += "."
            path += str(item)
    return path
