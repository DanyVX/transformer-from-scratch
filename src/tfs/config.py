"""Validated named experiment configurations.

The parameter-count formula assumes GPT-2-like pre-LN blocks with 4x GELU MLPs.
It is a structural prediction, not a measured benchmark.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROFILE_NAMES = ("tiny", "small", "gpt2-124m")


class ModelConfig(BaseModel):
    """Architectural dimensions for a decoder-only Transformer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vocab_size: int = Field(ge=256)
    block_size: int = Field(gt=0)
    n_layer: int = Field(gt=0)
    n_head: int = Field(gt=0)
    n_embd: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bias: bool = True
    tie_embeddings: bool = True
    positional_encoding: Literal["learned", "rope"] = "learned"
    normalization: Literal["pre", "post"] = "pre"
    activation: Literal["gelu", "swiglu"] = "gelu"
    scale_residual_init: bool = True

    @model_validator(mode="after")
    def validate_heads(self) -> "ModelConfig":
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        return self

    def parameter_count(self) -> int:
        """Predict parameters for the configured decoder-only architecture.

        Each block: 12*C*C projection/MLP weights; two layer norms;
        9*C linear biases if enabled. The final layer norm is separate.
        """
        c = self.n_embd
        norm_params = c * (2 if self.bias else 1)
        extra_gate = 4 * c * c + (4 * c if self.bias else 0) if self.activation == "swiglu" else 0
        block = 12 * c * c + 2 * norm_params + (9 * c if self.bias else 0) + extra_gate
        position = self.block_size * c if self.positional_encoding == "learned" else 0
        untied_head = 0 if self.tie_embeddings else self.vocab_size * c
        return self.vocab_size * c + position + self.n_layer * block + norm_params + untied_head


class TrainConfig(BaseModel):
    """Validated training settings for the M5 trainer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: Literal["cpu", "cuda", "auto"] = "cpu"
    batch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    min_learning_rate: float = Field(ge=0.0)
    warmup_steps: int = Field(ge=0)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_schedule(self) -> "TrainConfig":
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        return self


class ExperimentConfig(BaseModel):
    """Validated model/train pairing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str
    model: ModelConfig
    train: TrainConfig


class RuntimeSettings(BaseSettings):
    """Optional runtime overrides from .env and environment variables."""

    model_config = SettingsConfigDict(env_prefix="TFS_", env_file=".env", extra="ignore")

    profile: str = "tiny"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    results_dir: Path = Path("results")
    json_logs: bool = False


def load_profile(name: str, config_dir: Path | None = None) -> ExperimentConfig:
    """Load a checked-in YAML config with strict schema validation.

    Prefer checkout YAML files, falling back to packaged YAML inside a wheel.
    """
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile {name!r}; choose from {', '.join(PROFILE_NAMES)}")
    if config_dir is not None:
        root = config_dir
    else:
        checkout = Path(__file__).resolve().parents[2] / "configs"
        bundled = Path(__file__).resolve().parent / "configs"
        root = checkout if checkout.is_dir() else bundled
    path = root / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing configuration: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    config = ExperimentConfig.model_validate(data)
    if config.profile != name:
        raise ValueError(f"Profile mismatch: expected {name!r}, found {config.profile!r}")
    return config
