"""Tests avoid network access and GPU dependencies."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tfs.config import PROFILE_NAMES, ModelConfig, TrainConfig, load_profile


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_named_configs_load(profile: str) -> None:
    config = load_profile(profile)
    assert config.profile == profile
    assert config.model.n_embd % config.model.n_head == 0
    assert config.model.parameter_count() > 0


def test_parameter_profile_ranges() -> None:
    assert 1_000_000 <= load_profile("tiny").model.parameter_count() <= 3_000_000
    assert 10_000_000 <= load_profile("small").model.parameter_count() <= 30_000_000
    assert load_profile("gpt2-124m").model.parameter_count() == 124_439_808


def test_invalid_heads_rejected() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig(vocab_size=256, block_size=4, n_layer=2, n_head=3, n_embd=8)


def test_invalid_schedule_rejected() -> None:
    with pytest.raises(ValidationError, match="warmup_steps"):
        TrainConfig(
            device="cpu",
            batch_size=1,
            gradient_accumulation=1,
            max_steps=4,
            warmup_steps=5,
            learning_rate=0.001,
            min_learning_rate=0.0,
            seed=0,
        )


def test_unknown_profile_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        load_profile("invalid")


def test_missing_profile_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing configuration"):
        load_profile("tiny", tmp_path)


def test_config_does_not_ignore_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelConfig(vocab_size=256, block_size=4, n_layer=2, n_head=2, n_embd=8, invented_flag=True)
