"""LR boundaries, training checkpoints and exact replay on CPU."""

import numpy as np
import pytest
import torch

from tfs.config import ExperimentConfig, ModelConfig, TrainConfig
from tfs.data.shards import WindowSampler
from tfs.model.gpt import GPT
from tfs.train.schedule import learning_rate
from tfs.train.trainer import Trainer, seed_all


def experiment() -> ExperimentConfig:
    return ExperimentConfig(
        profile="test",
        model=ModelConfig(vocab_size=256, block_size=4, n_layer=1, n_head=2, n_embd=16),
        train=TrainConfig(
            batch_size=2,
            gradient_accumulation=2,
            max_steps=6,
            learning_rate=0.004,
            min_learning_rate=0.0004,
            warmup_steps=2,
            seed=19,
        ),
    )


def _make_trainer() -> Trainer:
    cfg = experiment()
    seed_all(cfg.train.seed)
    model = GPT(cfg.model)
    sampler = WindowSampler(np.arange(100, dtype=np.uint16), 4, 2, seed=cfg.train.seed)
    return Trainer(model, sampler, cfg)


def test_schedule_edges() -> None:
    assert learning_rate(0, 6, 2, 0.004, 0.0004) == 0.002
    assert learning_rate(1, 6, 2, 0.004, 0.0004) == 0.004
    assert learning_rate(5, 6, 2, 0.004, 0.0004) == pytest.approx(0.0004)
    assert learning_rate(6, 6, 2, 0.004, 0.0004) == 0.0004


def test_exact_cpu_resume(tmp_path) -> None:
    torch.set_num_threads(1)
    direct = _make_trainer()
    full_curve = direct.train_steps(6).copy()
    final = {k: v.detach().clone() for k, v in direct.model.state_dict().items()}

    interrupted = _make_trainer()
    interrupted.train_steps(3)
    checkpoint = tmp_path / "state.pt"
    interrupted.save(checkpoint)

    resumed = _make_trainer()
    resumed.resume(checkpoint)
    assert resumed.train_steps(3) == pytest.approx(full_curve, abs=1e-8, rel=1e-8)
    for name, param in final.items():
        torch.testing.assert_close(resumed.model.state_dict()[name], param, atol=0, rtol=0)


def test_refuse_config_mismatch(tmp_path) -> None:
    trainer = _make_trainer()
    path = tmp_path / "safe.pt"
    trainer.save(path)
    cfg = experiment().model_copy(update={"profile": "different"})
    other = Trainer(GPT(cfg.model), WindowSampler(np.arange(100), 4, 2), cfg)
    with pytest.raises(ValueError, match="mismatch"):
        other.resume(path)


def test_nonfinite_loss_does_not_advance_step(tmp_path) -> None:
    trainer = _make_trainer()
    path = tmp_path / "last_good.pt"
    trainer.save(path)
    with torch.no_grad():
        trainer.model.wte.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        trainer.train_steps(1, checkpoint=path)
    assert trainer.step == 0
    fresh = _make_trainer()
    fresh.resume(path)
    assert torch.isfinite(fresh.model.wte.weight).all()


def test_gradient_accumulation_equivalence() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(33)
    cfg = experiment().model
    small = GPT(cfg)
    large = GPT(cfg)
    large.load_state_dict(small.state_dict())
    tokens = torch.randint(0, cfg.vocab_size, (4, 4))
    labels = torch.randint(0, cfg.vocab_size, (4, 4))
    small.zero_grad()
    for ids, targets in zip(tokens.chunk(2), labels.chunk(2), strict=True):
        output = small(ids, targets)
        assert output.loss is not None
        (output.loss / 2).backward()
    large.zero_grad()
    output = large(tokens, labels)
    assert output.loss is not None
    output.loss.backward()
    for (left_name, left), (right_name, right) in zip(
        small.named_parameters(), large.named_parameters(), strict=True
    ):
        assert left_name == right_name
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-6)
