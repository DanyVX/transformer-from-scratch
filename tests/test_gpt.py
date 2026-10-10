"""GPT geometry, gradients, determinism and tiny overfitting."""

import math

import pytest
import torch

from tfs.config import ModelConfig
from tfs.model.gpt import GPT
from tfs.train.optim import configure_adamw


def config(**kwargs: object) -> ModelConfig:
    params: dict[str, object] = dict(
        vocab_size=256, block_size=8, n_layer=1, n_head=2, n_embd=32, dropout=0.0
    )
    params.update(kwargs)
    return ModelConfig.model_validate(params)


def test_parameter_count_matches_formula() -> None:
    torch.manual_seed(9)
    model = GPT(config())
    assert model.count_parameters() == model.config.parameter_count()
    assert model.wte.weight is model.lm_head.weight


def test_forward_shapes_and_initial_loss() -> None:
    model = GPT(config())
    tokens = torch.randint(0, 256, (2, 8))
    out = model(tokens, targets=tokens.roll(-1, 1))
    assert out.logits.shape == (2, 8, 256)
    assert out.loss is not None
    assert abs(out.loss.item() - math.log(256)) < 1.5


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_gpt_cache_matches_full(backend: str) -> None:
    torch.manual_seed(6)
    model = GPT(config(), attention_backend=backend)
    model.eval()
    tokens = torch.randint(0, 256, (1, 6))
    full = model(tokens).logits
    cache = None
    outputs = []
    for index in range(tokens.shape[1]):
        output = model(tokens[:, index : index + 1], past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        outputs.append(output.logits)
    torch.testing.assert_close(full, torch.cat(outputs, dim=1), atol=2e-5, rtol=2e-5)


def test_gradients_and_overfit() -> None:
    torch.manual_seed(111)
    model = GPT(config(vocab_size=256, block_size=4, n_embd=16))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5], [2, 3, 4, 5]])
    initial = float(model(ids, targets).loss.item())
    for _ in range(60):
        opt.zero_grad()
        loss = model(ids, targets).loss
        assert loss is not None
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 and loss.item() < initial / 5
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_eval_determinism_and_state_dict_roundtrip() -> None:
    torch.manual_seed(11)
    cfg = config(dropout=0.15)
    model = GPT(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    expected = model(ids).logits
    assert torch.equal(expected, model(ids).logits)
    twin = GPT(cfg)
    twin.load_state_dict(model.state_dict())
    twin.eval()
    torch.testing.assert_close(expected, twin(ids).logits, atol=0, rtol=0)


def test_invalid_indices_and_context_rejected() -> None:
    model = GPT(config())
    with pytest.raises(ValueError, match="vocabulary"):
        model(torch.tensor([[256]]))
    with pytest.raises(ValueError, match="block_size"):
        model(torch.ones((1, 9), dtype=torch.long))


def test_optimizer_does_not_decay_embeddings_or_norms() -> None:
    model = GPT(config())
    opt = configure_adamw(model, 0.001)
    decayed_ids = {id(p) for p in opt.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in opt.param_groups[1]["params"]}
    assert id(model.wte.weight) in no_decay_ids
    assert id(model.h[0].ln_1.weight) in no_decay_ids
    assert id(model.h[0].attn.c_attn.weight) in decayed_ids
    assert not decayed_ids.intersection(no_decay_ids)
