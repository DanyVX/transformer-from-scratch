"""CPU-only correctness and causal isolation tests."""

import pytest
import torch

from tfs.model.attention import CausalSelfAttention
from tfs.model.rope import apply_rope, rotate_half


def _module(backend: str = "naive", dropout: float = 0.0) -> CausalSelfAttention:
    return CausalSelfAttention(32, 4, 12, backend=backend, dropout=dropout)


@pytest.mark.parametrize("length", [1, 3, 12])
def test_shape_and_mask(length: int) -> None:
    attn = _module()
    y, cache = attn(torch.randn(1, length, 32), use_cache=True)
    assert y.shape == (1, length, 32)
    assert cache is not None and cache[0].shape == (1, 4, length, 8)


def test_naive_matches_sdpa() -> None:
    torch.manual_seed(17)
    naive = _module()
    fused = _module("sdpa")
    fused.load_state_dict(naive.state_dict())
    naive.eval()
    fused.eval()
    x = torch.randn(2, 11, 32)
    y1, _ = naive(x)
    y2, _ = fused(x)
    torch.testing.assert_close(y1, y2, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_future_tokens_cannot_change_past(backend: str) -> None:
    torch.manual_seed(4)
    attn = _module(backend)
    attn.eval()
    x = torch.randn(2, 8, 32)
    changed = x.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 5
    original, _ = attn(x)
    modified, _ = attn(changed)
    torch.testing.assert_close(original[:, :5], modified[:, :5], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_cached_attention_matches_full(backend: str) -> None:
    torch.manual_seed(7)
    module = _module(backend)
    module.eval()
    x = torch.randn(1, 8, 32)
    full, _ = module(x)
    past = None
    pieces = []
    for i in range(x.shape[1]):
        piece, past = module(x[:, i : i + 1], past_kv=past, use_cache=True)
        pieces.append(piece)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=2e-6, rtol=2e-6)


def test_dropout_eval_deterministic_and_train_stochastic() -> None:
    torch.manual_seed(5)
    module = _module(dropout=0.4)
    x = torch.randn(2, 9, 32)
    module.eval()
    assert torch.equal(module(x)[0], module(x)[0])
    module.train()
    assert not torch.equal(module(x)[0], module(x)[0])


def test_attention_rejects_invalid_geometry_and_context() -> None:
    with pytest.raises(ValueError):
        CausalSelfAttention(31, 4, 8)
    with pytest.raises(ValueError):
        CausalSelfAttention(30, 6, 8, positional_encoding="rope")
    module = _module()
    with pytest.raises(ValueError, match="block_size"):
        module(torch.randn(1, 13, 32))


def test_rope_norm_and_relative_offset() -> None:
    torch.manual_seed(8)
    q = torch.randn(1, 2, 1, 8)
    k = torch.randn(1, 2, 1, 8)
    a, _ = apply_rope(q, k, offset=2)
    b, _ = apply_rope(q, k, offset=7)
    torch.testing.assert_close(a.norm(dim=-1), b.norm(dim=-1), atol=1e-6, rtol=1e-6)
    assert torch.equal(rotate_half(torch.tensor([[[[1., 2., 3., 4.]]]])),
                       torch.tensor([[[[-2., 1., -4., 3.]]]]))


def test_rope_shift_invariant_dot_product() -> None:
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    q1, _ = apply_rope(q, q, 2)
    _, k1 = apply_rope(k, k, 5)
    q2, _ = apply_rope(q, q, 12)
    _, k2 = apply_rope(k, k, 15)
    torch.testing.assert_close((q1 * k1).sum(), (q2 * k2).sum(), atol=1e-5, rtol=1e-5)
