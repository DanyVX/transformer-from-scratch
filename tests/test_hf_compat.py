"""Offline small-reference comparison, no weight downloads."""

import pytest
import torch

from tfs.config import ModelConfig
from tfs.hf_compat.gpt2 import load_gpt2_weights
from tfs.model.gpt import GPT


@pytest.mark.slow
def test_small_random_hf_gpt2_equivalence() -> None:
    transformers = pytest.importorskip("transformers")
    hf_cfg = transformers.GPT2Config(
        vocab_size=256,
        n_positions=16,
        n_ctx=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    reference = transformers.GPT2LMHeadModel(hf_cfg).eval()
    model = GPT(
        ModelConfig(vocab_size=256, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    ).eval()
    load_gpt2_weights(model, reference)
    tokens = torch.tensor([[1, 50, 2, 0, 53]])
    with torch.no_grad():
        expected = reference(tokens).logits
        actual = model(tokens).logits
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
