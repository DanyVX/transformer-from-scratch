"""Ablation switches retain expected shapes and parameter-count accounting."""

import torch

from tfs.config import ModelConfig
from tfs.model.gpt import GPT


def test_swiglu_parameter_count_matches_unique_tensors() -> None:
    cfg = ModelConfig(
        vocab_size=256,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=16,
        activation="swiglu",
    )
    assert GPT(cfg).count_parameters() == cfg.parameter_count()


def test_post_layernorm_and_rope_forward() -> None:
    cfg = ModelConfig(
        vocab_size=256,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=16,
        normalization="post",
        positional_encoding="rope",
    )
    model = GPT(cfg).eval()
    result = model(torch.randint(0, 256, (2, 8)))
    assert result.logits.shape == (2, 8, 256)


def test_tied_vs_untied_parameter_counts() -> None:
    base = ModelConfig(vocab_size=256, block_size=8, n_layer=1, n_head=4, n_embd=16)
    untied = base.model_copy(update={"tie_embeddings": False})
    assert GPT(untied).count_parameters() - GPT(base).count_parameters() == 256 * 16
