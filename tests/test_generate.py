"""Sampling edges and exact cached generation comparisons."""

import pytest
import torch

from tfs.config import ModelConfig
from tfs.generate.generate import generate, generate_batch
from tfs.generate.sampling import sample_next
from tfs.generate.stream import stream_text
from tfs.model.gpt import GPT
from tfs.tokenizer import ByteBPETokenizer


def _model() -> GPT:
    torch.manual_seed(71)
    return GPT(ModelConfig(vocab_size=256, block_size=8, n_layer=1, n_head=2, n_embd=16))


def test_temperature_zero_and_top_p_zero_are_greedy() -> None:
    scores = torch.tensor([[1.0, 3.0, 2.0]])
    assert sample_next(scores, temperature=0).item() == 1
    assert sample_next(scores, top_p=0).item() == 1
    assert sample_next(scores, top_k=99, top_p=1, temperature=0).item() == 1


def test_invalid_filtering_parameters() -> None:
    with pytest.raises(ValueError):
        sample_next(torch.randn(1, 5), temperature=-1)
    with pytest.raises(ValueError):
        sample_next(torch.randn(1, 5), top_k=0)
    with pytest.raises(ValueError):
        sample_next(torch.randn(1, 5), top_p=1.1)


def test_same_seed_deterministic() -> None:
    model = _model()
    tokens = [1, 2, 3]
    first = generate(model, tokens, max_new_tokens=7, top_k=10, seed=44)
    second = generate(model, tokens, max_new_tokens=7, top_k=10, seed=44)
    assert first == second and len(first) == len(tokens) + 7


def test_cache_vs_no_cache_greedy_even_after_window_overflow() -> None:
    model = _model()
    cached = generate(model, [2, 3, 4], max_new_tokens=15, temperature=0, use_cache=True)
    direct = generate(model, [2, 3, 4], max_new_tokens=15, temperature=0, use_cache=False)
    assert cached == direct


def test_empty_and_ragged_prompts() -> None:
    model = _model()
    with pytest.raises(ValueError, match="start_token"):
        generate(model, [], max_new_tokens=2)
    result = generate_batch(
        model, [[2, 3], [], [1] * 12],
        max_new_tokens=3, start_token_id=0, temperature=0
    )
    assert [len(row) for row in result] == [5, 4, 15]


def test_eos_stops_early() -> None:
    model = _model()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    result = generate(model, [4], max_new_tokens=8, eos_token_id=0, temperature=0)
    assert result == [4, 0]


def test_stream_decode_handles_split_utf8_tokens() -> None:
    tokenizer = ByteBPETokenizer(special_tokens=())
    text = "A🌍اردو"
    assert "".join(stream_text(tokenizer, list(text.encode("utf-8")))) == text
