"""Correct incremental autoregression with bounded context and stop tokens."""

from collections.abc import Sequence

import torch

from tfs.generate.sampling import sample_next
from tfs.model.gpt import GPT


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: Sequence[int],
    *,
    max_new_tokens: int,
    start_token_id: int | None = None,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    use_cache: bool = True,
) -> list[int]:
    """Generate including the original prompt, left-truncating only model context.

    After exceeding context window, invalidate cache and recompute with the new
    left-truncated window; learned absolute positions must restart at zero.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if not prompt and start_token_id is None:
        raise ValueError("Empty prompt requires start_token_id")
    sequence = list(prompt) if prompt else [start_token_id]  # type: ignore[list-item]
    device = next(model.parameters()).device
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        window = sequence[-model.config.block_size:]
        prediction = model(
            torch.tensor([window], device=device, dtype=torch.long), use_cache=use_cache
        )
        cache = prediction.past_key_values
        for step in range(max_new_tokens):
            next_token = sample_next(
                prediction.logits[:, -1, :],
                temperature=temperature, top_k=top_k, top_p=top_p,
                previous=torch.tensor([sequence], device=device, dtype=torch.long),
                repetition_penalty=repetition_penalty,
                generator=rng if seed is not None else None,
            )
            token_id = int(next_token.item())
            sequence.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            if step + 1 == max_new_tokens:
                break
            if use_cache and cache is not None and cache[0][0].shape[-2] < model.config.block_size:
                prediction = model(
                    next_token.view(1, 1), past_key_values=cache, use_cache=True
                )
                cache = prediction.past_key_values
            else:
                window = sequence[-model.config.block_size:]
                prediction = model(
                    torch.tensor([window], dtype=torch.long, device=device), use_cache=use_cache
                )
                cache = prediction.past_key_values
        return sequence
    finally:
        model.train(was_training)


def generate_batch(
    model: GPT, prompts: Sequence[Sequence[int]], **kwargs: object
) -> list[list[int]]:
    """Ragged batches decoded independently, so EOS/padding never mix positions."""
    return [generate(model, prompt, **kwargs) for prompt in prompts]  # type: ignore[arg-type]
