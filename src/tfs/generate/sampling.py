"""Greedy, temperature, top-k, top-p, and optional repetition penalties."""

import torch
from torch import Tensor


def sample_next(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    previous: Tensor | None = None,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return one token per batch row, never filtering out all probability mass."""
    if logits.ndim != 2:
        raise ValueError("logits must be [batch, vocabulary]")
    if temperature < 0 or top_k is not None and top_k < 1:
        raise ValueError("Invalid temperature or top_k")
    if not 0 <= top_p <= 1 or repetition_penalty < 1:
        raise ValueError("Invalid top_p or repetition_penalty")
    scores = logits.clone()
    if previous is not None and repetition_penalty != 1:
        if previous.ndim != 2 or previous.size(0) != scores.size(0):
            raise ValueError("previous must have matching batch size")
        for row in range(scores.size(0)):
            used = torch.unique(previous[row])
            values = scores[row, used]
            scores[row, used] = torch.where(
                values < 0, values * repetition_penalty, values / repetition_penalty
            )
    if temperature == 0:
        return scores.argmax(dim=-1)
    scores = scores / temperature
    if top_k is not None:
        selected = min(top_k, scores.size(-1))
        threshold = torch.topk(scores, selected, dim=-1).values[:, -1, None]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if top_p < 1:
        sorted_scores, order = torch.sort(scores, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_scores.float(), dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative - probabilities >= top_p
        remove[:, 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.empty_like(scores).scatter(-1, order, sorted_scores)
    probabilities = torch.softmax(scores.float(), dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Invalid/nonfinite logits after filtering")
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)
