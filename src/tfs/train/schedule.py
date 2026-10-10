"""Linear warmup followed by a cosine decay schedule."""

import math


def learning_rate(
    step: int, total_steps: int, warmup_steps: int, peak: float, minimum: float
) -> float:
    """Learning rate at zero-indexed optimizer step, inclusive of final step."""
    if total_steps < 1 or not 0 <= warmup_steps <= total_steps:
        raise ValueError("Invalid total or warmup steps")
    if peak <= 0 or not 0 <= minimum <= peak or step < 0:
        raise ValueError("Invalid rate or step")
    if step >= total_steps:
        return minimum
    if warmup_steps and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps - 1)
    ratio = min(1.0, (step - warmup_steps) / decay_steps)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * ratio))
