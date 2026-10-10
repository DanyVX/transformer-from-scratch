"""Separate AdamW decay groups by parameter role; keep tied weights unique."""

import torch

from tfs.model.gpt import GPT


def configure_adamw(
    model: GPT, learning_rate: float, weight_decay: float = 0.1
) -> torch.optim.AdamW:
    """Decay non-embedding matrix weights, never biases, norms, or embeddings."""
    decay_ids = {
        id(param)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
        for param_name, param in module.named_parameters(recurse=False)
        if param_name == "weight"
    }
    unique = {id(p): p for p in model.parameters() if p.requires_grad}
    decay = [p for key, p in unique.items() if key in decay_ids]
    no_decay = [p for key, p in unique.items() if key not in decay_ids]
    if len(decay) + len(no_decay) != len(unique):
        raise AssertionError("Invalid optimizer parameter grouping")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
