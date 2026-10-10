"""Interleaved pair-wise rotary position encoding."""

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """Rotate each adjacent even/odd pair: (x, y) -> (-y, x)."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE head width must be even")
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(
    q: Tensor, k: Tensor, offset: int = 0, base: float = 10000.0
) -> tuple[Tensor, Tensor]:
    """Rotate (B, H, T, D) queries/keys using absolute token positions."""
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] % 2:
        raise ValueError("RoPE requires matching 4-D tensors with even head dimensions")
    if offset < 0:
        raise ValueError("RoPE offset cannot be negative")
    width = q.shape[-1]
    positions = torch.arange(offset, offset + q.shape[-2], device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, width, 2, device=q.device).float() / width))
    phases = positions[:, None] * inv_freq[None, :]
    cos = torch.repeat_interleave(phases.cos(), 2, dim=-1).to(q.dtype)[None, None, :, :]
    sin = torch.repeat_interleave(phases.sin(), 2, dim=-1).to(q.dtype)[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
