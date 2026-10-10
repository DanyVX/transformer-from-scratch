"""Causal multi-head attention with explicit and PyTorch fused backends."""

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tfs.model.rope import apply_rope

KVCache = tuple[Tensor, Tensor]


class CausalSelfAttention(nn.Module):
    """GPT-2-compatible packed QKV attention.

    Input: (B, T, C). Q,K,V: (B, H, T, D). Output: (B, T, C).
    The mask uses absolute token positions when a KV cache is supplied.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        backend: Literal["naive", "sdpa"] = "naive",
        positional_encoding: Literal["learned", "rope"] = "learned",
    ) -> None:
        super().__init__()
        if n_embd < 1 or n_head < 1 or n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if block_size < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid block_size or dropout")
        if backend not in ("naive", "sdpa"):
            raise ValueError("Unknown attention backend")
        if positional_encoding == "rope" and (n_embd // n_head) % 2:
            raise ValueError("RoPE needs an even head dimension")
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout
        self.backend = backend
        self.positional_encoding = positional_encoding
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, *, past_kv: KVCache | None = None, use_cache: bool = False
    ) -> tuple[Tensor, KVCache | None]:
        if x.ndim != 3 or x.shape[-1] != self.n_embd:
            raise ValueError("Expected input shape (batch, length, embedding)")
        batch, length, channels = x.shape
        offset = 0 if past_kv is None else past_kv[0].shape[-2]
        if length == 0 or length + offset > self.block_size:
            raise ValueError("Attention sequence exceeds block_size or is empty")

        q, k, v = self.c_attn(x).split(channels, dim=-1)
        q = q.view(batch, length, self.n_head, -1).transpose(1, 2)
        k = k.view(batch, length, self.n_head, -1).transpose(1, 2)
        v = v.view(batch, length, self.n_head, -1).transpose(1, 2)
        if self.positional_encoding == "rope":
            q, k = apply_rope(q, k, offset)

        if past_kv is not None:
            old_k, old_v = past_kv
            if (
                old_k.shape[:2] != (batch, self.n_head)
                or old_v.shape != old_k.shape
                or old_k.shape[-1] != self.n_embd // self.n_head
                or old_k.device != k.device
            ):
                raise ValueError("Invalid or incompatible KV cache")
            k = torch.cat((old_k, k), dim=-2)
            v = torch.cat((old_v, v), dim=-2)

        total = k.shape[-2]
        rows = torch.arange(offset, offset + length, device=x.device)[:, None]
        columns = torch.arange(total, device=x.device)[None, :]
        allowed = columns <= rows  # (T_query, T_key)

        if self.backend == "naive":
            weights = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(q.shape[-1]))
            weights = weights.masked_fill(~allowed, torch.finfo(weights.dtype).min)
            weights = F.softmax(weights.float(), dim=-1).to(q.dtype)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            attended = weights @ v
        else:
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        result = self.resid_dropout(self.c_proj(attended))
        return result, (k, v) if use_cache else None
