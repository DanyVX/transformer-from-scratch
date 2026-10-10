"""Decoder-only GPT architecture, weight-tied token predictions, shifted-label loss."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tfs.config import ModelConfig
from tfs.model.attention import KVCache
from tfs.model.block import GPTBlock


@dataclass
class GPTOutput:
    logits: Tensor
    loss: Tensor | None
    past_key_values: tuple[KVCache, ...] | None


class GPT(nn.Module):
    """GPT-2 geometry with configurable dimensions and attention backend.

    Inputs and targets are both (B,T). Targets are *already shifted* by data
    sampling: targets[b,t] = raw_sequence[b,t+1]. Do not shift twice.
    """

    def __init__(
        self, config: ModelConfig, *, attention_backend: Literal["naive", "sdpa"] = "naive"
    ) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe: nn.Embedding | None = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.positional_encoding == "learned"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(
            GPTBlock(
                config.n_embd,
                config.n_head,
                config.block_size,
                config.dropout,
                config.bias,
                attention_backend,
                config.positional_encoding,
                config.normalization,
                config.activation,
            )
            for _ in range(config.n_layer)
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for block in self.h:
            if not config.scale_residual_init:
                continue
            torch.nn.init.normal_(
                block.attn.c_proj.weight,
                mean=0.0,
                std=0.02 / (2 * config.n_layer) ** 0.5,
            )
            torch.nn.init.normal_(
                block.mlp.c_proj.weight,
                mean=0.0,
                std=0.02 / (2 * config.n_layer) ** 0.5,
            )
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        past_key_values: tuple[KVCache, ...] | None = None,
        use_cache: bool = False,
    ) -> GPTOutput:
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-two torch.long tensor")
        if input_ids.numel() == 0:
            raise ValueError("Empty prompt; provide a BOS/EOS token")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("Token ID outside configured vocabulary")
        batch, seq_len = input_ids.shape
        if past_key_values is not None and len(past_key_values) != len(self.h):
            raise ValueError("KV cache layer count mismatch")
        offset = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if seq_len + offset > self.config.block_size:
            raise ValueError("Input exceeds block_size")

        x = self.wte(input_ids)
        if self.wpe is not None:
            positions = torch.arange(offset, offset + seq_len, device=input_ids.device)
            x = x + self.wpe(positions)[None, :, :]
        x = self.drop(x)
        new_cache: list[KVCache] = []
        for index, block in enumerate(self.h):
            past = past_key_values[index] if past_key_values is not None else None
            x, cache = block(x, past_kv=past, use_cache=use_cache)
            if cache is not None:
                new_cache.append(cache)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape or targets.dtype != torch.long:
                raise ValueError("targets must be long tensors matching input IDs")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return GPTOutput(logits, loss, tuple(new_cache) if use_cache else None)

    def count_parameters(self) -> int:
        """Count unique tensors, avoiding tied-weight double counting."""
        return sum(param.numel() for param in self.parameters())
