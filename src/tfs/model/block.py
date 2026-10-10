"""Pre-LayerNorm GPT decoder block."""

from typing import Literal

from torch import Tensor, nn

from tfs.model.attention import CausalSelfAttention, KVCache
from tfs.model.mlp import GPTMLP


class GPTBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        context: int,
        dropout: float,
        bias: bool = True,
        backend: Literal["naive", "sdpa"] = "naive",
        positional_encoding: Literal["learned", "rope"] = "learned",
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, eps=1e-5, elementwise_affine=True, bias=bias)
        self.attn = CausalSelfAttention(
            width,
            heads,
            context,
            dropout=dropout,
            bias=bias,
            backend=backend,
            positional_encoding=positional_encoding,
        )
        self.ln_2 = nn.LayerNorm(width, eps=1e-5, elementwise_affine=True, bias=bias)
        self.mlp = GPTMLP(width, dropout, bias=bias)

    def forward(
        self, x: Tensor, *, past_kv: KVCache | None = None, use_cache: bool = False
    ) -> tuple[Tensor, KVCache | None]:
        attention, cache = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attention
        x = x + self.mlp(self.ln_2(x))
        return x, cache
