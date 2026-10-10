"""GPT-2's feed-forward expansion with approximate GELU."""

from typing import cast

from torch import Tensor, nn
from torch.nn import functional as F


class GPTMLP(nn.Module):
    def __init__(self, width: int, dropout: float, bias: bool = True) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, 4 * width, bias=bias)
        self.c_proj = nn.Linear(4 * width, width, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))))
