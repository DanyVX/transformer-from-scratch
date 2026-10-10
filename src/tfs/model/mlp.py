"""GPT-2's feed-forward expansion with approximate GELU."""

from typing import Literal, cast

from torch import Tensor, nn
from torch.nn import functional as F


class GPTMLP(nn.Module):
    def __init__(
        self,
        width: int,
        dropout: float,
        bias: bool = True,
        activation: Literal["gelu", "swiglu"] = "gelu",
    ) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, 4 * width, bias=bias)
        self.gate = nn.Linear(width, 4 * width, bias=bias) if activation == "swiglu" else None
        self.activation = activation
        self.c_proj = nn.Linear(4 * width, width, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.c_fc(x)
        if self.gate is not None:
            hidden = F.silu(self.gate(x)) * hidden
        else:
            hidden = F.gelu(hidden, approximate="tanh")
        return cast(Tensor, self.dropout(self.c_proj(hidden)))
