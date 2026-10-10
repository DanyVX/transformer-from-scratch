"""Map GPT-2 HuggingFace weights to the original PyTorch model.

Accepts a supplied pretrained or randomly initialized HF model. Importing
transformers and downloading weights is the caller's explicit responsibility.
"""

from typing import Any

import torch

from tfs.model.gpt import GPT


def load_gpt2_weights(target: GPT, source: Any) -> None:
    """Copy official GPT-2 architecture tensors and transpose HF Conv1D matrices."""
    c = target.config
    if (
        c.positional_encoding != "learned" or not c.tie_embeddings or not c.bias
        or target.wpe is None
    ):
        raise ValueError("HF GPT-2 mapping needs learned positions, bias and tied embeddings")
    hf_config = source.config
    if (
        c.vocab_size != hf_config.vocab_size or c.block_size != hf_config.n_positions
        or c.n_embd != hf_config.n_embd or c.n_head != hf_config.n_head
        or c.n_layer != hf_config.n_layer
    ):
        raise ValueError("HF reference dimensions differ from target configuration")

    source_state = source.transformer.state_dict()
    with torch.no_grad():
        target.wte.weight.copy_(source_state["wte.weight"])
        target.wpe.weight.copy_(source_state["wpe.weight"])
        target.ln_f.weight.copy_(source_state["ln_f.weight"])
        target.ln_f.bias.copy_(source_state["ln_f.bias"])
        for index, block in enumerate(target.h):
            prefix = f"h.{index}."
            for name in ("ln_1", "ln_2"):
                module = getattr(block, name)
                module.weight.copy_(source_state[f"{prefix}{name}.weight"])
                module.bias.copy_(source_state[f"{prefix}{name}.bias"])
            for key, linear in (
                ("attn.c_attn", block.attn.c_attn),
                ("attn.c_proj", block.attn.c_proj),
                ("mlp.c_fc", block.mlp.c_fc),
                ("mlp.c_proj", block.mlp.c_proj),
            ):
                linear.weight.copy_(source_state[f"{prefix}{key}.weight"].T.contiguous())
                linear.bias.copy_(source_state[f"{prefix}{key}.bias"])
    target.eval()
