"""Compare HF GPT-2 reference logits using an already-local model directory.

This script explicitly avoids automatic downloads of large model weights.
Supply a licensed local GPT-2 124M checkpoint path yourself after approval.
"""

import argparse
from pathlib import Path

import torch

from tfs.config import load_profile
from tfs.hf_compat.gpt2 import load_gpt2_weights
from tfs.model.gpt import GPT
from tfs.provenance import write_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-local-path", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/gpt2-logits.json"))
    args = parser.parse_args()
    if not args.hf_local_path.is_dir():
        parser.error("Supply an existing local directory containing GPT-2 reference weights")
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as exc:
        parser.error(f"Install the test-only 'transformers' library first: {exc}")
    config = load_profile("gpt2-124m")
    reference = GPT2LMHeadModel.from_pretrained(args.hf_local_path, local_files_only=True).eval()
    model = GPT(config.model).eval()
    load_gpt2_weights(model, reference)
    prompts = [
        torch.tensor([[15496, 11, 995, 0]]),  # GPT-2 token IDs for a short text
        torch.tensor([[40, 716, 257, 3303]]),
        torch.tensor([[50256]]),
    ]
    diffs = []
    with torch.inference_mode():
        for prompt in prompts:
            expected = reference(prompt).logits
            obtained = model(prompt).logits
            diffs.append(float((expected - obtained).abs().max().item()))
    max_error = max(diffs)
    write_result(
        args.output,
        {
            "max_absolute_logit_error": max_error,
            "per_prompt_max_errors": diffs,
            "matched_tolerance_1e-4": max_error < 1e-4,
        },
        config={
            "profile": "gpt2-124m",
            "source": "existing local HF weights",
            "prompts": [p.tolist() for p in prompts],
        },
    )
    print(f"Max absolute GPT-2 logit error: {max_error:.7f}")
    if max_error >= 1e-4:
        raise SystemExit("Reference logits outside requested tolerance")


if __name__ == "__main__":
    main()
