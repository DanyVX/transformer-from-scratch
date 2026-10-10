"""Run short, controlled GPT studies on reproducible synthetic byte data.

These are smoke-scale experiments, NOT evidence of real-world language quality.
Every trial records the sample budget, seed, architecture and runner metadata.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from tfs.config import ExperimentConfig, ModelConfig, TrainConfig
from tfs.data.shards import WindowSampler
from tfs.model.gpt import GPT
from tfs.provenance import write_result
from tfs.train.trainer import Trainer, seed_all

TRAIN_BYTES = (
    b"the quick brown fox jumps over the lazy dog\n"
    b"the model predicts the next byte from the previous bytes\n"
    b"for i in range(4): print(i)\n"
) * 90
VAL_BYTES = (
    b"validation text is synthetic and has no real-world performance meaning\n"
    b"the fox and dog appear in a different sentence\n"
) * 30


def _run(
    *,
    seed: int,
    width: int,
    layers: int,
    block: int,
    steps: int,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    torch.set_num_threads(1)
    seed_all(seed)
    overrides = overrides or {}
    model_values: dict[str, object] = {
        "vocab_size": 256,
        "block_size": block,
        "n_layer": layers,
        "n_head": 4,
        "n_embd": width,
        "dropout": 0.0,
    }
    for key in (
        "positional_encoding",
        "normalization",
        "activation",
        "tie_embeddings",
        "scale_residual_init",
    ):
        if key in overrides:
            model_values[key] = overrides[key]
    model_config = ModelConfig.model_validate(model_values)
    peak = float(overrides.get("learning_rate", 0.002))
    warmup = int(overrides.get("warmup_steps", 2))
    config = ExperimentConfig(
        profile="synthetic-study",
        model=model_config,
        train=TrainConfig(
            device="cpu",
            batch_size=2,
            gradient_accumulation=1,
            max_steps=steps,
            learning_rate=peak,
            min_learning_rate=peak / 10,
            warmup_steps=min(warmup, steps),
            seed=seed,
        ),
    )
    trainer = Trainer(
        GPT(model_config),
        WindowSampler(np.frombuffer(TRAIN_BYTES, dtype=np.uint8), block, 2, seed=seed),
        config,
    )
    trainer.train_steps(steps)
    trainer.model.eval()
    with torch.inference_mode():
        data = np.frombuffer(VAL_BYTES, dtype=np.uint8).astype(np.int64)
        # Fixed, non-overlapping validation windows.
        positions = range(0, len(data) - block, block)
        per_window_loss = []
        for start in positions:
            ids = torch.from_numpy(data[start : start + block].copy())[None, :]
            targets = torch.from_numpy(data[start + 1 : start + block + 1].copy())[None, :]
            output = trainer.model(ids, targets)
            assert output.loss is not None
            per_window_loss.append(float(output.loss.item()))
    loss = statistics.mean(per_window_loss)
    parameter_count = trainer.model.count_parameters()
    total_tokens = steps * config.train.batch_size * block
    return {
        "seed": seed,
        "params": parameter_count,
        "train_tokens": total_tokens,
        "val_loss_nats_per_byte": loss,
        "bits_per_byte": loss / math.log(2),
        "perplexity_per_byte": math.exp(loss),
        "estimated_train_flops_6NT": 6 * parameter_count * total_tokens,
        "config": config.model_dump(mode="json"),
        "validation_windows": len(per_window_loss),
    }


def variants(study: str) -> dict[str, dict[str, object]]:
    if study == "scaling":
        return {
            "width16_layers1": {"width": 16, "layers": 1},
            "width24_layers2": {"width": 24, "layers": 2},
            "width32_layers3": {"width": 32, "layers": 3},
        }
    if study == "ablations":
        return {
            "learned_positions": {"positional_encoding": "learned"},
            "rope_positions": {"positional_encoding": "rope"},
            "pre_ln": {"normalization": "pre"},
            "post_ln": {"normalization": "post"},
            "gelu": {"activation": "gelu"},
            "swiglu": {"activation": "swiglu"},
            "tied": {"tie_embeddings": True},
            "untied": {"tie_embeddings": False},
            "scaled_init": {"scale_residual_init": True},
            "unscaled_init": {"scale_residual_init": False},
            "lr_low": {"learning_rate": 0.0005},
            "lr_high": {"learning_rate": 0.02},
            "warmup": {"warmup_steps": 2},
            "no_warmup": {"warmup_steps": 0},
            "short_context": {"block": 8},
            "long_context": {"block": 16},
        }
    raise ValueError("Unknown study name")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["scaling", "ablations"], default="scaling")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/synthetic-study.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    trials: list[dict[str, object]] = []
    for label, options in variants(args.study).items():
        for seed in (11, 29):
            opts = dict(options)
            width = int(opts.pop("width", 16))
            layers = int(opts.pop("layers", 1))
            block = int(opts.pop("block", 8))
            row = _run(
                seed=seed, width=width, layers=layers, block=block, steps=args.steps, overrides=opts
            )
            row["variant"] = label
            trials.append(row)
    write_result(
        args.output,
        {"study": args.study, "trials": trials, "seeds": [11, 29]},
        config={
            "synthetic_only": True,
            "steps": args.steps,
            "train_bytes": len(TRAIN_BYTES),
            "validation_bytes": len(VAL_BYTES),
        },
    )
    print(f"Saved {len(trials)} measured trials: {args.output}")
    if args.print_json:
        print(
            "SCALING_STUDY_JSON="
            + json.dumps(json.loads(args.output.read_text(encoding="utf-8")))
        )


if __name__ == "__main__":
    main()
