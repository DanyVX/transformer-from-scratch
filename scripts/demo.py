"""Train a small GPT on synthetic text, then print a sampled continuation.

This is a reproducible pipeline smoke demo, not a useful pretrained LM.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from tfs.config import load_profile
from tfs.data.shards import WindowSampler, load_shard, save_token_shards, split_documents
from tfs.generate.generate import generate
from tfs.model.gpt import GPT
from tfs.provenance import write_result
from tfs.tokenizer import ByteBPETokenizer
from tfs.train.trainer import Trainer, seed_all

DOCUMENTS = [
    ("cloud code and careful testing make software easier to trust. " * 12) + "\n",
    ("a transformer predicts the next token using a causal attention mask. " * 12) + "\n",
    ("byte pair encoding merges common adjacent byte sequences by rank. " * 12) + "\n",
    ("training data should be separate from the validation documents. " * 12) + "\n",
    ("checkpoint recovery should reproduce the optimizer and random state. " * 12) + "\n",
    ("the model learns only from the synthetic sentences it is shown. " * 12) + "\n",
    ("the weather today is just a made up example for this tiny text model. " * 12) + "\n",
    ("an engineer measures failures instead of claiming untested success. " * 12) + "\n",
    ("small CPU experiments are useful for checking tensor dimensions. " * 12) + "\n",
    ("validation tells us whether the model can generalize a little. " * 12) + "\n",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["tiny", "small", "gpt2-124m"], default="tiny")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/demo-run.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    config = load_profile(args.profile)
    if args.profile != "tiny":
        parser.error("CPU demo uses tiny. Use larger profiles only on suitable GPU hardware.")

    seed_all(config.train.seed)
    train_docs, val_docs = split_documents(DOCUMENTS, validation_fraction=0.2, seed=42)
    tokenizer = ByteBPETokenizer.train(train_docs, vocab_size=config.model.vocab_size)
    if tokenizer.vocab_size > config.model.vocab_size:
        raise AssertionError("Tokenizer exceeds GPT vocabulary")
    shards_dir = Path("data/processed/demo")
    save_token_shards(
        shards_dir,
        {"train": train_docs, "val": val_docs},
        tokenizer.encode,
        config.model.vocab_size,
        tokenizer.special_ids["<|endoftext|>"],
    )
    training = load_shard(shards_dir, "train")
    validation = load_shard(shards_dir, "val")
    adjusted_train = config.train.model_copy(update={"max_steps": args.steps})
    if adjusted_train.warmup_steps > args.steps:
        adjusted_train = adjusted_train.model_copy(update={"warmup_steps": args.steps})
    adjusted = config.model_copy(update={"train": adjusted_train})
    model = GPT(adjusted.model)
    sampler = WindowSampler(
        training, adjusted.model.block_size, adjusted.train.batch_size, seed=adjusted.train.seed
    )
    trainer = Trainer(
        model, sampler, adjusted, device="cpu", log_path=Path("results/demo-train.csv")
    )
    losses = trainer.train_steps(args.steps)
    trainer.model.eval()
    val_sampler = WindowSampler(validation, adjusted.model.block_size, 2, seed=123, sequential=True)
    validation_losses = []
    with torch.inference_mode():
        for _ in range(4):
            x, y = val_sampler.next_batch()
            result = trainer.model(torch.from_numpy(x), torch.from_numpy(y))
            assert result.loss is not None
            validation_losses.append(result.loss.item())

    prompt = "cloud"
    sampled = generate(
        trainer.model,
        tokenizer.encode(prompt),
        max_new_tokens=28,
        temperature=0.8,
        top_k=20,
        seed=42,
    )
    print("Prompt:", prompt)
    print("Synthetic sample:", tokenizer.decode(sampled))
    mean_val_loss = sum(validation_losses) / len(validation_losses)
    write_result(
        args.output,
        {
            "train_final_loss": losses[-1],
            "validation_loss": mean_val_loss,
            "validation_bits_per_token": mean_val_loss / math.log(2),
            "steps": trainer.step,
            "parameter_count": trainer.model.count_parameters(),
            "generated_ids": sampled,
        },
        config=adjusted.model_dump(mode="json"),
    )
    print("Saved measured metrics to", args.output)
    if args.print_json:
        print("MODEL_DEMO_JSON=" + json.dumps(json.loads(args.output.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
