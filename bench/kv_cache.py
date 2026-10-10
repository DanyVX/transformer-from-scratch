"""Measured CPU KV-cache incremental speed comparison with saved provenance."""

import argparse
import json
import time
from pathlib import Path

import torch

from tfs.config import ModelConfig
from tfs.model.gpt import GPT
from tfs.provenance import write_result


def run(context: int, repeats: int) -> dict[str, float | int]:
    torch.set_num_threads(1)
    torch.manual_seed(9)
    cfg = ModelConfig(vocab_size=256, block_size=context + 12, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg).eval()
    tokens = torch.randint(0, 256, (1, context + 8))
    times: dict[str, float] = {}
    with torch.inference_mode():
        for cache_on in (False, True):
            start = time.perf_counter()
            for _ in range(repeats):
                past = None
                for index in range(context, context + 8):
                    if cache_on:
                        ids = (
                            tokens[:, :context]
                            if index == context
                            else tokens[:, index - 1 : index]
                        )
                        result = model(ids, past_key_values=past, use_cache=True)
                        past = result.past_key_values
                    else:
                        model(tokens[:, :index])
            times["cached" if cache_on else "full"] = (time.perf_counter() - start) / repeats
    return {
        "context": context,
        "full_seconds": times["full"],
        "cached_seconds": times["cached"],
        "speedup": times["full"] / times["cached"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/kv-cache.json"))
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    metrics = [run(context, repeats=2) for context in (8, 16, 32)]
    write_result(args.output, {"timings": metrics}, config={"seed": 9, "repeats": 2})
    if args.print_json:
        print("KV_CACHE_JSON=" + json.dumps(json.loads(args.output.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
