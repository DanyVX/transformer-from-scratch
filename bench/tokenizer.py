"""Reproducible, synthetic-only BPE encoding benchmark.

Example: python bench/tokenizer.py --output results/tokenizer-smoke.json
No benchmark number is hard-coded; results include automatically captured provenance.
"""

import argparse
import json
import time
from pathlib import Path

from tfs.provenance import write_result
from tfs.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/tokenizer-smoke.json"))
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    samples = {
        "english": "Hello world. This is a synthetic benchmark text.\n" * 120,
        "urdu": "السلام علیکم یہ اردو متن کا تجرباتی نمونہ ہے۔\n" * 120,
        "code": "for i in range(10):\n    print('value', i)\n" * 120,
    }
    vocab_size = 360
    tokenizer = ByteBPETokenizer.train(samples.values(), vocab_size=vocab_size)
    metrics: dict[str, dict[str, float | int]] = {}
    for label, sample in samples.items():
        iterations = 4
        start = time.perf_counter()
        tokens = []
        for _ in range(iterations):
            tokens = tokenizer.encode(sample)
        elapsed = time.perf_counter() - start
        token_count = len(tokens)
        metrics[label] = {
            "tokens": token_count,
            "bytes": len(sample.encode("utf-8")),
            "characters": len(sample),
            "tokens_per_second": iterations * token_count / elapsed,
            "bytes_per_token": len(sample.encode("utf-8")) / token_count,
            "tokens_per_character": token_count / len(sample),
            "seconds": elapsed,
        }

    write_result(
        args.output,
        {"effective_vocab_size": tokenizer.vocab_size, "corpora": metrics},
        config={
            "profile": "synthetic-tokenizer-smoke",
            "vocab_size_requested": vocab_size,
            "iterations": iterations,
            "seed": 0,
            "data": "synthetic strings in script",
        },
    )
    if args.print_json:
        print(
            "TOKENIZER_BENCHMARK_JSON="
            + json.dumps(json.loads(args.output.read_text(encoding="utf-8")), ensure_ascii=False)
        )
    else:
        print(f"Wrote measured results to {args.output}")


if __name__ == "__main__":
    main()
