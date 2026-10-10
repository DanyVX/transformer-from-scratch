# Results

All figures below originate from executable scripts. The raw measurement file is [tokenizer-smoke.json](../results/tokenizer-smoke.json). Regenerate this table with:

    python results/render.py results/tokenizer-smoke.json

## M1: synthetic tokenizer smoke (measured)

| Synthetic corpus | Tokens/sec | Tokens/character | Bytes/token |
| --- | ---: | ---: | ---: |
| code | 353,674.8 | 0.381 | 2.625 |
| english | 329,857.3 | 0.306 | 3.267 |
| urdu | 189,220.2 | 0.304 | 5.929 |

Measured source commit: 80f22860301f8651c99a396d65d30f0c9996e9be

Runner: Linux / Python 3.12.15 / CPU count 4; CUDA=False.

Synthetic corpus and a single short timing run. Do not extrapolate to large datasets.

The three corpora are repeated literals in bench/tokenizer.py. The benchmark used a tokenizer fitted on all three synthetic samples, a requested/effective vocabulary size of 360, and four encoding iterations per sample. These are **tiny in-sample throughput measurements**, not held-out compression or statistical evidence. Urdu is UTF-8 multibyte text, so bytes/token and tokens/character answer different questions.

## Future milestones

| Metric | Result |
| --- | --- |
| GPT-2 (124M) reference-logit matching | TBD (not yet measured) |
| Causal-mask leakage test | TBD (not yet measured) |
| KV-cache generation speedup | TBD (not yet measured) |
| Tiny GPT validation loss and bits-per-character | TBD (not yet measured) |
| Scaling, ablation seed variance, out-of-sample tokenizer comparison | TBD (not yet measured) |

M0 parameter counts are structural predictions, not benchmarks. Report uncertainty and negative outcomes for all future data-dependent claims.
