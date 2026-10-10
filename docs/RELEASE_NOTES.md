# v0.1.0 | Transformer From Scratch

This release packages an original PyTorch decoder-only GPT implementation and its reproducible **CPU-first research and testing pipeline**. No pretrained model weights are included.

## Shipped

- Byte-level deterministic BPE tokenizer, Unicode/raw bytes, opt-in special token handling.
- Causal multihead attention (explicit and SDPA comparison), RoPE, GPT decoder blocks with tied embeddings.
- Offline input-shape-safe Hugging Face GPT-2 weight-import utility. Tiny randomly initialized HF parity is tested separately in CI, not a 124M pretrained claim.
- Document-separated and SHA-256 validated token shards; deterministic next-token sampling.
- AdamW trainer, step schedule, gradient accumulation/clipping, trusted-local checkpoint/resume with RNG and sampler state.
- KV-cached generation, greedy/temperature/top-k/top-p sampling, stop tokens and streaming UTF-8.
- Measured CPU-only synthetic results: tiny one-step training demonstration, two-seed two-step scaling smoke, and small-context cached decoding benchmarks, with committed JSON provenance.
- Python 3.11/3.12 CI: lint, type checking, unit/property tests, secret scan and dependency audit; wheel-profile install validation and an offline small HF reference comparison.

## Start on Linux, macOS or Windows

Install Python 3.11 or 3.12, create a virtual environment, then:

```bash
python -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,model]'
python scripts/tasks.py test-fast
python scripts/demo.py --profile tiny --steps 8
```

See the [README](../README.md), [results](RESULTS.md), [limitations](LIMITATIONS.md), [data declaration](../DATA.md), and [model card](../MODEL_CARD.md).

## Verification limits

The released pretrained GPT-2 124M logits, true TinyShakespeare/TinyStories trained accuracy, safe useful language generation, CUDA throughput, and GPU/AMP/distributed training are **not verified**. Benchmark numbers are tiny synthetic smoke tests with limited repetitions; do not interpret them as real-world language quality or scaling laws. The local-only parity helper never downloads large artifacts. Checkpoint pickle deserialization requires trusted-local files.

This is a versioned source-code research implementation, **not a shipped pretrained foundation model**.
