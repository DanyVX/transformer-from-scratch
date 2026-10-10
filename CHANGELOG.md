# Changelog

All notable changes follow Keep a Changelog.

## [0.1.0] - 2026-10-10

### Implemented and tested

- M0: hardware-aware configuration, environment provenance, package, CLI and CI.
- M1: deterministic byte-level BPE with Unicode/raw byte tests and synthetic benchmark.
- M2: explicit causal attention, PyTorch SDPA comparison, RoPE, future-token isolation.
- M3: GPT-2-shaped decoder, tied embeddings, optional HuggingFace weight mapping.
- M4: document-split, hashed memmapped token shards and reproducible samplers.
- M5: AdamW, warmup/cosine schedule, accumulation, gradient clipping, RNG/exact-resume checkpointing.
- M6: KV-cached sampling, context rollover, EOS and incremental UTF-8 decoding.
- M7: seeded CPU-only scaling and architecture ablation study harness.
- M8: synthetic one-step training/generation demo, evidence-backed benchmark JSON, documentation.
- Release hardening: wheel-bundled YAML profiles, optional offline randomly initialized HF-reference test, documentation cleanup.

### Known gaps

Released GPT-2 124M pretrained reference-logit parity, third-party real-corpus quality, and GPU training/benchmark coverage are **not verified**. This release contains source code and synthetic smoke experiments, not pretrained model weights.
