# Changelog

All notable changes will follow Keep a Changelog.

## [Unreleased]

### Added
- M0 hardware-aware configuration, environment provenance, CLI, test and CI scaffold.
- Design, limitations, attribution, data and model-card documentation.

There is no release or trained model yet.

### M1 tokenizer milestone

- Implemented deterministic byte-level BPE training, heap/linked-neighbor merge-ranked encoding, opt-in special tokens, raw-byte round-trips, JSON/merges save-load and tests.
- Added synthetic-only benchmark with measured environment provenance and a table renderer.
- Reference GPT-2 token-level equivalence, large-corpus training efficiency, and model stages are still outstanding.

### Added (M2–M8)

- Packed-QKV causal attention, fused SDPA comparison, RoPE, GPT model blocks, GELU MLP and reference-weight conversion.
- Checked, document-separated token shards; reproducible gradient accumulation and atomic full-state checkpoints.
- KV-cached generation, greedy/top-k/top-p sampling, repetition penalties and streaming UTF-8.
- Synthetic, seeded model scaling and architecture-ablation experiments, plus a small CPU training demo.
- This work is a pre-release milestone. True GPT-2 124M pretrained-logit equivalence and large-corpus experiments remain unverified.
