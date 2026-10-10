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

## Milestone verification

| Check | Result |
| --- | --- |
| Causal-mask and future-token isolation | Automated CPU tests passed |
| Naive vs SDPA forward matching, KV-cache equivalence | Automated CPU tests passed |
| Deterministic resume and checkpoint recovery | Automated CPU tests passed |
| Tiny trained model validation loss | Measured below, synthetic one-step run |
| Scaling and KV-cache speed | Measured below, short synthetic benchmarks |
| Small randomized HF GPT-2 reference-logit match | **Passed**, max absolute difference 5.96046448×10⁻⁸ |
| Released GPT-2 124M pretrained-logit match | **Not tested**, requires large externally supplied weights |
| Real-data generalization and safety | **Not measured** |

All stored measurements correspond to their listed Git commit and runner. The repository's tests are updated over time, so a historical raw JSON result is not automatically a result from the latest source.

## M8 CPU model pipeline smoke (measured)

A **single optimizer step** on synthetic, hand-authored sentences; this verifies the data-to-training-to-generation wiring only. It does **not** demonstrate language quality, convergence, or dataset generalization. The validation sample contains only a few tiny documents and is not statistically significant.

- Training step count: 1 (measured)
- Unique model parameters: 1,271,808
- Last training loss: 6.293098 nats/token
- Validation loss: 6.221196 nats/token
- Validation bits/token: 8.975288
- Runner: Linux, Python 3.12.15, CPU count 4, CUDA=false
- Source commit: `c067d398b3e116897607a7202fc055fdb84ff4b8`
- Raw provenance: [results/demo-ci.json](../results/demo-ci.json). Regenerate via `python scripts/demo.py --profile tiny --steps 1 --output results/demo-ci.json --print-json`. Timing and stochastic training values may differ across systems.

## M7/M8 measured synthetic scaling and KV-cache smoke

All values below are from `eval/study.py` and `bench/kv_cache.py`, recorded in [scaling-ci.json](../results/scaling-ci.json) and [kv-cache-ci.json](../results/kv-cache-ci.json). CI environment: Linux, Python 3.12.15, CPU count 4, CUDA=false; source commit `344df54bbe0fb0762989d6286646b80b2c9c3259`.

**Scaling smoke**: two fixed seeds (11,29), two optimizer steps per run, synthetic repeated English/code text. The apparent lower bits/byte at larger widths is **not** a statistically supported scaling law.

| Model | Parameters | Mean validation bits/byte | Seeds |
| --- | ---: | ---: | ---: |
| width16_layers1 | 7,536 | 7.9607 | 2 |
| width24_layers2 | 20,832 | 7.9068 | 2 |
| width32_layers3 | 46,624 | 7.8356 | 2 |

**Incremental KV-cache smoke**: the reported speedups are full-prefix wall time divided by cached wall time, with only two repeats. They are noisy and not transferable to other CPUs, GPUs, or context lengths.

| Prompt context length | Full recompute (ms) | KV cached (ms) | Observed speedup |
| --- | ---: | ---: | ---: |
| 8 | 4.596 | 2.868 | 1.60× |
| 16 | 3.614 | 2.747 | 1.32× |
| 32 | 4.395 | 2.911 | 1.51× |

To regenerate tables, run `python results/study_render.py results/scaling-ci.json`. For KV, inspect the measured JSON. No confidence interval is claimed with two timing repeats.

## v0.1 offline randomly initialized Hugging Face GPT-2 reference test

On the GitHub Actions [reference job](https://github.com/DanyVX/transformer-from-scratch/actions/runs/38048948620) at source commit `6adf6857b1f410e1ccbc4e72a00663c618e96775`, the small (2-layer, 32-width, 256-vocabulary) model and the independent Hugging Face GPT2LMHeadModel were run with seed 1234 and three deterministic prompts. The **observed maximum absolute logit difference was 5.96046448 × 10⁻⁸**; the test passed with a 1e-4 absolute/relative per-logit tolerance. Dependencies included transformers==4.38.2, torch==2.5.1+cpu and Python 3.11. No third-party model weights or external training data were downloaded. This **does not verify 124M released pretrained GPT-2 weights**, nor tokenizer ID equivalence with tiktoken.
