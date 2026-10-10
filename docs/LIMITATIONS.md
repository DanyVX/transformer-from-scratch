# Limitations and verification boundaries

The **M0–M8 source implementation, CPU tests, basic benchmarks and one-step synthetic demo are complete**. GitHub Actions verifies Python 3.11 and 3.12, Ruff, strict mypy, tests, dependency audits and secret scanning. These validations do not establish that the model generates fluent text or operates as a production service.

## Unverified or deliberately out of scope

- **GPT-2 124M released pretrained-logit parity:** no official large pretrained weight artifact was fetched. The weight-mapping script accepts a user-supplied, licensed local reference. An offline 2-layer, randomly initialized Hugging Face reference test passed at maximum absolute logit difference 5.96×10⁻⁸; this is not evidence for the released weights.
- **Real-corpus quality:** no genuine TinyShakespeare, TinyStories or other licensed training dataset was downloaded or redistributed. Synthetic train/validation samples cannot establish general language quality, safety, or cross-corpus generalization.
- **Tokenizer reference equivalence:** byte-BPE roundtrips and deterministic merges are tested, but official GPT-2/tiktoken token-ID agreement has not been confirmed. Training pair frequencies are rescanned at each merge and not optimized for large corpora.
- **Performance:** measured tokenizer throughput, KV-cache speedups and scaling studies use small synthetic samples with too few repetitions for statistically reliable speed or scaling claims; GPU benchmarks, mixed precision, torch.compile and distributed training have not been verified.
- **Checkpoint trust:** training checkpoints include optimizer and Python/NumPy/PyTorch RNG state and are loaded with `weights_only=False`. **Never load untrusted .pt files**, which can execute arbitrary code. The local resume and exact CPU replay tests assume a trusted checkpoint and matching configuration.
- **Security and privacy:** no user data, personal biometrics or sensitive identity material is bundled. No moderation, safety, fairness, multilingual utility, or production deployment validation is claimed.

## Known engineering compromises

- UTF-8 byte tokenizer defaults to replacement decoding for malformed bytes; `decode_bytes` provides exact reconstruction.
- Rank-based merges and output files are practical for small corpora, not a high-throughput tokenizer at production scale.
- CUDA-dependent configuration profiles describe geometry and a nominal training plan; their end-to-end execution has not been benchmarked.
- Tiny training on synthetic text is a smoke test. Useful generation and any runtime guarantee on arbitrary hardware are **not** claimed.
- If a Docker environment is unavailable, use the documented cross-platform Python task runner and virtual environment instead of assuming Docker.

See [measured results](RESULTS.md), [architecture](DESIGN.md), and [reproduction guide](COLAB.md).
