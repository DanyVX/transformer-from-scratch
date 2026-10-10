# Transformer From Scratch

A decoder-only GPT-style transformer and byte-level BPE tokenizer built from first principles in PyTorch. This project prioritizes correctness, measured results, and a documented failure diary over unsupported performance claims.

![CI](https://github.com/DanyVX/transformer-from-scratch/actions/workflows/ci.yml/badge.svg)

## Demo

TBD (not yet implemented). M0 only validates configurations and reports runtime metadata.

## Key results

| Experiment | Result |
| --- | --- |
| GPT-2 124M reference-logit agreement | TBD (not yet measured) |
| Causal-mask leakage test | TBD (not yet measured) |
| KV-cache speedup | TBD (not yet measured) |
| Tokenizer tokens/second | TBD (not yet measured) |
| Validation loss / bits-per-character | TBD (not yet measured) |

M0's displayed parameter counts are analytical predictions for the planned model, not training results.

## Why this exists

Understanding a transformer requires more than copying a notebook. The implementation will cover BPE merge ranks, attention masking, reference-weight parity, gradient accumulation, exact resume, and KV cache correctness.

## Architecture

The planned path is: token IDs -> token and position embeddings -> repeated pre-LN causal attention plus GELU feedforward blocks -> final LayerNorm -> tied language-model head. Attention and model code begin in M2 and M3. See [design](docs/DESIGN.md).

## Quickstart

From a clean clone, using Python 3.11+:

1. Install lightweight M0 dependencies with python -m pip install -e ".[dev]".
2. Print the named profiles with python -m tfs profiles.
3. Validate the tiny profile with python -m tfs config --profile tiny.
4. Capture the runner environment with python -m tfs environment.
5. Run the CPU-only M0 tests with python scripts/tasks.py test-fast.

The eventual make demo training command is not implemented yet. These commands validate M0 only.

## Usage examples

- python -m tfs config --profile small
- python -m tfs config --profile gpt2-124m
- python scripts/tasks.py lint
- python scripts/tasks.py typecheck
- python scripts/tasks.py test

## Configuration

Named YAML files live in configs/. The environment variables described in .env.example control the default profile, logging and output directory. Tiny is CPU-oriented; small and gpt2-124m are intended for GPU use and are **not runnable training pipelines yet**.

## Reproducing benchmarks

No benchmarks have been run. M1 onward will save measurements under results/ with the current commit SHA and hardware/software metadata, using tfs.provenance.write_result. Until a benchmark script exists, the bench task exits with an explicit error.

## Design decisions

See [decision log](docs/DESIGN.md) and [references](docs/REFERENCES.md).

## Limitations

No tokenizer, model, training, inference, reference weight-loading, real benchmark, dataset or release exists in M0. See [limitations](docs/LIMITATIONS.md).

## Roadmap

M0 setup -> M1 BPE -> M2 attention -> M3 GPT model + GPT-2 logit parity -> M4 data pipeline -> M5 training -> M6 generation + KV cache -> M7 experiments + failure diary -> M8 release.

## License

MIT for the original scaffold. External dataset, pretrained-weight and dependency license checks are outstanding; see DATA.md and MODEL_CARD.md.

## Acknowledgements and citations

See [references](docs/REFERENCES.md). No tutorial implementation was copied.
