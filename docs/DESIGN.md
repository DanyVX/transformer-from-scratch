# Design decisions

## M0 scope

Validated model/train profile schemas, cross-platform CLI and reproducible environment provenance. No actual model or benchmark code exists yet.

## Decision log

| Date | Options | Choice | Why |
| --- | --- | --- | --- |
| 2026-10-10 | GPU-required vs CPU-first | CPU-first tiny config | This execution environment exposes no CUDA GPU. |
| 2026-10-10 | Hand-written parsing vs Pydantic + YAML | Strict Pydantic v2 schemas | Invalid heads, unrecognized fields and schedule boundaries fail fast. |
| 2026-10-10 | RoPE vs learned absolute positions | Learned for GPT-2-compatible baseline | RoPE is reserved for a controlled ablation. |
| 2026-10-10 | Hosted tracking vs offline files | JSON/CSV and scripts | No secret tokens or internet needed to reproduce output. |
| 2026-10-10 | Bash-only Makefile vs Python tasks | Cross-platform Python task runner, optional Make wrapper | Windows and Linux can call the same commands. |

## Profiles

- tiny: 512 vocab, 128 context, 6 layers, 4 heads, width 128. Analytically predicted 1,271,808 parameters, within the target 1–3M.
- small: 8192 vocab, 256 context, 8 layers, 6 heads, width 384. Analytically predicted 17,440,512 parameters, within the target 10–30M.
- gpt2-124m: 50257 vocab, 1024 context, 12 layers, 12 heads, width 768. Analytically predicted 124,439,808 parameters with GPT-2-style weight tying.

Predictions use architectural dimensions, not a built PyTorch module. The gpt2-124m training profile is a placeholder for compatibility tests, **not** a full pretraining recipe. Users can run GPU-heavy tests on Colab or Kaggle in later milestones.
