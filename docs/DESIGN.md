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

## M1 byte-level tokenizer decisions (2026-10-10)

| Alternatives | Choice | Why |
| --- | --- | --- |
| Normalize Unicode vs preserve input | No normalization | Round-trip fidelity, including decomposition and byte-level invalid inputs. |
| Regex-free splitting vs GPT-2-like regex | Unicode-aware regex pre-tokenization | Comparable training boundaries without copying a reference library's encode/merge code. |
| Repeated whole-piece rescans vs ranked heap encoding | Rank heap plus linked-neighbor indices | Encoding complexity grows more gently for long tokens; stale heap entries are validated. |
| Incremental pair-frequency caches vs recount per training merge | Deterministic whole-corpus recount for M1 | Easier to verify; large-scale trainer performance is a documented limitation. |
| Treat special-looking text as control tokens by default vs explicit opt-in | Explicit allowed_special only | User input cannot silently inject special controls. |
| UTF-8 only vs raw byte support | encode_bytes/decode_bytes exact byte round trip | Supports malformed UTF-8 with no out-of-vocabulary bytes. |

Merge frequency ties are resolved by the lexicographically smallest pair IDs. Token IDs reserve 0–255 for raw bytes, then allocate one ID per merge by rank, followed by special tokens. This differs from exact pretrained GPT-2 merge ranks; no token-by-token reference parity is claimed. Decode to str uses replacement for invalid UTF-8; decode_bytes preserves bytes exactly.
