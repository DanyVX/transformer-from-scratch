# Limitations and negative results

- M0 includes only scaffolding, environment reporting, strict configs and tests. Tokenizer, attention, model, data, training, generation and evaluation milestones remain unimplemented.
- The available execution environment is CPU-only. Whether the owner's machine has a GPU is unknown.
- Tiny CPU training time and quality are not measured.
- Reference GPT-2 logits have not been verified.
- The YAML loader currently uses the editable source checkout; wheel distribution of named YAML configs is not tested.
- This runner lacks locally installed ruff, mypy, hypothesis, pip-audit and pre-commit. CI results have not yet been verified.
- uv.lock is not generated. The version-bounded pyproject is not an exact lockfile.
- No dataset, pretrained-weight or dependency license due diligence has been completed.
- No Dockerfile or usable training demo exists.
- Hardware memory detection may report the host rather than container cgroup limits.
- The bench task intentionally fails until a real measurement script exists.
- No fairness/bias evaluations or dataset splits exist yet.
- All project-spec numerical edge cases remain open until the corresponding implementation milestone.

## M1 tokenizer limitations

- Tokenizer training recounts pair frequencies across all pieces after every merge. It is correct for small corpora but not tuned for massive training corpora.
- Unicode text is not normalized. This preserves original input but may reduce compression for canonically equivalent spellings.
- Model serialization writes JSON and merges.txt separately and does not yet use an atomic directory swap.
- A vocabulary can end up smaller than requested when the supplied corpus contains no more mergeable pairs.
- No real GPT-2 merge-rank import, tiktoken token-level parity, or third-party tokenizer reference benchmark has been run.
- Unicode Urdu/Arabic compression on synthetic samples will not establish general-language efficiency.
- String decode replaces malformed UTF-8 by default; use decode_bytes for exact arbitrary-byte reconstruction.
- Heavy/real-world tokenizer benchmarks and long-corpus training throughput are outstanding.

## Engineering completion boundaries

- Full pretrained GPT-2 124M equivalence is unverified; its pretrained weights cannot be downloaded automatically when a single artifact exceeds 500 MB. Reference-weight mapping exists, but a toy randomized HF test is not the flagship proof.
- Ablations use two short seeds on a tiny synthetic corpus. They are smoke experiments, not statistically robust model comparisons or transferable scaling-law estimates.
- The trainer uses CPU/GPU fp32 by default. AMP (bf16/fp16), torch.compile, distributed training and advanced loss spike policies are not validated.
- No true dataset-backed TinyShakespeare or TinyStories evaluation has been performed; licenses/checksums require independent review.
- No working pretrained model, production service, or human-generated benchmark is provided.
- A small-model CPU demo does not demonstrate useful language ability or a sub-five-minute runtime on arbitrary hardware.
