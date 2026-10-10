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
