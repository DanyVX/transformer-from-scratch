# Failure diary: real engineering findings

| Date | Symptom | Root cause | Fix and guardrail |
| --- | --- | --- | --- |
| 2026-10-10 | BPE CI lint failed | Adjacent zip iterators differ by one element | Explicit `strict=False`; keep Ruff B905 enabled |
| 2026-10-10 | Attention needed future-token guarantees | A shape-correct attention module can still leak future tokens | Test perturbing future token embeddings and comparing earlier outputs, for explicit and SDPA paths |
| 2026-10-10 | Strict type checks failed on NumPy memmap | Mypy requires dtype and shape parameters for numpy 1.x | Type `np.memmap[Any, Any]`, validate sampler state and test resume |
| 2026-10-10 | A nonfinite loss could destroy the last good checkpoint | Error path saved a poisoned model as if it were recoverable | Retain the previous trusted checkpoint; regression test injects NaN |
| 2026-10-10 | CI repeatedly failed formatting on stacked branches | Files were committed before running the same Ruff formatting as CI | Apply exact Ruff diffs and require green PR checks before merging |
| 2026-10-10 | M7 branch conflicted after M6 squash merge | A stacked branch retained ancestry from intermediate feature branches | Rebase by creating a new commit tree on the current main, then verify CI |
| 2026-10-10 | Source checkout loaded profiles but wheel-only install could not | YAML was stored only in repository-level `configs/` | Bundle profile files inside `tfs` package and include a wheel smoke test |

These are real implementation/process findings, not claims about model quality. Full GPT-2 pretrained parity and real-data generalization are still unknown.
