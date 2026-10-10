# Failure diary

Only record actual bugs with symptom, root cause, fix, and a regression test. There are no confirmed model mistakes yet because the implementation is at M0.

| Date | Actual failure | Diagnosis | Fix | Guarding test |
| --- | --- | --- | --- | --- |
| TBD | No entries yet | TBD | TBD | TBD |

## M1 actual issue

- **2026-10-10:** Initial CI lint caught a missing explicit strict= parameter in zip(chunk, chunk[1:]) during pair counting. Adjacent pair iterators intentionally differ in length by one, so strict=False is the correct contract. Fixed the call and left Ruff B905 enabled to prevent recurrence. This was a code-quality issue, not a measured accuracy failure.
