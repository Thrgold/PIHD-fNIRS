# Script/result index

Original numerical script names are retained to preserve dynamic source imports.
Result directories use zero-padded stage numbers for sorting.

| Script prefix | Purpose | Archived result directory |
|---|---|---|
| 0 | Feature construction from processed subjects; optional | data, not an experiment result |
| 2 | Real-data decomposition and parameters | 02_decomposition |
| 6.5 | Full-cohort key ablations | 06.5_key_ablation |
| 6.6 (v2 only) | Matched-smoothness control; requires 6.5 Full results | 06.6_smoothness_v2 |
| 6.7 | Three-seed matched-smoothness statistics | 06.7_smoothness_statistics |
| 6.8 | ODE-weight sensitivity | 06.8_lambda_ode |
| 7 (9 methods only) | Classification and reusable features | 07_loso_classification |
| 8.5 | LOSO permutation and paired comparison; requires 7 | 08.5_loso_permutation |
| 8.6 | Task-minus-preceding-rest analysis; uses 7 for classification summary | 08.6_task_rest |
| 8.7 | Correct/shifted masks, corrected implementation | 08.7_wrong_mask_v2 |
| 8.8 | Cross-condition statistics; requires three seeds of 8.7 | 08.8_wrong_mask_statistics |
| 12.5 | Full optimization runtime; dynamically reads script 2 | 12.5_runtime |
| 13 and 13.5 | Data-derived figure generation | final selected assets in figures |

Seed-specific final analyses retain 42, 123, and 999. Runtime reporting retains
only the sanitized seed-42, n=30 aggregate benchmark; the n=2 smoke test is excluded.
Scripts 6.7 and 8.8 aggregate seeds rather than requiring three independent runs.
Consult each script's header for optional flags.

Within each numbered directory, the original root/json/npy placement is preserved.
For example `result/08.5_loso_permutation/npy/` contains binary results and its
`json/` contains summaries. Preparation reconstructs their historical locations.

Excluded from the public package: conflicting legacy decomposition summaries,
the task-window-only diagnostic (whose legacy metadata was misleading),
superseded 7-method results, old small-cohort ablation, wrong-mask v1,
smoothness v1, checkpoints, unreported exploratory analyses, stale plotting
alternatives, temporary outputs and manuscript source files. These remain
recoverable from the local pre-packaging backup and are not evidence for the
reported final results.
