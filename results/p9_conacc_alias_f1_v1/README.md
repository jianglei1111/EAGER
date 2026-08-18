# Current paper result tables

This directory is the compact result index for the current manuscript. All
answer values were recomputed offline from frozen predictions. ConAcc tests
whether the normalized primary canonical answer occurs in the prediction.
Alias-aware token F1 takes the maximum token overlap over the canonical answer
and benchmark-provided aliases. Empty and failed outputs remain in the
denominator.

| Paper content | File |
| --- | --- |
| Table 1(a), complete-system answer results | `system_metrics.csv` |
| Table 1(b), matched EAGER--Standalone effects | `main_matched.csv` |
| Reader-controlled comparison discussed in RQ2 | `common_reader.csv` |
| Table 2 component interventions | `component_interventions.csv` |
| Generated-RQ versus repeated-question control | `repeated_question.csv` |
| Fixed-trajectory last-step-only state diagnostic | `last_step_only.csv` |
| Table 3 dependency-structure profiles | `structure.csv` |
| Table 4 native-corpus Hard-200 comparison | `hard200.csv` |
| Figure 2 plotting inputs | `figure2/figure2_scaling_cost_v1.csv` and `.json` |

`main_matched.csv` applies Holm correction over six dataset--controller
comparisons per answer metric. `component_interventions.csv` applies Holm
correction over the 21 reported interventions per answer metric. The remaining
tables report their stated paired confidence intervals or descriptive values.

The release includes aggregate tables, executable EAGER code, configurations,
split IDs, and result-bound wrapper snapshots. Hosted-model response caches and
large retrieval checkpoints are intentionally omitted.
