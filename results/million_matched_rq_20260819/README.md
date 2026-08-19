# Million-scale matched RQ comparison

This directory contains the compact, paper-facing analysis for the full-cohort
Hard-200 EAGER-RQ versus Standalone-RQ scale comparison. Each dataset contains
the same 1,000 formal question IDs at the low and approximately million-unit
corpus levels.

- `system_metrics.csv` reports ConAcc, alias-aware token F1, GPTAcc,
  Candidate/Final Full, reranker pairs, LLM calls, tokens, and error counts.
- `paired_comparisons.csv` contains EAGER-versus-Standalone comparisons,
  within-method scale changes, and paired scale-change interactions, with
  95% bootstrap intervals and Holm-3 adjusted tests.
- `protocol_audit.json` verifies both corpus levels for each dataset: question
  IDs, corpus SHA-256, RQ prompt, retrieval calls, admitted-candidate budget,
  maximum Qwen pairs, and final evidence budget.
- `report.md` is a compact rendering of the million-scale direct comparison.

Canonical ConAcc and alias-aware token F1 were recomputed from the 1,000
predictions per run. HotpotQA and 2Wiki provenance uses supporting sentences;
MuSiQue uses supporting paragraphs. Wall-clock latency is intentionally absent
because the two full-cohort systems ran at different times; Figure 2 supplies
the controlled retrieval/state-construction microbenchmark.
