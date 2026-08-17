# Main-table verification (2026-07-29)

This audit verifies that the active paper table uses the latest complete EAGER
V1 runs rather than any pre-V1 tuning artifact.

## Result lineage

The six registered EAGER rows come from:

- `eager_v1_20260728_hotpotqa_rq_main1000_v1`
- `eager_v1_20260728_hotpotqa_sc_main1000_v1`
- `eager_v1_20260728_2wiki_rq_main1000_v1`
- `eager_v1_20260728_2wiki_sc_main1000_v1`
- `eager_v1_20260728_musique_rq_main1000_v1`
- `eager_v1_20260728_musique_sc_main1000_v1`

Every run is `complete` and `validated`, with exactly 1,000 predictions, 1,000
retrieval traces, and 1,000 GPTAcc decisions. No newer complete EAGER
`main1000` artifact exists in the workspace. Earlier tuning artifacts are under
`artifacts/test_runs/wsdm_history/pre_v1_20260728/` and are not paper inputs.

All files bound by the six canonical preflights still match the current
checkout except five orchestration wrappers (`028` through `032`) that were
standardized after execution. The retrieval kernel, controllers, prompts,
configuration, prepared-data manifests, and index summaries all match. The
original preflight hashes remain authoritative for the wrapper bytes.

## Independent recomputation

Run:

```powershell
conda run -n raptor-hotpotqa `
  python wsdm\scripts\build_table1_canonical_fairness_v2.py
```

The script joins every method to the same 1,000 formal question IDs and
recomputes ConAcc from `annotations.answer` with the shared EAGER normalizer.
The resulting EAGER values are:

| Dataset | Config. | ConAcc / GPTAcc | Candidate / Final Full |
| --- | --- | ---: | ---: |
| HotpotQA | RQ | 65.8 / 71.5 | 96.3 / 90.0 |
| HotpotQA | SC | 65.9 / 71.9 | 92.5 / 91.7 |
| 2WikiMultiHopQA | RQ | 76.9 / 72.6 | 97.0 / 92.8 |
| 2WikiMultiHopQA | SC | 76.0 / 71.9 | 92.3 / 91.9 |
| MuSiQue | RQ | 39.2 / 44.8 | 65.9 / 52.8 |
| MuSiQue | SC | 42.3 / 48.4 | 77.0 / 66.0 |

The underlying metrics also include normalized EM and token-level answer F1 as
supplementary checks. The canonical mainline keeps ConAcc/GPTAcc as its primary
answer pair. Errors are never filtered from the denominator: the HotpotQA-SC
artifact has one `question_execution_error` caused by the per-step LLM-call
budget (`4 > 3`), and it is counted as an incorrect answer.

For the hosted judge and generation calls, the requested model is the
`gpt-4o-mini` alias. Where the provider exposed it, `llm_calls.jsonl` records
`gpt-4o-mini-2024-07-18`; other records expose only the alias. This distinction
is retained as provenance rather than presented as a universal immutable API
snapshot. Local MiniLM/Qwen revisions are fixed by the model registry.

MuSiQue is the only apparent discrepancy between raw runner metrics and the
paper. Its raw runner used answer aliases and recorded ConAcc 43.2/47.5. The
paper applies the same canonical-only contract to every method, which removes
40 RQ and 52 SC alias-only matches and yields 39.2/42.3. GPTAcc already uses the
canonical answer and therefore remains 44.8/48.4.

The two lightweight retrieval controls are not a single-variable Qwen
ablation. `MPNet Dense Top-5` directly retrieves five MPNet candidates;
`MiniLM + Qwen Top-5` retrieves 20 MiniLM candidates and reranks them to five.
The paper labels and describes them accordingly.

## Additional prefix evidence-quality check

The optional prefix diagnostic is computed offline from the same six
`metrics.json` files, without rerunning inference. It macro-averages native
support precision, recall, and F1 of the first five terminal-state units in
`evidence_quality_at5.csv`.
HotpotQA and 2Wiki use sentence annotations; MuSiQue uses paragraph
annotations. The metric complements, rather than replaces, Candidate Full /
Final Full: it measures prefix concentration, while the full-support rows
measure complete-chain reachability.

Recompute it with:

```powershell
python wsdm\scripts\build_evidence_quality_at5.py
```

## Hard-200 complete-cohort verification

The active Hard-200 follow-up is under
`artifacts/test_runs/wsdm/scaling_hard_budget_v3_single/`. Each of the three
runs has 1,000 unique formal handles aligned with the uncapped EAGER-RQ run on
the same native pooled corpus. The summary script also validates all retrieval
trace steps and finds a maximum `expanded_candidate_count` of 200 in each
dataset. Errors are retained in the 1,000-question denominator.

The paper-facing canonical answer values are 64.6/70.5, 74.2/70.1, and
37.9/44.9 (Hard-200 ConAcc/GPTAcc) for HotpotQA, 2Wiki, and MuSiQue,
respectively. The corresponding uncapped values are 65.8/71.5, 76.9/72.6,
and 39.2/44.8. MuSiQue stored runner ConAcc is alias-aware; the summary keeps
that raw field but recomputes the reported value from the primary answer in
`annotations.jsonl` using the shared normalizer.

Run:

```powershell
python wsdm\scripts\summarize_scaling_hard_budget_v2.py `
  --hard-root artifacts/test_runs/wsdm/scaling_hard_budget_v3_single
```

The older `scaling_hard_budget_v2/` directory is failed provenance from a
parallel GPU-OOM attempt and is not a paper input.
