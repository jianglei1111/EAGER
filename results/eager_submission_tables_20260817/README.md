# EAGER paper tables

This directory provides compact, directly readable tables for the anonymous
EAGER artifact. The executable package is intentionally limited to EAGER; the
tables retain the comparisons needed to inspect the paper's empirical claims.

| Paper question | Released table |
| --- | --- |
| Primary matched causal comparison | `../standalone_signal_controls_v2_20260816/paired_eager_vs_standalone.csv` |
| EAGER main results | `../eager_v1_main1000_20260728/main_results.csv` |
| Component interventions | `../eager_v1_component_ablation_20260730/paired_deltas.csv` |
| Anchor use and structure profiles | `../eager_anchor_structure_analysis_20260817/` |
| Stepwise-system operating ranges | `stepwise_system_boundaries.csv` |
| RQ/SC structure analysis | `../eager_v1_signal_structure_20260730/table3_signal_structure_main_v1.csv` |
| Bounded scaling plot data | `../scaling/figure2_scaling_cost_v1.csv` |
| GPTAcc audit | `../gptacc_audit/` |

All accuracy and provenance values are percentages. Native-pipeline rows use
each locally executed system's own final-answer path. Common-reader rows freeze
retrieved evidence and regenerate answers with one target-slot reader.
