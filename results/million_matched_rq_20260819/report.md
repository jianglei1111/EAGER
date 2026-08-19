# Million-corpus matched Standalone-RQ control

All datasets use the same 1,000 question IDs. The protocol audit matches
RQ steps, admitted-candidate budget, maximum Qwen pairs, final evidence
budget, corpus SHA256, and question-ID SHA256. ConAcc is canonical-answer
containment; F1 is the maximum token F1 over canonical answer and aliases.
Wall-clock latency is not compared because runs occurred at different times.

## EAGER-RQ Hard-200 minus Standalone-RQ at million scale

| Dataset | Metric | EAGER | Standalone | Delta | 95% CI | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | conacc | 59.1 | 58.1 | +1.0 | [-1.0, 3.1] | 0.7904 |
| HotpotQA | f1 | 65.5 | 64.5 | +1.0 | [-1.1, 3.1] | 0.4758 |
| HotpotQA | gptacc | 64.0 | 62.9 | +1.1 | [-1.2, 3.5] | 0.8127 |
| HotpotQA | candidate_full | 74.8 | 64.8 | +10.0 | [7.2, 12.8] | 8.359e-12 |
| HotpotQA | final_full | 69.5 | 58.4 | +11.1 | [8.0, 14.2] | 5.09e-12 |
| 2Wiki | conacc | 71.1 | 63.4 | +7.7 | [4.9, 10.5] | 2.749e-07 |
| 2Wiki | f1 | 69.2 | 61.2 | +8.1 | [5.4, 10.7] | 0.0003 |
| 2Wiki | gptacc | 67.0 | 59.8 | +7.2 | [4.4, 10.0] | 2.271e-06 |
| 2Wiki | candidate_full | 85.6 | 66.3 | +19.3 | [16.3, 22.2] | 2.006e-36 |
| 2Wiki | final_full | 78.8 | 61.7 | +17.1 | [13.8, 20.4] | 8.068e-23 |
| MuSiQue | conacc | 31.9 | 32.6 | -0.7 | [-2.9, 1.5] | 0.7904 |
| MuSiQue | f1 | 43.2 | 41.9 | +1.3 | [-0.8, 3.3] | 0.4758 |
| MuSiQue | gptacc | 38.4 | 37.3 | +1.1 | [-1.3, 3.6] | 0.8127 |
| MuSiQue | candidate_full | 47.8 | 39.2 | +8.6 | [6.0, 11.3] | 6.787e-10 |
| MuSiQue | final_full | 40.6 | 32.8 | +7.8 | [5.2, 10.5] | 2.51e-08 |
