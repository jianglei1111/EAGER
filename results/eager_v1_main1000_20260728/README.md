# EAGER V1 main1000 canonical results (2026-07-28)

Canonical preflights preserve the historical OpenAI-compatible endpoint used
for these runs; the portable current configuration may point to a different
compatible endpoint. The paper therefore describes a temperature-zero
GPT-4o-mini-compatible model rather than asserting a provider-verifiable model
snapshot. Returned model identifiers are retained where the service supplied
them.

本目录是当前唯一有效的 EAGER 主结果入口。论文、项目概览和后续统计应以
`main_results.csv` 与 `manifest.json` 为准，不再使用 2026-07-28 之前的
EAGER full1000 数字作为主线结果。

新增的 `evidence_quality_at5.csv` 提供同一六组 V1 运行在终局状态前五个证据单元上的
原生支撑 Precision/Recall/F1 宏平均；HotpotQA/2Wiki 使用句子粒度，MuSiQue 使用段落粒度。

## EAGER V1

单元格为 ConAcc / GPTAcc，Candidate Full 与 Final Full 均为段落级完整支持链覆盖率。

| Dataset | Signal | ConAcc / GPTAcc | Candidate / Final Full | LLM calls | Qwen pairs |
|---|---|---:|---:|---:|---:|
| HotpotQA | RQ | 65.8 / 71.5 | 96.3 / 90.0 | 3,000 | 56,851 |
| HotpotQA | SC | 65.9 / 71.9 | 92.5 / 91.7 | 3,862 | 66,319 |
| 2WikiMultiHopQA | RQ | 76.9 / 72.6 | 97.0 / 92.8 | 4,005 | 52,650 |
| 2WikiMultiHopQA | SC | 76.0 / 71.9 | 92.3 / 91.9 | 4,104 | 71,382 |
| MuSiQue | RQ | 39.2 / 44.8 | 65.9 / 52.8 | 5,003 | 58,207 |
| MuSiQue | SC | 42.3 / 48.4 | 77.0 / 66.0 | 4,671 | 74,191 |

ConAcc 按论文冻结口径仅使用 primary answer。MuSiQue runner artifact 中原始的
alias-aware ConAcc 为 43.2/47.5；离线 canonical-only 复算分别剔除了 40/52 个
仅命中 alias 的预测，因此论文主线值为 39.2/42.3。

每组均满足 `run_manifest.status=complete`、
`canonical_validation.status=validated`，并包含 1,000 条 prediction、1,000 条
retrieval trace 和 1,000 道 GPTAcc 分母。执行错误和答案格式错误均按错误答案计入
分母，不从指标中排除。

六组 `metrics.json` 同时保存 canonical normalized EM 和 token-level answer F1，
作为补充稳健性指标；论文主表继续报告冻结协议下的 ConAcc/GPTAcc。HotpotQA-SC
有 1 条 `question_execution_error`（单步 LLM 调用预算为 3，但该题触发了第 4 次），
该题按错误答案保留在分母中，未被删除或事后重跑。

LLM 配置请求的是 `gpt-4o-mini` 别名。服务日志在部分运行中返回不可变模型标识
`gpt-4o-mini-2024-07-18`，另一些记录仅返回请求别名；两种原始记录均保留在
各运行目录的 `llm_calls.jsonl`，因此不能把服务别名表述为所有请求都绑定了同一
个公开快照。MiniLM 与 Qwen reranker 则按 `configs/wsdm/models/registry.json`
中的 revision 和文件哈希固定。

## Main observations

- HotpotQA 上 RQ 与 SC 基本持平，SC 的 GPTAcc 略高。
- 2WikiMultiHopQA 上 RQ 是 EAGER 的最佳固定信号。
- MuSiQue 上 SC 明显优于 RQ，并将 ConAcc/GPTAcc 提高 3.1/3.6 个百分点。
- SC 在 MuSiQue 上相对 RQ 将 Candidate/Final Full 提高 11.1/13.2 个百分点，
  说明其优势主要出现在长依赖链的证据恢复。

HippoRAG 与 HippoRAG2 的论文公开汇总值单独登记在
`../reported_prior_results.csv`，不与本目录的本地逐题输出混合，也不用于配对统计、
provenance 或统一 reader 分析。IRCoT 使用独立的 official-code adaptation，其紧凑逐题
配对统计与统一 reader 汇总随匿名包发布。

## Superseded data

2026-07-28 之前的 EAGER full1000 结果已退出主线并归入历史区；活跃论文、统计脚本和项目说明
只允许引用本目录登记的六组 V1 结果。

当前 V1 artifact 中的 MuSiQue alias-aware 43.2/47.5 只用于 artifact 内部诊断，
不作为论文 ConAcc。

Table 1 的 canonical-only 公平性统计、论文 Table 2 的 V1 component ablation 和论文 Table 3 的
structure interaction 均已重新对齐本目录登记的六组 V1 逐题输出。旧 ablation
结果与最终 V1 执行契约不一致，已经退出活跃论文稿并移入历史区，不得与本目录数值混用。

## Terminal state audit

`terminal_retention_audit.csv` 由
`wsdm/scripts/audit_v1_terminal_retention.py` 从六组原始
`retrieval_traces.jsonl` 重建。CSV 逐组绑定 manifest 中的 trace SHA-256，覆盖全部
6,000 道题。审计以每条运行的 `retrieval_state.retained_evidence` 为累计状态，并区分
直接使用累计状态的运行与带有 terminal-completion 记录的运行。六组中没有 retained
evidence 丢失、顺序破坏或 terminal-completion 状态替换；在存在空槽时，terminal
completion 只允许在累计状态前缀之后追加新单元。该结果验证的是实际 V1 轨迹，不会
修改或重新生成任何预测。

The audit reports 1,773 runs that directly use the accumulated state and 4,227 runs with
an explicit terminal-completion record; no completion record changed the retained state.

## Hard-200 complete-cohort follow-up

The active follow-up is `artifacts/test_runs/wsdm/scaling_hard_budget_v3_single/`.
It evaluates the same 1,000 formal IDs as the uncapped EAGER-RQ runs on the
native pooled corpus. The paired summary is generated with
`wsdm/scripts/summarize_scaling_hard_budget_v2.py` and uses primary-answer-only
canonical ConAcc, while retaining the runner's alias-aware value as an audit
field. The complete-cohort results are:

| Dataset | Hard-200 ConAcc / GPTAcc | Uncapped ConAcc / GPTAcc | Hard-200 Candidate / Final | Uncapped Candidate / Final |
|---|---:|---:|---:|---:|
| HotpotQA | 64.6 / 70.5 | 65.8 / 71.5 | 91.6 / 87.2 | 96.3 / 90.0 |
| 2WikiMultiHopQA | 74.2 / 70.1 | 76.9 / 72.6 | 91.2 / 88.2 | 97.0 / 92.8 |
| MuSiQue | 37.9 / 44.9 | 39.2 / 44.8 | 65.1 / 52.1 | 65.9 / 52.8 |

The 80-question same-environment million-scale endpoints remain the source of
the latency and corpus-growth figure; the 1,000-question native-corpus runs
are the source of the complete quality comparison. The prior parallel V2 run
is failed provenance only and is excluded from paper statistics.
