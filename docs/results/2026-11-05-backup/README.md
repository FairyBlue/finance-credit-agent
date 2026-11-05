# 2026-11-05迁移：关键结果快照

采集时间：2026-11-05T21:12:04+08:00。目标为私有仓库[FairyBlue/finance-credit-agent](https://github.com/FairyBlue/finance-credit-agent)。

本次选定275份关键结果，共9,826,668字节。文件保留原来的`runs/`路径，便于从状态页和研究计划直接访问。
[manifest.json](manifest.json)记录每份文件的大小与SHA-256；哈希对应本次Git暂存版本，不会把随后产生的实时文件冒充同一快照。

保留报告、研究回顾、小型汇总、冻结合同和身份收据；单文件上限1 MiB。模型权重、optimizer、原始数据、完整请求/响应及轨迹不纳入。较大读数和未选文件仍在本地，manifest列出被排除的引用目标。

运行中的V3状态按采集时实际内容保存；不因本次迁移将未完成实验改称完成。后续结果需要另行提交。

## 当前研究入口

| 运行 | 报告 | 研究回顾 | 文件数 |
| --- | --- | --- | --- |
| `agentic-rl-learning-20261030-v2` | [报告](../../../runs/agentic-rl-learning-20261030-v2/REPORT.md) | [回顾](../../../runs/agentic-rl-learning-20261030-v2/RESEARCH_REVIEW.md) | 32 |
| `agentic-rl-transfer-probe-20261105-v1` | [报告](../../../runs/agentic-rl-transfer-probe-20261105-v1/REPORT.md) | — | 6 |
| `agentic-rl-mc-20261030-v1` | [报告](../../../runs/agentic-rl-mc-20261030-v1/REPORT.md) | [回顾](../../../runs/agentic-rl-mc-20261030-v1/RESEARCH_REVIEW.md) | 11 |
| `m3-financial-credit-20261030-v1` | [报告](../../../runs/m3-financial-credit-20261030-v1/REPORT.md) | — | 9 |
| `m3-financial-evidence-20261030-v2` | [报告](../../../runs/m3-financial-evidence-20261030-v2/REPORT.md) | — | 14 |
| `agentic-rl-coverage-20261105-v3` | [报告](../../../runs/agentic-rl-coverage-20261105-v3/REPORT.md) | — | 8 |

## 历史结果

| 运行 | 已纳入文件数 | 字节 |
| --- | --- | --- |
| `disk-cleanup-20261025` | 1 | 1,919 |
| `m2-action-pooling-comparison-20261012-v2` | 1 | 1,592 |
| `m2-action-pooling-comparison-20261012-v3` | 1 | 1,652 |
| `m2-action-pooling-comparison-20261012-v4` | 1 | 1,873 |
| `m2-action-pooling-comparison-20261013-v5` | 2 | 3,604 |
| `m2-action-pooling-comparison-20261013-v6` | 4 | 283,309 |
| `m2-action-pooling-mechanism-audit-20261014-v1` | 3 | 310,904 |
| `m2-broad-rloo-order-main-20261028-v1` | 1 | 4,589 |
| `m2-common-validity-penalty-20260917` | 1 | 4,601 |
| `m2-counterfactual-order-demo-20261027-v1` | 5 | 838,152 |
| `m2-credit-oracle-probe-20261014-v1` | 3 | 609,993 |
| `m2-fixed-j15-policy-probe-20261024-v1` | 3 | 42,268 |
| `m2-grid-preparation-20261006` | 1 | 1,094 |
| `m2-low-memory-handoff-20261004` | 2 | 2,847 |
| `m2-low-memory-pilot-20260930` | 1 | 864 |
| `m2-onpolicy-credit-20261018-v1` | 6 | 148,480 |
| `m2-onpolicy-credit-replication-20261019-v1` | 6 | 200,517 |
| `m2-onpolicy-credit-replication-selection-20261019-v1.json` | 1 | 7,702 |
| `m2-order-recovery-20261011-v1` | 1 | 218,109 |
| `m2-order-recovery-20261011-v2` | 1 | 40,989 |
| `m2-order-recovery-acceptance-all-20261011-v2` | 1 | 509,675 |
| `m2-order-recovery-audit-20261011` | 3 | 101,335 |
| `m2-order-recovery-episodes-20261011-v2` | 1 | 31,332 |
| `m2-order-recovery-learning-stage1-20261011-v2` | 2 | 3,176 |
| `m2-order-recovery-sft-20261011-v1` | 1 | 3,134 |
| `m2-parameter-grid-20261006` | 2 | 3,132 |
| `m2-parameter-grid-20261006-r1` | 2 | 4,070 |
| `m2-parameter-grid-20261008-r2` | 2 | 3,496 |
| `m2-parameter-grid-20261008-r3` | 2 | 3,496 |
| `m2-parameter-grid-20261008-r4` | 2 | 3,496 |
| `m2-parameter-grid-20261008-r5` | 4 | 21,320 |
| `m2-policy-transfer-eval-20260914-continuation` | 1 | 3,555 |
| `m2-portfolio-management-step1-20261028-v1` | 3 | 59,850 |
| `m2-portfolio-teacher-sft-20261028-v1` | 2 | 110,857 |
| `m2-rloo-eight-window-20261025-v1` | 5 | 144,006 |
| `m2-rloo-local-aux-demo-20261026-v1` | 7 | 284,867 |
| `m2-rloo-local-credit-demo-20261025-v1` | 7 | 353,814 |
| `m2-rloo-local-credit-fresh-eval-20261026-v1` | 3 | 273,578 |
| `m2-rloo-terminal-demo-20261025-v3` | 6 | 412,388 |
| `m2-s0-learning-curve-20261024-v1` | 5 | 206,942 |
| `m2-s0-multistep-demo-20261024-v1` | 4 | 45,105 |
| `m2-same-prefix-gate-20261018-v1` | 5 | 128,044 |
| `m2-same-prefix-menu-20261018-v1` | 11 | 177,940 |
| `m2-same-prefix-online-8window-demo-20261024-v1` | 4 | 75,345 |
| `m2-same-prefix-online-demo-20261024-v2` | 4 | 88,785 |
| `m2-same-prefix-online-demo-seed43-20261024-v1` | 4 | 150,749 |
| `m2-same-prefix-sampling-20261018-v1` | 5 | 404,638 |
| `m2-same-prefix-storage-20261024-v1` | 1 | 3,897 |
| `m2-same-prefix-storage-20261024-v2` | 1 | 1,309 |
| `m2-same-prefix-storage-20261024-v3` | 1 | 1,315 |
| `m2-same-prefix-training-demo-20261022-v2` | 2 | 4,914 |
| `m2-same-prefix-training-demo-20261022-v3-eval` | 1 | 2,544 |
| `m2-sampling-profile-eval-20260915` | 1 | 5,132 |
| `m2-sft-four-round-comparison-20261004` | 1 | 1,559 |
| `m2-sft-four-round-comparison-seed43-20261005` | 1 | 1,342 |
| `m2-sft-four-round-comparison-seed43-20261005-recovery` | 1 | 1,581 |
| `m2-state-order-oracle-20261017-v1` | 11 | 1,306,326 |
| `m2-stochastic-continuation-20261018-v3` | 5 | 47,014 |
| `m2-stochastic-credit-comparison-20261018-v1` | 6 | 155,872 |
| `m2-train-coverage-20260915` | 1 | 4,046 |
| `m2-train-coverage-20260916` | 1 | 4,046 |
| `m2-train-coverage-20260916-v2` | 1 | 4,071 |
| `m2-train-coverage-diagnosis-20260917` | 2 | 16,835 |
| `m2-training-tuning-20260914` | 2 | 12,397 |
| `m2-training-tuning-preparation-20260914` | 4 | 236,107 |
| `m3-financial-evidence-20261030-v1` | 2 | 4,658 |
| `m3-mandate-composition-20261029-v1` | 4 | 18,735 |
| `m3-mandate-repair-20261029-v1` | 3 | 24,597 |

Git提交历史和原作者记录保持不变。仓库迁移及这份文件清单本身不重新验证实验，不改变数据区划、预算或OOS边界。
