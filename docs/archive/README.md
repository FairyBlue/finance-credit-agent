# 旧文档原文归档

整理日期：2026-09-14。日常阅读请使用 [docs 导航](../README.md)和四份阶段总结。
本目录将整理前的 49 份文档压成一个 [ZIP 原文包](pre_2026-09-13_documents.zip)，
包括当时尚未提交的本地补充；不是仅从 Git 历史恢复的旧版本。压缩包不包含模型、数据或运行日志。

34 份过程文档已合并，G1/M1 结果与接手指南等保留精简入口；设计/协议主体继续保留。
删除的是日常目录中的重复过程文件，原始失败、修正前判断、精确执行身份和哈希仍可在 ZIP 中追溯。
9 月 13、14 日的 12 份文档未改动，也没有装入本次旧文档包。

## 查找旧文件

ZIP 内保持仓库相对路径，例如 `docs/2026-09-07_M2_SCREENING_RESUMPTION.md`。
逐文件字节数、SHA-256、去向，以及本次保护的 12 份文档哈希见 [manifest.json](manifest.json)；
包内另有 `MANIFEST.json`。历史 run 中绑定的旧文档哈希应核对包内原文，不应拿新总结替代。

已有实验的 `runs/` 收据和原文引用没有改写；遇到旧文件名可按下表定位，
需要原文本时解压到临时目录，不覆盖当前文档：

```bash
python3 -m zipfile -e docs/archive/pre_2026-09-13_documents.zip /tmp/finance-credit-docs-originals
```

## 已合并的旧文件

| 原文件名（相对 docs） | 现在读哪里 |
|---|---|
| `2026-08-27_H100_G1_AUDIT_RESULTS.md` | [一：H100 与 M1](../history/01_H100_AND_M1.md) |
| `2026-08-27_H100_G1_VALIDATION_RUNBOOK.md` | [一：H100 与 M1](../history/01_H100_AND_M1.md) |
| `2026-08-30_H100_G1_P1_INTEGRATION_HANDOFF.md` | [一：H100 与 M1](../history/01_H100_AND_M1.md) |
| `2026-08-30_H100_G1_P1_VALIDATION_RUNBOOK.md` | [一：H100 与 M1](../history/01_H100_AND_M1.md) |
| `2026-08-30_H100_G1_PRE_SMOKE_VALIDATION_RESULTS.md` | [一：H100 与 M1](../history/01_H100_AND_M1.md) |
| `2026-09-06_M2_SCREENING_RUNBOOK.md` | [二：screening 工程](../history/02_M2_SCREENING_ENGINEERING.md) |
| `2026-09-07_M2_SCREENING_EXPERIMENT_REPORT.md` | [二：screening 工程](../history/02_M2_SCREENING_ENGINEERING.md) |
| `2026-09-07_M2_SCREENING_RESUMPTION.md` | [二：screening 工程](../history/02_M2_SCREENING_ENGINEERING.md) |
| `2026-09-08_DISK_CLEANUP_AUDIT.md` | [二：screening 工程](../history/02_M2_SCREENING_ENGINEERING.md) |
| `2026-09-08_M2_GENERATION_DIAGNOSTIC_PLAN.md` | [二：screening 工程](../history/02_M2_SCREENING_ENGINEERING.md) |
| `2026-09-08_M2_GENERATION_DIAGNOSTIC_REPORT.md` | [二：screening 工程](../history/02_M2_SCREENING_ENGINEERING.md) |
| `2026-09-08_M2_SINGLE_TOOL_DIAGNOSTIC_PLAN.md` | [二：screening 工程](../history/02_M2_SCREENING_ENGINEERING.md) |
| `2026-09-08_M2_THINKING_SINGLE_TOOL_DIAGNOSTIC_PLAN.md` | [二：screening 工程](../history/02_M2_SCREENING_ENGINEERING.md) |
| `2026-09-09_M2_DECISION_DIAGNOSTIC_PLAN.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-09_M2_DECISION_DIAGNOSTIC_REPORT.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-09_M2_HOLD_DIAGNOSTIC_PLAN.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-09_M2_HOLD_DIAGNOSTIC_REPORT.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-09_M2_OBSERVATION_DIAGNOSTIC_PLAN.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-09_M2_OBSERVATION_DIAGNOSTIC_REPORT.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-09_M2_R5_ENGINEERING_REPORT.md` | [二：screening 工程](../history/02_M2_SCREENING_ENGINEERING.md) |
| `2026-09-12_M2_ACTION_SUPPORT_DIAGNOSTIC_PLAN.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-12_M2_ACTION_SUPPORT_DIAGNOSTIC_REPORT.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-12_M2_DECODE_IMPORT_ATTRIBUTION_CORRECTION.md` | [四：概率排查](../history/04_INFERENCE_PROBABILITY.md) |
| `2026-09-12_M2_DECODE_PAIRED_PLAN.md` | [四：概率排查](../history/04_INFERENCE_PROBABILITY.md) |
| `2026-09-12_M2_DECODE_PAIRED_REPORT.md` | [四：概率排查](../history/04_INFERENCE_PROBABILITY.md) |
| `2026-09-12_M2_DECODE_PAIRED_V2_AMENDMENT.md` | [四：概率排查](../history/04_INFERENCE_PROBABILITY.md) |
| `2026-09-12_M2_DECODE_PAIRED_V3_AMENDMENT.md` | [四：概率排查](../history/04_INFERENCE_PROBABILITY.md) |
| `2026-09-12_M2_DECODE_RECOVERY_AMENDMENT.md` | [四：概率排查](../history/04_INFERENCE_PROBABILITY.md) |
| `2026-09-12_M2_LOGPROB_ALIGNMENT_PLAN.md` | [四：概率排查](../history/04_INFERENCE_PROBABILITY.md) |
| `2026-09-12_M2_LOGPROB_ALIGNMENT_REPORT.md` | [四：概率排查](../history/04_INFERENCE_PROBABILITY.md) |
| `2026-09-12_M2_SEMANTICS_DIAGNOSTIC_PLAN.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-12_M2_SEMANTICS_DIAGNOSTIC_REPORT.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-12_M2_TRANSFER_DIAGNOSTIC_PLAN.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |
| `2026-09-12_M2_TRANSFER_DIAGNOSTIC_REPORT.md` | [三：行为诊断](../history/03_M2_BEHAVIOR_DIAGNOSTICS.md) |

## 同时保留的原始版本

ZIP 还保存整理前的 G1/M1 结果、T2/manifest 结果、M2 计划 v2、两份合作者设计意见、
R1—R5 修订、9 月 12 日 Train 实用计划、EXPERIMENT_HANDOFF 和 H100_COMPATIBILITY。
保留路径不表示文本绝对未变：部分文件精简，部分只修正指向已合并日志的导航链接；
具体整理前原文以清单哈希为准。最新安装说明、实验状态继续看日常文档。
