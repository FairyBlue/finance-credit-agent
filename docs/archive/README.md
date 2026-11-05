# 原文归档与复现

更新：2026-11-05。日常阅读从[当前状态](../CURRENT_STATUS.md)和[文档入口](../README.md)开始；本目录用于查精确原文、旧方案、失败记录与原 SHA。

## 查哪一份包

| 原文包 | 范围 | 文件映射 / SHA |
|---|---|---|
| [11/05 V3精简前草案](pre_2026-11-05_v3_simplification.zip) | 未执行的双臂覆盖对照草案；按用户要求精简为单臂快速试验，现稿见[V3计划](../2026-11-05_AGENTIC_RL_THREE_STEP_PLAN_V3.md) | 包内保持原路径和完整字节，manifest记录SHA；原草案未执行 |
| [10/30改为长期RL Brief前原文](pre_2026-10-30_agentic_rl_brief_v04.zip) | 完整v0.3 Brief（含尚未执行的推理核验三步）及当时方向演变页；现长期目标见[Brief v0.4](../../RESEARCH_BRIEF.md)，具体执行计划在docs迭代 | 包内保持原仓库路径与完整字节，无新增manifest；旧三步未执行，不追溯改成RL计划 |
| [10/30研究计划合并前原文](pre_2026-10-30_research_plan_merge.zip) | 原RESEARCH_BRIEF.md与docs/2026-10-28_NEXT_RESEARCH_DIRECTION.md全部内容；当时合并生成v0.3，现长期目标见[Brief](../../RESEARCH_BRIEF.md) | 包内保留原仓库路径与完整字节，无新增manifest；不含模型或运行产物 |
| [10/29 状态页压缩前原文](pre_2026-10-29_current_status.zip) | 原391行`docs/CURRENT_STATUS.md`，含10月22—29日训练、断线、清理和历次计划；当前页仅保留结果与决策 | 包内保留原路径与完整字节，无新增manifest；不含模型或运行产物 |
| [10/18 主题合并前原文](pre_2026-10-18_topic_consolidation.zip) | 本轮整理前的 58 份文档，包括旧计划、续跑说明、10/11—17 细报、六份旧总结及维护入口的原版本 | [本轮清单](pre_2026-10-18_manifest.json)，`read_now` 指向当前主题 |
| [10/10 前原文](pre_2026-10-10_documents.zip) | 上轮整理前 53 份日期文档及两份旧交接，包含当时未提交的本地材料 | [当轮清单](pre_2026-10-10_manifest.json) |
| [9/13 前原文](pre_2026-09-13_documents.zip) | 更早 49 份原始文档，包括当时已合并的过程文件 | [原清单](manifest.json) |

本轮将 36 份顶层旧文档收进原文包，原六份总结改为[基础与诊断](../history/FOUNDATIONS_AND_DIAGNOSTICS.md)、[共同初始化与 4B 学习](../history/INITIALIZATION_AND_4B_LEARNING.md)、[恢复与池化](../history/RECOVERY_AND_ACTION_POOLING.md)、[机制与 CPU 验证](../history/CREDIT_ORACLE_DIAGNOSTICS.md)。最新安排统一写在当前状态页；原文件不留跳转壳。

四份 10/11、10/12、10/14、10/17 方案仍在原路径，原字节未改，以保持当前运行合同及模型 lineage 的绝对路径/SHA 校验。整理期间新增的 10/18 方案和它引用的 10/17 报告也保留原路径；17 日报告另存原文快照，18 日新增方案未纳入本轮快照。封存方案中的旧相对链接属于历史记录，其目标可在 ZIP 内按原目录找到。

## 复现历史工具

更早的计划已归档，旧诊断、训练 CLI 仍保留当时路径和源码。复现时应使用隔离的历史 checkout，先恢复对应原文，再核对当时源码、配置和数据身份；不能在当前目录只恢复一份旧计划后就认为满足旧合同。

例如把本轮原文解压到临时目录查看（不会覆盖工作区）：

```bash
python3 -m zipfile -e docs/archive/pre_2026-10-18_topic_consolidation.zip /tmp/finance-credit-docs-before-topics
```

ZIP 内保持 `docs/...` 等仓库相对路径，并含 `MANIFEST.json`。manifest 的 SHA 对应包内原始字节；较早两轮的 `read_now` 属于当轮导航，若指向已合并总结，按本轮清单或四主题入口继续定位，不把整理后文本冒充原始身份。

归档只含文档，不含模型、行情、optimizer 或完整运行日志。既有 `runs/` 收据、历史来源路径、两份 `docs/results/` 的 manifest 绑定有效载荷均未改写。历史容量、排队、“当前/下一步”和保存状态只在原记录时点有效；最新资源及可恢复性须核对实际文件与收据。
