# 实验接手指南

**当前进度与 GitHub 备份请先读 [2026-09-17 交接](2026-09-17_PROJECT_BACKUP_HANDOFF.md)。**
本页保留 9 月 14 日整理的接手顺序；随后已完成的训练与诊断见新交接和 [docs 导航](README.md)。

整理日期：2026-09-14。原 9 月 2 日长篇 G1 运行交接已汇入阶段总结，完整旧文本见
[原文归档](archive/README.md)。本文提供接手顺序；具体执行使用对应实验计划和实际代码/产物。

1. 先读 [RESEARCH_BRIEF](../RESEARCH_BRIEF.md)、[EVAL_PROTOCOL](../EVAL_PROTOCOL.md)
   和 [M2 预注册](../M2_V1_PREREGISTRATION.md)，理解研究问题、共同对照和数据边界。
2. 从 [docs 导航](README.md)找当前计划/报告；需要实际运行状态时读
   [M2_SCREENING_LATEST](../runs/M2_SCREENING_LATEST.md)及其中对应的 JSON/服务记录。
3. 了解先前为什么改变方向时，按顺序阅读四份历史总结：
   [H100/M1](history/01_H100_AND_M1.md)、[screening 工程](history/02_M2_SCREENING_ENGINEERING.md)、
   [行为诊断](history/03_M2_BEHAVIOR_DIAGNOSTICS.md)、[概率排查收束](history/04_INFERENCE_PROBABILITY.md)。
4. 安装/版本查 [H100_COMPATIBILITY](H100_COMPATIBILITY.md)，协作查
   [COLLABORATION](COLLABORATION.md)。旧 runbook 中的 commit、目录、容量和点火命令只属于历史运行。

截至本次整理，共同冷启动与同起点两轮 RL 已完成。当前应一起阅读
[冷启动 v3 报告](2026-09-14_M2_COLDSTART_V3_REPORT.md)和
[共同 SFT 后短 RL 报告](2026-09-14_M2_COMMON_SFT_SHORT_RL_REPORT.md)；两者回答的问题不同，
不能把共同初始化改善归功于 M2 reward。

复核结果时区分完整轨迹、真实角色更新、模型保存和收益；失败不补零，不将单步采样当完整轨迹，
也不将 script/teacher 成交算作模型自主交易。历史实验的具体范围与未完成项保留在总结中。

`runs/` 和大部分 `data/` 是本机证据，不随 GitHub 源码同步；若只取得源码，需另取对应运行归档。
历史模型是否还能恢复，以实际文件清单为准，不能只看旧完成标记。人工时间统一使用 SGT（UTC+8）。
