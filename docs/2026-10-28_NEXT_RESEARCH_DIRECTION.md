# 研究方向演变：从交易credit到证据驱动的条件执行

整理：2026-10-30（UTC+8）。**本文件保留方向演变；现行长期目标见[Research Brief v0.6](../RESEARCH_BRIEF.md)，新轮安排见[三步计划v2](2026-10-30_AGENTIC_RL_THREE_STEP_PLAN_V2.md)，已完成收尾。** 用户已明确以Agent-R1背景下的RL为主线，Brief与具体三步分开维护；实时进度以[CURRENT_STATUS](CURRENT_STATUS.md)为准。

## 这份方向建议后来做到了哪里

| 阶段 | 实际完成与结论 |
|---|---|
| 2026-10-28原建议 | 从未来交易收益转向可验证的委托约束与组合修复，复用工具和账本；先完成可运行任务，再共同SFT和credit对照 |
| 首版合成委托 | SFT开发92%、测试84%；M3无独立增量，[结论](../runs/m3-mandate-repair-20261029-v1/CONCLUSION.md) |
| 组合约束泛化 | SFT开发97%、测试95.5%；普通RL首批零优势后停止，M3未训练，[结论](../runs/m3-mandate-composition-20261029-v1/CONCLUSION.md) |
| 真实财报条件委托 | 完整SFT开发82%、旧保留测试72%；RL未建立稳定收益，M3实际局部项全零，[结论](../runs/m3-financial-evidence-20261030-v2/CONCLUSION.md) |
| 强SFT真实前缀探测 | 检出部分非零业务影响，但2题稳定阳性未过事前3题工程门槛，四臂跳过，不能判定credit无效，[结论](../runs/m3-financial-credit-20261030-v1/CONCLUSION.md) |
| v0.3方向草案 | 曾拟先做推理核验与恢复；该三步未启动，现被独立RL三步替代，[完整草案归档](archive/pre_2026-10-30_agentic_rl_brief_v04.zip) |
| 强SFT后的MC学习对照 | 同起点Dev为SFT82、RLOO80、均匀MC82、候选81；未改变逐题指标正确性，MC覆盖稀疏且原因未定位，[实际回顾](../runs/agentic-rl-mc-20261030-v1/RESEARCH_REVIEW.md) |
| v0.5—v0.6与V2 | 高LR改善训练正控，留出/Dev无净收益；目标MC40/64对普通RL42/64，补充留出采样42/64对SFT43/64。保留RL主线，后续检验训练覆盖与迁移 |

## 当前长期定位

总问题为：**固定训练采样预算下，如何获得更有用的step-level advantage，使多步工具Agent更有效地学习？**

金融证据、核验与恢复作为任务载体；上一轮MC接线与三臂短训已结束，新轮先定位关键步骤的奖励/优势/梯度，再做有限多题批次学习，按证据决定是否继续MC或改任务。不以纯推理、普通RL或继续SFT必须先胜出作为credit研究的绝对前提，也不以信息遮蔽效应替代动作advantage或训练收益。

研究问题与候选贡献以[长期Brief](../RESEARCH_BRIEF.md)为准；具体任务/预算在[docs三步计划v2](2026-10-30_AGENTIC_RL_THREE_STEP_PLAN_V2.md)迭代；执行进展见[CURRENT_STATUS](CURRENT_STATUS.md)。每轮结束回顾长期目标。

## 原文与历史合同

[合并前完整原文包](archive/pre_2026-10-30_research_plan_merge.zip)保存旧RESEARCH_BRIEF.md及本页合并前全部内容，包括2026-10-28原提案、已执行三步及2026-10-30的核验建议。包内保持原仓库路径和原始字节；历史run按各自合同和收据解释，合并不追溯修改实验身份。
