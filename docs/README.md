# 文档导航

**2026-09-17 GitHub 备份入口：**先读[当前交接](2026-09-17_PROJECT_BACKUP_HANDOFF.md)，
再看[随仓库保存的小型结果快照](results/2026-09-17-backup/README.md)。模型与完整运行目录仍在服务器。

整理日期：2026-09-14。日常阅读从本页进入；9 月 13、14 日的 12 份文档保持原文。
更早的日志已汇总为四个阶段，完整旧文本收在一个[压缩归档](archive/README.md)，无需逐份翻阅。

## 先看这两份

- [共同冷启动 v3 报告](2026-09-14_M2_COLDSTART_V3_REPORT.md)：共同初始化的合法性、完整性和仍存在的教师先验。
- [共同 SFT 后两轮 M1/M2 报告](2026-09-14_M2_COMMON_SFT_SHORT_RL_REPORT.md)：真实训练、同起点比较及收益结果。

截至整理时，两阶段都已完成。冷启动通过不等于 M2 有收益优势；后者在本次短 RL 中尚未体现。
实时状态与具体运行位置见 [M2_SCREENING_LATEST](../runs/M2_SCREENING_LATEST.md)。

后续[共同 critic 设置与八轮信号强度对照](2026-09-14_M2_TRAINING_TUNING_PLAN.md)也已于 18:16 SGT 完成，
主比较 244/244 轨迹完整。较低 critic LR 改善诊断拟合，但三臂第八轮收益均低于共同 SFT；
两种 M2 也低于 M1，详见[完整报告与训练曲线](2026-09-14_M2_TRAINING_TUNING_REPORT.md)。

[日期与解码方式交叉诊断](2026-09-14_M2_POLICY_TRANSFER_EVAL_PLAN.md)已于21:07 SGT结束：
四模型240条均已观察，228完整、12条训练greedy拒单循环。M2训练日期sampled提高，另一组日期仍低于SFT；
[最终报告](2026-09-14_M2_POLICY_TRANSFER_EVAL_REPORT.md)记录收益、合法性和下一步训练方向。
原SFT失败及[事后继续剩余模型的修正](2026-09-14_M2_POLICY_TRANSFER_EVAL_CONTINUATION.md)保留原样。

[两组 non-thinking 采样对照](2026-09-15_M2_SAMPLING_PROFILE_PLAN.md)已于9月15日21:03 SGT结束：
384条全部观察，369完整、15条推荐参数下的拒单循环。当前采样192/192完整，推荐177/192；
收益改善不一致，建议保留当前共同配置，详见[最终报告](2026-09-15_M2_SAMPLING_PROFILE_REPORT.md)。

[扩大 Train 日期覆盖与较低 actor LR 对照](2026-09-15_M2_TRAIN_COVERAGE_PLAN.md)
已于 **9 月 16 日 18:55 SGT 完成**：308/308 条完整，共同 SFT 起点的 M1/M2 η=0.25 各 8 轮。
最终 M2 greedy 净收益 1.823533%，高于 M1 的 1.565643% 与 SFT 的 1.618652%；
sampled 仅略高于 M1，仍低于 SFT，尚无稳定跨解码模式的收益优势。
前两次诊断兼容性失败的原始记录保留，最终结果仅来自成功 v2；详见
[完整报告](2026-09-15_M2_TRAIN_COVERAGE_REPORT.md)和
[最终读数](../runs/m2-train-coverage-20260916-v2/FINAL_RESULTS.md)。

9 月 17 日已完成[持仓损益与真实训练信号诊断](2026-09-17_M2_TRAIN_COVERAGE_DIAGNOSTIC_REPORT.md)：
后期 sampled 差额主要来自实际持仓路径，M2 没有直接给非法步骤额外 credit；
采样不确定性仍大。下一项是[共同拒单罚四轮小试候选](2026-09-17_M2_COMMON_VALIDITY_PENALTY_PLAN.md)，
现已实现、测试并封存，于 9 月 17 日 15:22 SGT 启动独立 GPU 排队和监视服务，尚未开始训练。
固定 164 条、M1/M2 各四轮；不同时调整 M2 公式、学习率和窗口归一化。
详见[执行记录](2026-09-17_M2_COMMON_VALIDITY_PENALTY_REPORT.md)与
[实时进度](../runs/m2-common-validity-penalty-20260917/LIVE_STATUS.md)。

[9 月 17 日磁盘盘点与清理](2026-09-17_DISK_CLEANUP_AUDIT.md)：用户确认后已删除冷启动前旧实验的
六份 optimizer，释放约 171.24 GiB，可用约 186.44 GiB；模型权重及原始实验记录保留。

## 近期文档原样保留

| 日期 / 工作 | 计划 | 结果或材料 |
|---|---|---|
| 9 月 13 日：Train 实用迭代 | [9 月 12 日原计划](2026-09-12_M2_TRAIN_PRACTICE_PLAN.md) | [报告](2026-09-13_M2_TRAIN_PRACTICE_REPORT.md) |
| 9 月 13 日：回到模型学习 | [学习重启计划](2026-09-13_M2_LEARNING_RESTART_PLAN.md) | [报告](2026-09-13_M2_LEARNING_RESTART_REPORT.md) |
| 9 月 13 日：外部训练经验 | — | [训练经验与资料](2026-09-13_TRADING_AGENT_TRAINING_LESSONS.md) |
| 9 月 13 日：初次冷启动 | [计划](2026-09-13_M2_COLDSTART_PLAN.md) | [报告](2026-09-13_M2_COLDSTART_REPORT.md) |
| 9 月 13 日：冷启动 v2 | [计划](2026-09-13_M2_COLDSTART_V2_PLAN.md) | [报告](2026-09-13_M2_COLDSTART_V2_REPORT.md) |
| 9 月 14 日：冷启动 v3 | [计划](2026-09-14_M2_COLDSTART_V3_PLAN.md) | [报告](2026-09-14_M2_COLDSTART_V3_REPORT.md) |
| 9 月 14 日：共同 SFT 后短 RL | [计划](2026-09-14_M2_COMMON_SFT_SHORT_RL_PLAN.md) | [报告](2026-09-14_M2_COMMON_SFT_SHORT_RL_REPORT.md) |

## 历史阶段总结

| 阶段 | 包含内容 | 核心结论 |
|---|---|---|
| [一：H100 与 M1](history/01_H100_AND_M1.md) | 8 月 27 日—9 月 2 日的环境、prompt、G1、数据与 M1 接线 | 工程链路通过，不是正式收益结果 |
| [二：screening 工程](history/02_M2_SCREENING_ENGINEERING.md) | R1—R5、内存/遥测、thinking 截断、non-thinking、存储 | R5 的 76 条工程轨迹完整，但当时没有交易及非零 residual 覆盖 |
| [三：行为诊断](history/03_M2_BEHAVIOR_DIAGNOSTICS.md) | schema、历史/记忆、工具循环、4B 理解与交易转移、1024 单步采样 | 接口问题有修复，额外说明未带来稳定交易，转向共同初始化 |
| [四：概率排查收束](history/04_INFERENCE_PROBABILITY.md) | logprob、BF16/FP32、真实 KV decode、未完成配对与归因更正 | 原始生成概率读取正确；精确配对仍未完成，结束底层扩展排查 |

## 保留的设计与协议

这些文件定义研究问题或执行变更，保留其主体，不与流水日志一起删去。

- [RESEARCH_BRIEF](../RESEARCH_BRIEF.md)、[EVAL_PROTOCOL](../EVAL_PROTOCOL.md)、[M2 预注册](../M2_V1_PREREGISTRATION.md)。
- [M2 实验计划 v2](2026-09-05_M2_EXPERIMENT_PLAN_V2.md)；
  [合作者 ladder review](2026-09-04_M2_M3_LADDER_REVIEW.md)、[合作者实验计划](claude/2026-09-04_EXPERIMENT_PLAN.md)。
- 执行修订：[R1](2026-09-06_M2_SCREENING_EXECUTION_AMENDMENT_R1.md)、
  [R2](2026-09-07_M2_SCREENING_EXECUTION_AMENDMENT_R2.md)、
  [R3](2026-09-07_M2_SCREENING_EXECUTION_AMENDMENT_R3.md)、
  [R4](2026-09-07_M2_SCREENING_EXECUTION_AMENDMENT_R4.md)、
  [R5](2026-09-08_M2_SCREENING_EXECUTION_AMENDMENT_R5.md)。
- 数据与基础验收：[formal manifest](2026-09-01_FORMAL_MANIFEST_RESULTS.md)、
  [T2 v2](2026-09-01_T2_V2_RESULTS.md)、[G1 精简结果](2026-08-31_H100_G1_SMOKE_RESULTS.md)、
  [M1 精简结果](2026-09-02_H100_A3_M1_WIRING_SMOKE_RESULTS.md)。

## 安装、协作与追溯

[接手指南](EXPERIMENT_HANDOFF.md) · [H100 安装与版本](H100_COMPATIBILITY.md) ·
[协作方式](COLLABORATION.md) · [旧文件到新总结的映射与原文归档](archive/README.md)。

历史记录中的“当前/下一步”只对当时有效。新工作优先引用本页的近期计划/报告；
具体运行成功与否以相应实际证据为准。`runs/` 链接指向本机资料，不代表已随源码上传。
