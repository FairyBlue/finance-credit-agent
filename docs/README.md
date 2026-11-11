# 文档入口

更新：2026-11-11。**先读[当前状态](CURRENT_STATUS.md)**；长期方向读[Research Brief v0.8](../RESEARCH_BRIEF.md)。[V6直接训练已结束](../runs/agentic-rl-action-q-20261111-v6/CONCLUSION.md)：42/64，对SFT43、V5 44；188/192辅助后继回报不变，停止当前配置、保留原SFT。按用户最新决定继续金融探索，尚未启动新实验。

用户已确认**继续优先探索金融场景**；原Brief保留。[Research Brief 2.0](../RESEARCH_BRIEF2.0.md)单独保存RAGEN上的未来研究目标与候选贡献，尚未启动，不替代当前金融方向。

## 按问题读，一类只保留一份总结

| 想了解什么 | 阅读文档 | 已合并的内容 |
|---|---|---|
| 环境、数据和早期诊断为什么这样定 | [基础与诊断](history/FOUNDATIONS_AND_DIAGNOSTICS.md) | H100 / G1 / M1、screening R1—R5、行为与概率诊断 |
| 共同初始化及早期训练有什么结果 | [共同初始化与 4B 学习](history/INITIALIZATION_AND_4B_LEARNING.md) | 冷启动 v1—v3、学习重启、两轮/八轮 RL、日期与采样诊断、低显存、两 seed 四轮及旧网格准备；相关旧计划和续跑说明一并归档 |
| 拒单恢复后，池化是否带来收益 | [恢复与动作池化](history/RECOVERY_AND_ACTION_POOLING.md) | 参数网格 R5、两版反馈、共同恢复 SFT / 验收、四轮 plain / M1 / M2 对照及失败过程 |
| 功劳分配机制还有什么问题 | [功劳机制与条件采样](history/CREDIT_ORACLE_DIAGNOSTICS.md) | 存档分析、合成oracle、多状态回放、同前缀采样、gate隔离、随机后续、当前π固定收缩及独立种子复验 |

这些主题文档直接包含结果、边界和证据入口，不需要按日期逐份翻计划。精确原稿、旧方案与历史哈希统一查[归档说明](archive/README.md)，当前导航不留跳转壳。

## 接手、安装与协作

长期研究目标：[Research Brief v0.8](../RESEARCH_BRIEF.md)，只维护研究问题、候选贡献、证据与重要方向调整；滚动三步另存`docs/`，每轮结束回顾Brief。旧交易实验边界：[评测协议](../EVAL_PROTOCOL.md) · [M2历史规格](../M2_V1_PREREGISTRATION.md)。安装复现：[SETUP](../SETUP.md) · [H100版本合同](H100_COMPATIBILITY.md)。协作方式：[COLLABORATION](COLLABORATION.md)。

接手时先读当前状态，再按需要选择主题；具体执行核对该次收据、冻结源码和输入，不从旧容量或旧命令推定当前资源。`runs/`、行情和模型主要在服务器，不随源码完整同步。[9/17](results/2026-09-17-backup/README.md)和 [10/04](results/2026-10-04-backup/README.md)备份只代表当时快照。

公开代码与最近结果的文件清单见[11月11日快照](results/2026-11-11-public-update/README.md)。

## 最新结果与计划

[V6动作Q直接训练](2026-11-11_AGENTIC_RL_ACTION_Q_TRAINING.md)：00:47完整结束，六批、192父、24次更新、64条评估齐全。42/64低于SFT43与V5 44；仅16个目标raw优势真正变化，额外同动作续写边际信息很少。新增806.98万token、GPU阶段24.2分钟，当前配置收尾。[结论与下一项建议](../runs/agentic-rl-action-q-20261111-v6/CONCLUSION.md)。

[V5固定Dev100检查](../runs/agentic-rl-v5-dev-check-20261110/CONCLUSION.md)：80/100对SFT82/100，不替换底座；无训练，新增255.9万token、GPU阶段3.3分钟。随后旧[Q估值pilot草案](2026-11-10_AGENTIC_RL_ACTION_Q_EXPERIMENT.md)被用户要求的V6直接训练替代。

[V5保留解析失败反馈](2026-11-10_AGENTIC_RL_INVALID_FEEDBACK_V5.md)：19:22完整结束，44/64、无效输出0、未提交0；新增577.3万token、GPU阶段13.0分钟。保留为工程修正候选，不将对SFT多1条当作方法收益。[本轮方向回顾](../runs/agentic-rl-invalid-feedback-20261110-v5/CONCLUSION.md)。

[V4仅计算响应更新](2026-11-10_AGENTIC_RL_STEP_FOCUS_V4.md)：16:55完整结束，6批/24次更新后41/64，未超过全步骤宽覆盖42/64或SFT43/64。实际586.7万token、GPU阶段13.4分钟，停止当前配置。坏JSON响应被mask排除是后续候选机制，未启动新实验。[结果与回顾](../runs/agentic-rl-step-focus-20261110-v4/CONCLUSION.md)。

[Agentic RL三步v3精简版](2026-11-05_AGENTIC_RL_THREE_STEP_PLAN_V3.md)：**11月5日21:08完整结束**。24报告、192父轨迹、24次实际更新，宽覆盖42/64未超过旧窄覆盖42/64或SFT43/64。实际664.3万token、GPU阶段32.3分钟；按平坦出口收尾，不追加复验。[结果与回顾](../runs/agentic-rl-coverage-20261105-v3/CONCLUSION.md)。

[Agentic RL三步v2](2026-10-30_AGENTIC_RL_THREE_STEP_PLAN_V2.md)：已完整收尾。高LR改善训练正控但未带来报告留出/Dev净收益；目标阶段MC14个baseline实际进入更新，40/64未超过普通RL42/64。[补充留出采样](2026-11-05_HELDOUT_SAMPLING_PROBE.md)亦完整完成，无正向频率差异；合计4005.4万生成token、GPU阶段约2.91小时。后续聚焦训练覆盖与迁移，旧模型和收据保留。

[Agentic RL已结束三步](2026-10-30_AGENTIC_RL_THREE_STEP_PLAN.md)：真实MC baseline和状态恢复通过；普通RL/均匀MC/候选分配MC均未超过完整SFT；无增量出口收尾，新Test未评估。实际新增生成1517.9万，未补采追门槛。[完整读数](../runs/agentic-rl-mc-20261030-v1/REPORT.md) · [结果与Brief回顾](../runs/agentic-rl-mc-20261030-v1/RESEARCH_REVIEW.md)。

[方向演变](2026-10-28_NEXT_RESEARCH_DIRECTION.md)保留从交易到条件委托及RL主线调整的说明；[v0.3与未执行推理三步原文](archive/pre_2026-10-30_agentic_rl_brief_v04.zip)、[更早两份合并前原文](archive/pre_2026-10-30_research_plan_merge.zip)供回顾，不是当前待办。

[强SFT局部信用探测结论](../runs/m3-financial-credit-20261030-v1/CONCLUSION.md)：父128条全部完成；12个真实前缀各3条件×2续写，原返回/遮蔽/格式对照业务18/24、12/24、18/24，仅2题两次一致损失，预设需3。四臂跳过，不能写credit训练失败；[真实对照与恢复例子](../runs/m3-financial-credit-20261030-v1/examples/README.md)、[成本与状态](../runs/m3-financial-credit-20261030-v1/REPORT.md)已保留。

[真实财报条件委托结论](../runs/m3-financial-evidence-20261030-v2/CONCLUSION.md)：共同起点开发41%、新Test31%，教师SFT82%/72%；普通RLOO44%/34%、sham41%/31%、M3 41%/32%，按Dev门槛跳过第二seed。[完整结果与成本](../runs/m3-financial-evidence-20261030-v2/REPORT.md) · [同一财报双阈值的真实执行展示](../runs/m3-financial-evidence-20261030-v2/examples/paired_demo.md)。

[组合约束新一轮结论](../runs/m3-mandate-composition-20261029-v1/CONCLUSION.md)：新SFT在新任务开发194/200、保留测试191/200；普通RLOO首批零优势后提前结束，M3本轮未训练。[完整结果](../runs/m3-mandate-composition-20261029-v1/REPORT.md)与[真实同题前后对照](../runs/m3-mandate-composition-20261029-v1/examples/README.md)已交付。

[首版委托修复结论](../runs/m3-mandate-repair-20261029-v1/CONCLUSION.md)：共同SFT开发92%、保留测试84%，当前M3配置无独立增量；[完整报告](../runs/m3-mandate-repair-20261029-v1/REPORT.md)与[真实工具轨迹示例](../runs/m3-mandate-repair-20261029-v1/examples/README.md)已交付。

[已完成三步计划](2026-10-29_M3_COMPOSITION_PLAN.md)：改任务组合与信息选择、补齐共同SFT、按普通RL出口统一评估并收尾。首版[已完成计划](2026-10-28_NEXT_RESEARCH_DIRECTION.md)保留供回顾。

旧交易结果见[demo简报](2026-10-28_DEMO_SUMMARY.md)与[宽覆盖主实验报告](../runs/m2-broad-rloo-order-main-20261028-v1/FINAL_REPORT.md)；10月22—29日过程的整理前状态页见[原文快照](archive/pre_2026-10-29_current_status.zip)。

## 历史机制结果与封存输入

以下为当时的诊断结果与取样合同，后续训练/收益验证已完成；原文中的“下一步”不再作为当前待办。

[10/19 独立种子复验结果](history/CREDIT_ORACLE_DIAGNOSTICS.md#独立种子复验同窗口的收缩信号方向复现) · [取样前方案](2026-10-19_M2_ONPOLICY_CREDIT_REPLICATION_PLAN.md) · [314分支实际收据](../runs/m2-onpolicy-credit-replication-20261019-v1/README.md)。α=.25/.5对zero的六个预设方向均复现；仅同窗新seed，不称日期泛化或收益，下一步固定同前缀训练及同α plain对照。

[10/18 当前π状态验证结果](history/CREDIT_ORACLE_DIAGNOSTICS.md#当前π状态固定收缩出现有限参考改善) · [预先固定方案](2026-10-18_M2_ONPOLICY_CREDIT_PLAN.md) · [8父/16状态/300分支收据](../runs/m2-onpolicy-credit-20261018-v1/README.md)。全部完成，预设α=.25/.5在有限参考下低于zero，尚需独立复核，不选最佳α或扩大RL。

[10/18 独立估计/参考结果](history/CREDIT_ORACLE_DIAGNOSTICS.md#独立估计与参考细分pool未优于零credit) · [执行前方案](2026-10-18_M2_CREDIT_COMPARISON_PLAN.md) · [864路径实际收据](../runs/m2-stochastic-credit-comparison-20261018-v1/README.md)。全部完成，zero附加credit误差最低；暂不升级gate/pool或扩大RL。

[10/18 随机后续结果](history/CREDIT_ORACLE_DIAGNOSTICS.md#随机策略后续pilot154路径全部完成) · [执行方案](2026-10-18_M2_STOCHASTIC_CONTINUATION_PLAN.md) · [154路径实际收据](../runs/m2-stochastic-continuation-20261018-v3/README.md)。GPU优先、CPU并行的约定见[协作说明](COLLABORATION.md#8-gpu与cpu工作顺序)。

[10/18 gate隔离结果](history/CREDIT_ORACLE_DIAGNOSTICS.md#gate单因素验证候选实现与信号覆盖通过) · [执行前方案](2026-10-18_M2_SAME_PREFIX_GATE_PLAN.md) · [实际收据与复现](../runs/m2-same-prefix-gate-20261018-v1/README.md)。仅单因素实现与覆盖，生产未接入；上述随机后续pilot随后完成。

[10/18 原生首请求结果](history/CREDIT_ORACLE_DIAGNOSTICS.md#原生同输入采样重复支持存在但原gate漏掉差异) · [采样执行前方案](2026-10-18_M2_SAME_PREFIX_SAMPLING_PLAN.md) · [采样收据与复现](../runs/m2-same-prefix-sampling-20261018-v1/README.md) · [10/18 同前缀菜单结果](history/CREDIT_ORACLE_DIAGNOSTICS.md#同前缀菜单验证真实输入与分支账务通过) · [菜单执行前方案](2026-10-18_M2_SAME_PREFIX_MENU_PLAN.md) · [菜单输入与复现](../runs/m2-same-prefix-menu-20261018-v1/README.md) · [10/17 上轮完整报告](2026-10-17_M2_STATE_ORDER_ORACLE_REPORT.md)。新增结果并入机制主题；运行合同所需原文仍保留原路径。

[10/11 恢复方案](2026-10-11_M2_ORDER_RECOVERY_LEARNING_PLAN.md) · [10/12 池化方案](2026-10-12_M2_ACTION_POOLING_COMPARISON_PLAN.md) · [10/14 CPU 方案](2026-10-14_M2_CREDIT_ORACLE_PROBE_PLAN.md) · [10/17 多状态方案](2026-10-17_M2_STATE_ORDER_ORACLE_PLAN.md)。

上述四份封存文件被现有运行合同按绝对路径和 SHA 校验，保留原字节供工具使用；日常阅读使用上面的主题总结。封存正文里的旧状态及原相对链接属于当时记录，原链接目标可在归档 ZIP 中按原目录找到。更早的诊断 / 训练工具需在隔离历史 checkout 恢复相应原文后复现，不将最新源码与旧合同拼成同一执行身份。
