# 拒单恢复与动作池化对照（10 月 9 日—14 日）

整理日期：2026-10-18。合并 R5 最终读数、两版反馈诊断、共同恢复 SFT 和新初始化四轮对照；详细原稿见[历史归档](../archive/README.md)，当前安排见[当前状态](../CURRENT_STATUS.md)。`runs/` 完整产物主要在服务器，以下记录均为历史结果。

**共同拒单恢复已在固定验收范围内通过；随后的完整四轮对照没有建立动作池化收益优势。** 两阶段更换了共同 SFT 和观察协议，不能将旧 R5 与新 plain 臂拼接。全部为 Train 内开发证据，正式 Valid / Final OOS 未开启。

## R5：训练完成，greedy 金融覆盖不足

R5 于 10 月 9 日 07:30（UTC+8）结束，两个 seed、12 次 RL 训练均完成四轮：384/384 条训练、actor/critic 各 48 次更新。1,944 次轨迹尝试全部保留，1,915 条完整；评估 sampled 1,248/1,248、greedy 283/312。终态为 `COMPLETE_WITH_EVALUATION_FAILURES`，金融覆盖为 `INCOMPLETE`。

| 第四轮 sampled 平均净收益 | SFT | M1 | M2 η0.125 | M2 η0.25 |
|---|---:|---:|---:|---:|
| Actor LR=1e-7 | 0.5285% | 0.5649% | 0.6243% | 0.6402% |
| Actor LR=2e-7 | 0.5285% | 0.6310% | 0.5932% | 0.5459% |

收益为 12 个 21-session 窗口、两个 seed 的等权开发均值，不是年化收益。低 LR 的 M2 相对同 LR M1 在两 seed 均为正，高 LR 则均为负；最高 M2 均值只比高 LR M1 高 0.0092 个百分点，不能据此宣布最佳方法。各 sampled 条件平均金融 Y 仍为负、净收益低于同窗 benchmark。

29 条 greedy 失败全部耗尽 256 次交互：可见仓位已超过 20%，仍反复 `buy target_weight=0.2`。SFT/M1/M2 都出现此问题；基础提示原本已有总仓位与方向规则，主要缺口是拒单上下文丢失。失败收益保持空缺，没有补抽或当作零收益。

训练报表另发现 `std(ηS)/std(U)` 重复乘 η：32 条 M2 派生比例已版本化勘误，训练张量、收益及金融比较不受影响。校正后旧 M2 η0.25 的四轮比例为 seed42 5.97%–12.25%、seed43 2.84%–6.92%，不能解释为 credit 全零。

## 共同恢复：反馈单独不足，SFT 后固定验收通过

两版恢复观察只补充被拒订单、精确仓位、方向阈值及合法恢复说明；不改变成交、费用、奖励或交易日推进规则，动作仍由模型产生。

| 顺序与固定范围 | 实际结果 | 判读 |
|---|---|---|
| 首轮旧反馈 / v1，480 条短测 | 拒单恢复 122/180 → 171/180；正常均 60/60 | 局部改善，仍有 9 条四步内未恢复 |
| 方向解释 v2，480 条短测 + 240 完整回合 | v2 拒单短测 166/180；正常 60/60；完整回合 238/240 | 增加解释仍未通过，不继续靠失败调整提示 |
| 共同恢复 SFT | 384 条示范：192 恢复 + 192 原正常回放；72 次真实更新 | 24 个日期隔离的新状态留作验收 |
| 同初始化 SFT/M1/M2 最终验收 | 1,200/1,200 条、6,579 次请求；独立 raw、行为及完成回执 PASS | 第一步在本批范围内完成 |

最终 960 条短测均首个请求合法推进；180 条历史拒单恢复全部由模型卖出原股票到 20%，有真实成交和费用。240 条完整回合均到真实终点，仍有 819 次拒单，但全部在第一次后续请求恢复；8 次恢复为模型合法 hold。此结论是消除本批卡死，不是拒单归零或一般收益保证。

四个新 RL 臂从同一新 SFT、统一 v2 观察各做一次 actor/critic 更新，272 条原生轨迹均完整；它们与 SFT 再共同接受最终验收。原批编排失败、反馈失败及未完成轨迹全部保留。账务重放确认同动作在旧版/v1/v2 下逐步和终局一致，拒单不改账本。

## 新初始化四轮：完整结束，池化增量未转正

10 月 12 日启动四臂对照：两档 M1 LR=1e-7/2e-7、M2-plain 和 M2（均 LR=1e-7、η0.25），两个 seed、共同恢复 SFT fresh 初始化、各四轮。Plain 和 M2 使用相同分歧 gate，仅比较动作族池化的增量。

10 月 14 日 02:38 完成 1,336/1,336 条（256 训练 + 1,080 评估），零缺失、零未完成；八个 RL 臂各 actor/critic 四次更新，合计各 32 次。执行与金融覆盖均 `COMPLETE`，八份 actor 归档和临时 moments 退役均完成。

| 第四轮净收益 | seed42 greedy | seed42 sampled | seed43 greedy | seed43 sampled |
|---|---:|---:|---:|---:|
| SFT step0 | 0.7367% | 0.6132% | 0.6940% | 0.9580% |
| M1 LR1e-7 | 0.7621% | 0.9288% | 0.5785% | 0.8042% |
| M1 LR2e-7 | 0.9774% | 0.9751% | 0.8011% | 0.8526% |
| M2-plain | 1.1047% | 0.9115% | 0.6535% | 0.8314% |
| M2 | 0.7261% | 0.8734% | 0.7168% | 0.7384% |

| 第四轮 sampled 配对差（百分点） | seed42 | seed43 |
|---|---:|---:|
| M2 − plain | −0.0381 | −0.0930 |
| M2 − M1 LR1e-7 | −0.0555 | −0.0657 |
| M2 − M1 LR2e-7 | −0.1017 | −0.1142 |

各项均有 12 个共同完整窗口、48 个 sampled 槽，零配对排除。Greedy 的 M2−plain 为 −0.3786/+0.0633 个百分点，方向混合，不与 sampled 合成单一优势。SFT 与 RL 请求种子含不同 policy_step，不是 exact CRN；池化判断以 RL 臂间配对为依据。`performance_pass=false` 表示未建立优势，不是执行失败，也不是普遍方法失效或显著性结论。

## 恢复执行与资源边界

对照过程中保留了 v2 读数身份误报、v3 初始化 CUDA OOM，以及 v4/v5 的 CPU 显存预算等待超时。按哈希核验已完成 cell 后，只补余下任务：212 → 516 → 880 → 1,336 条，未重跑已完成 cell、未自动重试或恢复优化器。隔离预检子进程修复了 controller 意外建立 CUDA 上下文的问题。

76 GiB 是固定阶段预算，80 GiB 是保守软件停止线；实际 H100 NVL 约 93.584 GiB。全部完成 cell 的 ACTIVE 整卡采样峰值最高 70.5166 GiB。旧等待、PID 和 holder 快照不代表当前资源状态。

## 证据与后续

| 事项 | 证据入口 |
|---|---|
| R5 完整率、配对及失败 | [最终读数](../../runs/m2-parameter-grid-20261008-r5/grid_readout.json)、[失败表](../../runs/m2-parameter-grid-20261008-r5/failure_summary.csv) |
| 恢复原始归因与尺度勘误 | [29 条拒单审计](../../runs/m2-order-recovery-audit-20261011/rejection_audit.json)、[勘误](../../runs/m2-order-recovery-audit-20261011/m2_credit_ratio_erratum.json) |
| 两版反馈失败 | [v1 配对](../../runs/m2-order-recovery-20261011-v1/paired_readout.json)、[v2 短测审计](../../runs/m2-order-recovery-20261011-v2/raw_audit.json)、[v2 完整回合审计](../../runs/m2-order-recovery-episodes-20261011-v2/raw_audit.json) |
| 共同 SFT 与最终验收 | [真实训练证据](../../runs/m2-order-recovery-sft-20261011-v1/training_evidence_readout.json)、[1200 条读数](../../runs/m2-order-recovery-acceptance-all-20261011-v2/readout.json)、[raw 审计](../../runs/m2-order-recovery-acceptance-all-20261011-v2/raw_audit.json) |
| 四轮池化最终结果 | [读数](../../runs/m2-action-pooling-comparison-20261013-v6/stage2_readout.json)、[配对 CSV](../../runs/m2-action-pooling-comparison-20261013-v6/stage2_paired_comparisons.csv)、[恢复合同](../../runs/m2-action-pooling-comparison-20261013-v6/recovery_contract.json) |

随后完成的机制分析和两轮 CPU oracle 验证见[功劳机制与 CPU 验证](CREDIT_ORACLE_DIAGNOSTICS.md)。当前应先验证同 prefix 分支取样与合法具体订单菜单，分别处理目标、gate、pool 和状态 baseline；不直接扩展 RL 或 LR/η 网格。
