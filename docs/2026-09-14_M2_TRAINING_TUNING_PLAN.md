# M2 共同 critic 设置与信号强度训练计划

制定时间：2026-09-14 13:13 SGT（UTC+8）。此时新增实验尚未启动。

用户已同意先做两件事：检查共同 critic 训练，再比较既有 M2 在较长训练预算及较弱
残差信号下的表现。本轮保持 M2 的 action-family 分组及 `X = U + ηS` 后一次 whitening
公式，不增加推理时人工投资规则，也不扩展 vLLM 数值路径排查。

## 已知问题与要回答的问题

[原两轮报告](2026-09-14_M2_COMMON_SFT_SHORT_RL_REPORT.md)中，M1、M2 第二轮 critic
首 token 预测均值约 11，实际 step return 均值约 2。核对了 Agent-R1 的真实 critic
更新路径：首 token mask 和全批 step 数归一化正常，尚未发现确定的实现 bug。
`critic/values/mean` 的整段 token 均值和 `vpred_mean` 的首 token 均值不能直接比较。

根据保存的轨迹奖励与优势均值/方差重建，第二轮约 97.8% 的 value loss 来自平均偏差项。
但两轮训练窗口不同，旧记录没有同批更新后 value，不能直接断言第一次更新造成 overshoot。
因此先测量同一批状态在更新前后的拟合变化。

## 阶段一：两档共同 critic 学习率

固定 M1，从同一 v3 SFT 独立初始化两个候选，critic LR 分别为 `1e-5` 和 `1e-6`。
actor LR 均为 `1e-6`；各固定两轮、每轮 8 条完整 sampled Train 轨迹、critic 和 actor
各一次更新；末轮在相同四个 Train 窗口做 greedy 评估。另新跑零更新 SFT 基线。

仅增加只读测量：在 critic 更新函数返回前、critic 参数仍在 GPU 时，使用同一批输入
和 returns 做一次无梯度 forward。记录更新前/后首 token 的 MSE、平均偏差、数值范围、
预测变化、相对原值越过 cliprange 的比例、实际学习率及测量耗时。保留并恢复模型模式
与 RNG 状态。各候选均保留正常 actor 更新，不采用未经验证的 critic-only 资源路径。

选择共同设置时优先依据同批拟合和有限数值，而非这四个窗口的投资收益：如果 `1e-6`
减少更新后 MSE/平均偏移且没有新的训练故障，主对照使用 `1e-6`；若没有改善，明确报告，
依据两个预设候选的证据决定保留哪个设置。若两者都有严重数值或执行故障，先停止该阶段，
不把失败训练接到后续对照。实际选择及理由在第二阶段启动前单独落盘、封存。

诊断只保存轨迹、数值日志和完成标记，不保存大模型；产物明确不能用于推理或恢复训练。
这不会删除旧两轮研究 checkpoint 或共同 SFT 模型。

## 阶段二：固定八轮训练曲线

| 条件 | actor/critic 更新次数 | sampled Train 轨迹 | Train greedy 评估 |
|---|---:|---:|---:|
| 共同 SFT 基线 | 0 / 0 | 0 | 4 |
| M1，S=0 | 8 / 8 | 64 | 2、4、6、8 轮各 4 条 |
| M2，η=1 | 8 / 8 | 64 | 2、4、6、8 轮各 4 条 |
| M2，η=0.25 | 8 / 8 | 64 | 2、4、6、8 轮各 4 条 |

三条训练臂分别从原 v3 SFT 重新开始，采用阶段一选定的相同 critic LR、相同 critic seed
及 reference。各自新建 optimizer，不从诊断 checkpoint 或其他臂接续。M1 的显式 η=0
只表明 residual 为零，和历史 M1 的 arm 内清零行为相同。

使用原八个 Train 窗口 `0001/0005/0009/0013/0017/0021/0025/0029`，batch=2、n=4，
两个 epoch 共八轮。第二 epoch 重用原 sample_index，policy_step 继续递增。
greedy 窗口固定 `0113/0117/0121/0125`，均为 21 sessions。
每两轮只读评估不触发提前停在最佳收益、不重抽、不动态改变 η 或学习率。

共同模型为 `runs/m2-coldstart-v3-20260914/merged_model`，merge receipt SHA-256：
`cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db`。
共同条件沿用 `cpu_seed_v1` seed42、non-thinking、`history_features_v1`、
`single_tool_portfolio_v1`、原 scorer/成本/账本、gamma=lambda=1、response cap4096、
prompt8192、pageable Adam、BF16 reference 和已有 vLLM 资源配置。

## 保存、运行与结果解释

新入口与旧固定两轮入口分开，实际源码、模型身份、数据及阶段选择冻结后再启动。
阶段一目录 `runs/m2-critic-lr-diagnostic-20260914`，阶段二目录
`runs/m2-training-tuning-20260914`。均由独立 systemd user service 执行，
`Linger=yes`、`Restart=no`、标准输入为空，日志写服务器文件；客户端网络断开不影响运行。

服务器当前剩余约 56.7 GiB。历史文件保持不动；三条主训练臂只在末轮评估前各导出
一份分片 BF16 actor HF 模型，预计合计约 24.6 GiB。新 profile 明确只供推理，
不含 critic/Adam/RNG，不支持恢复训练。流式逐张量导出不改变训练中的 FP32 参数；
验收仍在原 worker 上进行，导出 BF16 与 worker 的等价性另以实际验证为准。
不为保存而额外复制完整 4B FP32 CPU state dict。

读数同时报告完整率、非法率、实际成交、hold、股票覆盖、仓位集中、成本、含成本收益、
terminal Y、M2 residual 强度、actor KL 及同批 critic 拟合。以每轮同四窗口比较
SFT→M1、SFT→M2、M1→M2；不同训练轮使用不同窗口的训练收益不当作直接学习曲线。
保留零改善、退化或失败结果，八轮结束后再讨论下一步。

此为小规模 Train 调参；共同 SFT 的教师先验和重复使用验收窗口的局限仍然存在。
本轮没有 A1 或正式 OOS 对照，不据此宣称 M2 优于 M1/A1。
