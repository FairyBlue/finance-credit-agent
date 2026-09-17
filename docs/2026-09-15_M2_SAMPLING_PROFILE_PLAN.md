# M2：当前采样与 Qwen 官方 non-thinking 采样对照

制定时间：**2026-09-15 20:00 SGT（UTC+8）**。此时本轮 GPU 对照尚未启动。

用户同意先检查解码设置。使用已有四个模型，仅改变评估时的采样参数；保持共同 SFT、
M1 与两种 M2 的权重不变，比较收益、完整率、拒单恢复和交易行为。

## 固定两组设置

| profile | temperature | top_p | top_k | min_p |
|---|---:|---:|---:|---:|
| current：当前采样 | 1.0 | 1.0 | -1 | 0.0 |
| qwen_recommended：官方 non-thinking 采样 | 0.7 | 0.8 | 20 | 0.0 |

官方来源：[Qwen3-4B 模型卡 Best Practices](https://huggingface.co/Qwen/Qwen3-4B#best-practices)，
2026-09-15 核对。这是三个参数共同改变的采样配置对照，不能把结果单独归因于 temperature。
推荐针对原 Qwen 模型；它在本项目 SFT/RL 后模型上的效果需要实际检验。

两组共同 `enable_thinking=False`、repetition_penalty=1、presence_penalty=0、frequency_penalty=0、
每请求 n=1、prompt cap8192、response cap4096、每轨迹128次请求上限。
本批不另加重复惩罚或增加输出长度。只检验上述官方采样配置，不宣称复现官方全部推理建议。

## 模型、窗口和预算

顺序为共同 v3 未 RL SFT、M1 第8轮、M2 η=1 第8轮、M2 η=0.25 第8轮。
模型身份直接沿用[上一轮交叉诊断](2026-09-14_M2_POLICY_TRANSFER_EVAL_REPORT.md)的现存导出。
每模型在原12个窗口上，两组各4条sampled siblings，共 **96条**；四模型总计 **384条**。

- 8个训练窗：`formal-train-0001/0005/0009/0013/0017/0021/0025/0029`。
- 4个Train验收窗：`formal-train-0113/0117/0121/0125`。
- 全部21 sessions、20次市场推进，从100万美元现金、空仓开始，无持仓前缀或新增教师标签。
- 上下文、scorer、成本、calibration、history_features_v1 和 single_tool_portfolio_v1 沿用原封存输入。

每模型只加载一次相同 BF16 standalone vLLM；同一context、同一sibling的两个profile交错提交，
共同seed42，identity为step1、sample_index=context序号、rollout_n=0..3、validate=false。
逐请求种子由同一identity和request_index生成，两组及四模型之间一致。
模型动作导致后续状态分叉是对照的一部分，不把整个轨迹当作相同中间状态的逐步配对。

当前采样组在本批重新执行；主要比较不混入昨天的sampled数值。
旧greedy仅作历史诊断参考，本批没有新增greedy，也不推断官方采样能修复greedy本身。

## 读数与结束条件

每模型/profile分别对8个训练窗、4个验收窗汇总；sampled先对同窗4条等权平均，再对固定窗等权平均。
比较官方−当前、以及各profile内M1/M2−SFT的收益、Y、成本、末期现金、非法率和完整率。
同时保留完整轨迹、动作/成交股票、拒单错误类型和重复行为；避免只以买入次数选择设置。

任何原定成员不完整，所在窗口及固定组的完整收益均值保持缺失，不补零、不缩小分母。
经核对的 REQUEST_CAP 保留为未完成行为并继续其他原定模型；其他执行错误、响应/提示截断
或身份不一致停止后续模型，保留证据。无自动重试、延长预算或选择性重抽。

新目录 `runs/m2-sampling-profile-eval-20260915`；入口独立放在 scripts/diagnostics，
不修改原训练、原交叉评估及底层推理库。源、输入和模型身份在启动前固定。
独立 systemd user service、Linger=yes、Restart=no，客户端断线不影响运行；实时读数由独立监视器更新。
不保存新大模型、不删除历史文件。本批结束后写报告，再决定后续共同解码/训练条件；不自动启动新 RL。

启动前资源补记（20:18 SGT）：H100出现容器外进程占用约15 GiB；本轮保留原
`gpu_memory_utilization=0.5`，每模型启动前要求至少60 GiB空闲并记录实际显存和利用率。
允许余量充足时共用GPU；不足则最多等待120秒后停止，不终止其他进程。

这仍是Train内、单seed、滚动窗口重叠的小规模诊断，不是正式OOS验证。
