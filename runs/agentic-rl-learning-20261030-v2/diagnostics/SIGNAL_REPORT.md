# 第一步：已有记录的实际训练信号

状态：COMPLETE。仅CPU读取历史Train记录，新增生成0；未读取Test或市场OOS。

| 臂 | 任务实例/父迹 | 有效更新 | 零方差轮数 | 反传响应token | MC替换/请求 |
|---|---:|---:|---:|---:|---:|
| rloo_seed42 | 23/92 | 20 | 0/6 | 24773 | 0/565 |
| mc_uniform_seed42 | 18/72 | 13 | 5/9 | 17901 | 12/445 |
| mc_priority_seed42 | 18/72 | 13 | 5/9 | 17853 | 13/444 |

## 动作阶段的实际反传

| 臂 | 阶段 | 请求 | 原始非零优势 | 白化后非零优势 | 实际反传请求 | MC替换 |
|---|---|---:|---:|---:|---:|---:|
| rloo_seed42 | constraint_execution | 89 | 41 | 49 | 69 | 0 |
| rloo_seed42 | invalid_or_missing | 1 | 1 | 1 | 1 | 0 |
| rloo_seed42 | metric_calculation | 110 | 62 | 70 | 90 | 0 |
| rloo_seed42 | read_financial_report | 92 | 44 | 52 | 72 | 0 |
| rloo_seed42 | read_other_evidence | 92 | 44 | 52 | 72 | 0 |
| rloo_seed42 | search | 92 | 44 | 52 | 72 | 0 |
| rloo_seed42 | submission | 89 | 41 | 49 | 69 | 0 |
| mc_uniform_seed42 | constraint_execution | 69 | 33 | 49 | 49 | 1 |
| mc_uniform_seed42 | invalid_or_missing | 1 | 1 | 1 | 1 | 0 |
| mc_uniform_seed42 | metric_calculation | 90 | 54 | 70 | 70 | 2 |
| mc_uniform_seed42 | read_financial_report | 72 | 36 | 52 | 52 | 2 |
| mc_uniform_seed42 | read_other_evidence | 72 | 35 | 52 | 52 | 1 |
| mc_uniform_seed42 | search | 72 | 35 | 52 | 52 | 5 |
| mc_uniform_seed42 | submission | 69 | 33 | 49 | 49 | 1 |
| mc_priority_seed42 | constraint_execution | 69 | 29 | 49 | 49 | 1 |
| mc_priority_seed42 | metric_calculation | 90 | 51 | 70 | 70 | 5 |
| mc_priority_seed42 | read_financial_report | 72 | 32 | 52 | 52 | 1 |
| mc_priority_seed42 | read_other_evidence | 72 | 32 | 52 | 52 | 2 |
| mc_priority_seed42 | search | 72 | 31 | 52 | 52 | 2 |
| mc_priority_seed42 | submission | 69 | 29 | 49 | 49 | 2 |

## 可确认与仍缺证据

- 有效更新确实发生，存在非零梯度和改变的权重；不能称未训练。数百万输入＋输出采样token不等于实际反传量。
- 两MC臂均有5轮全批优势方差为零，无有效更新。MC覆盖、回报差异和相对RLOO优势变化见JSON逐点记录。
- 阶段请求获得非零优势不等于其取数选择得到正确因果信用；白化还可能让原始零优势请求变成非零，须区分两者。
- 梯度范数来自混合请求的optimizer minibatch，不能归因于单个计算动作。平均ratio不是KL；旧日志没有逐请求更新后概率。
- 已知终点未修好任何SFT失败题，但尚不能区分支持稀疏、有效学习量、关键动作更新不足和估计目标价值。
- 第二步用独立重采的混合题正控、多题minibatch和两档学习率验证响应；不把Train正控当总体泛化。

[逐批与逐点JSON](signal_readout.json) · [冻结任务与教师过滤](../data/manifest.json)
