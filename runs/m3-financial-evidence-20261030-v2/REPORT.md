# 财报证据与条件式委托：执行结果

状态：`COMPLETE`；当前阶段：`COMPLETE`。
original保留原模型未适配接口的迁移基线；common仅作一次固定64任务协议适配。
若common标为INHERITED，它继承旧运行的64个Train监督checkpoint，新运行没有重新进行这64题训练；旧RL与旧部分完整SFT均不作为新候选。
完整教师SFT是独立强监督基线，不是三臂RL初始化。oracle_facts与fixed_retrieval均使用同一common适配策略；前者显式提供正确事实。
所有数据均来自已记录真实模型轨迹，缺失结果不是零分。

## 固定终点

| 模型 | 区域 | 状态 | 完整成功/100 | 业务正确 | 指标/期间/单位/触发正确 | 正确任务调用 |
|---|---|---|---:|---:|---|---:|
| original | dev | COMPLETE | 0 | 1 | 10/18/18/4 | NA |
| oracle_facts | dev | COMPLETE | 44 | 46 | 65/98/98/70 | 6.000 |
| fixed_retrieval | dev | COMPLETE | 55 | 55 | 71/100/100/75 | 9.000 |
| common | dev | COMPLETE | 41 | 41 | 48/96/96/67 | 6.000 |
| sft | dev | COMPLETE | 82 | 82 | 82/100/100/92 | 6.000 |
| rloo_seed42 | dev | COMPLETE | 44 | 44 | 50/96/96/64 | 6.000 |
| sham_seed42 | dev | COMPLETE | 41 | 41 | 46/92/92/65 | 6.000 |
| m3_seed42 | dev | COMPLETE | 41 | 41 | 48/96/96/66 | 6.000 |
| rloo_seed43 | dev | SKIPPED | None | NA | NA/NA/NA/NA | NA |
| sham_seed43 | dev | SKIPPED | None | NA | NA/NA/NA/NA | NA |
| m3_seed43 | dev | SKIPPED | None | NA | NA/NA/NA/NA | NA |
| original | test | COMPLETE | 0 | 2 | 14/34/36/12 | NA |
| oracle_facts | test | COMPLETE | 56 | 59 | 81/100/100/74 | 6.036 |
| fixed_retrieval | test | COMPLETE | 42 | 42 | 55/100/100/71 | 9.048 |
| common | test | COMPLETE | 31 | 31 | 40/100/100/62 | 6.000 |
| sft | test | COMPLETE | 72 | 72 | 72/100/100/86 | 6.028 |
| rloo_seed42 | test | COMPLETE | 34 | 34 | 43/100/100/62 | 6.029 |
| sham_seed42 | test | COMPLETE | 31 | 31 | 41/100/100/61 | 6.032 |
| m3_seed42 | test | COMPLETE | 32 | 32 | 40/100/100/60 | 6.000 |
| rloo_seed43 | test | SKIPPED | None | NA | NA/NA/NA/NA | NA |
| sham_seed43 | test | SKIPPED | None | NA | NA/NA/NA/NA | NA |
| m3_seed43 | test | SKIPPED | None | NA | NA/NA/NA/NA | NA |
| privileged_full_information_oracle | dev | COMPLETE | 100 | 100 | 100/100/100/100 | 6.000 |
| privileged_full_information_oracle | test | COMPLETE | 100 | 100 | 100/100/100/100 | 6.000 |

## 训练与实际成本

| 臂 | 状态 | 实际token | 父/辅助token | 有效更新 |
|---|---|---:|---:|---:|
| rloo_seed42 | COMPLETE | 1855171 | 1855171/0 | 16 |
| sham_seed42 | COMPLETE | 1929683 | 1400892/528791 | 12 |
| m3_seed42 | COMPLETE | 1920191 | 1387811/532380 | 12 |
| rloo_seed43 | SKIPPED | NA | NA | NA |
| sham_seed43 | SKIPPED | NA | NA | NA |
| m3_seed43 | SKIPPED | NA | NA | NA |

RL实际输入＋输出：5705045；起点采样：1853555；评估新增生成：47821561。SFT监督输入/标签：20319017/276148。
旧运行已消耗RL生成：2415704；新旧累计RL：8120749。每个新臂统一上限：2500000。1800万是含旧消耗的绝对RL上限，不是项目总成本。旧标签捷径运行的结果不混入本轮效果表；早停不同实际消耗不称严格等预算或等FLOPs。
确定性相同权重评估若复用，成本为新增0、来源明确；不是新的独立复验。

继承checkpoint时，旧标签捷径运行的RL、评估和已完成监督开销在readout的retired_prior_run_costs单列；部分SFT只报告有记录的已完成步数/标签token，不伪造未记录的输入开销，也不当成新端点。

## 推进与出口

第三步模式：`THREE_ARM_COMPARISON`；第二seed：`SKIPPED`，原因：`NO_THRESHOLD_GAIN`。
决定：`KEEP_TEACHER_SFT_DEMO_STOP_CURRENT_CREDIT_EXPANSION`。

业务正确包含财报事实/期间/单位、计算、条件触发和最终方案/合法不可行证明。不能将错误财报判断但保守组合可行算作正确信息决策。引用单列。
未运行的M3不构成M3负面对照；辅助分支的业务差异不是算法优势证据。
原始财报来自公开FinQA，委托规则与账户是模拟；不是真实PIT市场收益或部署证据。

[完整读数](summary/readout.json) · [固定分母表](summary/table.csv) · [配对结果](summary/paired_success.csv) · [执行状态](state.json)
