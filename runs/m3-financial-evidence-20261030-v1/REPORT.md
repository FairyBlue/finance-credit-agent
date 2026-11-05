# 财报证据与条件式委托：执行结果

状态：`FAILED`；当前阶段：`teacher_sft`。
original保留原模型未适配接口的迁移基线；common仅作一次固定64任务协议适配。
完整教师SFT是独立强监督基线，不是三臂RL初始化。oracle_facts与fixed_retrieval均使用同一common适配策略；前者显式提供正确事实。
所有数据均来自已记录真实模型轨迹，缺失结果不是零分。

## 固定终点

| 模型 | 区域 | 状态 | 完整成功/100 | 业务正确 | 指标/期间/单位/触发正确 | 正确任务调用 |
|---|---|---|---:|---:|---|---:|
| original | dev | COMPLETE | 0 | 0 | 12/17/17/2 | NA |
| oracle_facts | dev | COMPLETE | 44 | 48 | 68/98/98/68 | 6.000 |
| fixed_retrieval | dev | COMPLETE | 61 | 61 | 71/100/100/76 | 9.000 |
| common | dev | COMPLETE | 41 | 41 | 49/98/98/67 | 6.000 |
| sft | dev | PENDING | None | NA | NA/NA/NA/NA | NA |
| rloo_seed42 | dev | COMPLETE | 43 | 43 | 48/100/100/70 | 6.000 |
| sham_seed42 | dev | PENDING | None | NA | NA/NA/NA/NA | NA |
| m3_seed42 | dev | PENDING | None | NA | NA/NA/NA/NA | NA |
| rloo_seed43 | dev | PENDING | None | NA | NA/NA/NA/NA | NA |
| sham_seed43 | dev | PENDING | None | NA | NA/NA/NA/NA | NA |
| m3_seed43 | dev | PENDING | None | NA | NA/NA/NA/NA | NA |
| original | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| oracle_facts | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| fixed_retrieval | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| common | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| sft | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| rloo_seed42 | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| sham_seed42 | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| m3_seed42 | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| rloo_seed43 | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| sham_seed43 | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| m3_seed43 | test | NOT_OPEN_BEFORE_FREEZE | None | NA | NA/NA/NA/NA | NA |
| privileged_full_information_oracle | dev | COMPLETE | 100 | 100 | 100/100/100/100 | 6.000 |

## 训练与实际成本

| 臂 | 状态 | 实际token | 父/辅助token | 有效更新 |
|---|---|---:|---:|---:|
| rloo_seed42 | COMPLETE | 2415704 | 2415704/0 | 22 |
| sham_seed42 | PENDING | NA | NA | NA |
| m3_seed42 | PENDING | NA | NA | NA |
| rloo_seed43 | PENDING | NA | NA | NA |
| sham_seed43 | PENDING | NA | NA | NA |
| m3_seed43 | PENDING | NA | NA | NA |

RL实际输入＋输出：2415704；起点采样：1783159；评估新增生成：15903931。SFT监督输入/标签：1605841/22160。
1800万仅为RL生成上限，不是项目总成本。早停后的不同实际消耗不称严格等预算或等FLOPs。
确定性相同权重评估若复用，成本为新增0、来源明确；不是新的独立复验。

## 推进与出口

第三步模式：`PENDING`；第二seed：`PENDING`。
决定：`PENDING`。

业务正确包含财报事实/期间/单位、计算、条件触发和最终方案/合法不可行证明。不能将错误财报判断但保守组合可行算作正确信息决策。引用单列。
未运行的M3不构成M3负面对照；辅助分支的业务差异不是算法优势证据。
原始财报来自公开FinQA，委托规则与账户是模拟；不是真实PIT市场收益或部署证据。

[完整读数](summary/readout.json) · [固定分母表](summary/table.csv) · [配对结果](summary/paired_success.csv) · [执行状态](state.json)
