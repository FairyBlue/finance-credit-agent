# M2共同冷启动：小规模SFT与自主交易验收

制定时间：2026-09-13 14:43 SGT（UTC+8）。用户已同意按公开训练经验推进共同冷启动。运行目录：`runs/m2-coldstart-20260913`。

## 这轮要回答的问题

此前M1/M2每臂4次更新后评估仍全部现金，M2后三批S全零。先验证少量任务示范能否改变4B的自主交易行为、产生有差异的真实账本结果，再决定是否将其作为共同RL起点。这轮是SFT能力实验，不能证明M2的step reward有效。

依据[公开训练经验笔记](2026-09-13_TRADING_AGENT_TRAINING_LESSONS.md)：借鉴Trading-R1的任务冷启动与AlphaQuanter的短任务设计，不照搬其奖励或交易类别比例。[Trading-R1原论文](https://arxiv.org/html/2509.11420v1)；[AlphaQuanter作者训练脚本](https://github.com/horizon-llm/AlphaQuanter/blob/main/verl/recipe/langgraph_agent/stock_trading/run.sh)。公开实现已阅读，未宣称复现其结果。

## 共同示范数据

- 只用原始Train前20窗口（0001、0005……0077），按决策日期去重；每个日期独立构造现金、GS/MSFT/HD各20%已有仓位四种状态。固定前缀实际通过账本执行，与未来价格无关；不串接事后最优动作制造后续训练状态。
- 保留原20股行情、历史features、工具接口及真实next-open执行和成本。仅离线教师候选限于原manifest前三股GS/MSFT/HD，买入目标10%/20%，卖出目标0%/10%，另含hold；方向由当前仓位约束。
- 对同一状态逐个执行候选并hold至三个后续session结束，以真实成本后的终值选择当前动作标签；同收益保留hold。模型输入只含当时可见信息，不含后续价格、候选成绩或事后理由。
- 每个短环境共5个日期：一项固定前缀之后，模型/候选从index1决策至index4。决策样本最多416条，另加最多96条带明确意图的API示范，覆盖buy/sell/hold/abstain/compute/retrieve，总数上限512。超额按事先日期/身份次序裁定，不按收益或动作类别挑样本。
- **教师先验明确存在**：三股候选、目标仓位网格、三session净终值最大化、现金/持仓课程都是人工设计；财富最大化也可能偏向网格最大仓位。它教的是带历史结果标签的决策近似，不是纯JSON训练，也不等于学会最优仓位或获得可部署的先知策略。示范不作为M2 on-policy数据。

## 训练

原Qwen3-4B本地snapshot，non-thinking；现有Torch/Transformers/PEFT，不升级vLLM或依赖。LoRA r16、alpha32，覆盖attention及MLP七种线性投影，BF16、SDPA、gradient checkpointing。固定3 epochs、seed42、microbatch2、累积8、AdamW lr1e-4、无weight decay、梯度裁剪1。

只对completion和EOS计算loss，prompt/padding屏蔽；8192总token上限，不截断。逐步记录loss、梯度、样本身份和实际参数变化；保存adapter、优化器及RNG，训练进程退出后另进程CPU合并为HF模型供vLLM验收。不以训练loss下降代替交易验收，不根据评估延长训练直到成功。

## 固定自主验收

使用Train保留窗口0113/0117/0121/0125，日期与示范输入及标签日期均不重叠；这些窗口彼此仍有重叠，并在此前Train诊断中使用过，不能当正式独立测试集。无Dev/Valid/OOS。

| 范围 | 状态 | 每模型轨迹 |
|---|---|---:|
| 短回合 | 四窗口各现金＋三种已有仓位，从原index4开始决策，至index7终局 | 16状态×(1 greedy＋4 sampled)＝80 |
| 原完整回合 | 四窗口，原21日期、现金起点 | 4窗口×(1 greedy＋4 sampled)＝20 |

原模型与SFT后模型各100条，共200条；相同输入与请求seed方案。所有状态均由模型自主选择动作，**没有API意图、教师标签或手写投资指引**，可使用全部20股与原工具。前缀成交单独记录，不能计作模型成交。

greedy温度0，sample温度1/top_p1/top_k−1；prompt8192、response4096、context12288、并发最多32。每条短回合最多32次请求，完整回合最多128次。触及token/request限制记INCOMPLETE，不补抽。记录所有请求、实际工具执行及完整轨迹。

读数：完整率、非法操作、模型真实买卖及股票覆盖、交易成本、仓位集中/回撤、同状态siblings结果范围、净收益。短回合用接管时资产计算之后收益，前缀成本和预置持仓另外保留；原scorer仅作描述。失败轨迹不伪造零收益，也不只比较成功子集而隐去失败数。买入变多或SFT后收益提高，均不单独证明M2有效。

## 执行与后续边界

顺序：数据和CPU检查→原模型自主评估→SFT→CPU合并→SFT自主评估→汇总。用systemd user独立服务和持久日志运行，网络断开不影响服务；阶段进程依次退出释放GPU。保留旧数据、环境和checkpoint。初始约141GiB可用，LoRA与合并预计新增不到15GiB，先验证磁盘实耗。

若自主合法交易、完整率和结果差异改善，再从同一SFT checkpoint建立未RL基线和M1/M2共同起点，并启用已修复的critic head seed。若仍全hold、频繁反向交易或越界，明确报告失败形态，再针对性修改数据课程；不把教师交易规则加入推理提示，不直接扩成长RL掩盖问题。

后续M1/M2完整checkpoint较大；启动前按实际余量核对空间，不删除旧证据。当前计划先完成上述SFT与自主验收，不自动启动两套大checkpoint RL。

## 后续工作习惯

用户要求：遇到反复卡住的问题，更早查原论文、作者公开代码及官方文档，先比较任务、动作表示、奖励和训练预算，再决定能否借鉴；不以论文收益替代本项目验证。进度与报告使用SGT（UTC+8）。
