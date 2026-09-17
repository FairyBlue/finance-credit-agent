# M2：扩大Train日期覆盖与降低共同actor学习率

制定时间：**2026-09-15 22:34 SGT（UTC+8）**。本轮GPU训练尚未启动。

用户同意按[采样对照报告的下一步](2026-09-15_M2_SAMPLING_PROFILE_REPORT.md)继续。
本轮保留当前non-thinking采样与M2η0.25，改进两臂共同的学习条件，并直接检查非法动作的更新信号。

## 固定对照

| 组别 | 初始化 | actor / critic LR | 更新次数 | 训练轨迹 | 验收轨迹 |
|---|---|---|---:|---:|---:|
| 未RL共同SFT | v3 SFT | 无更新 | 0 | 0 | 20 |
| M1 | 同一v3 SFT | 2e-7 / 1e-6 | 8 / 8 | 64 | 80 |
| M2η0.25 | 同一v3 SFT | 2e-7 / 1e-6 | 8 / 8 | 64 | 80 |

总计 **128条训练+180条验收=308条轨迹**。各臂新optimizer，从同一actor/reference/critic backbone
和共同`cpu_seed_v1` seed42开始，核对真实初始critic head；不接着第八轮旧actor继续训练。
共同SFT模型仍为`runs/m2-coldstart-v3-20260914/merged_model`，merge receipt
`cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db`。

actor LR从此前1e-6降低五倍；critic保持此前已完成同批拟合诊断所选1e-6。
M2仍使用原action-family分组的`U+ηS`后统一whitening，η=0.25；M1 residual为0，critic returns不加S。
算法、scorer、成本、账本、历史特征及工具协议不变，不增加推理时人工投资或仓位规则。

## Train日期覆盖

原方案用8个窗口重复两遍；本轮用16个不同窗口各一次，保持每臂8轮、每轮2窗×4siblings的预算。
从原441个formal Train滚动窗口按交易日期筛选：

- 避开当前4个Train验收窗口（113/117/121/125，2023-06-14—08-01）；
- 避开冷启动fresh验收日期2023-05-23—06-13；
- 保留旧12个screen-dev窗口的日期角色（2023-11-01—2024-10-31），不转作本轮训练。

合计121个合法起点，在按日期排序后的序列等距取索引0、8、…、120；再将前8个与后8个交错配对。
选择不读取收益、奖励或教师交易标签。16窗合计**154个不同交易session**；每轮两窗日期不重叠，
不同轮之间允许滚动重叠，因此不把64条轨迹当作64个独立市场样本。

| 更新轮次 | 窗口一（日期） | 窗口二（日期） |
|---|---|---|
| 1 | formal-train-0001（2023-01-03—2023-02-01） | formal-train-0065（2023-04-05—2023-05-04） |
| 2 | formal-train-0009（2023-01-13—2023-02-13） | formal-train-0073（2023-04-18—2023-05-16） |
| 3 | formal-train-0017（2023-01-26—2023-02-24） | formal-train-0149（2023-08-07—2023-09-05） |
| 4 | formal-train-0025（2023-02-07—2023-03-08） | formal-train-0157（2023-08-17—2023-09-15） |
| 5 | formal-train-0033（2023-02-17—2023-03-20） | formal-train-0165（2023-08-29—2023-09-27） |
| 6 | formal-train-0041（2023-03-02—2023-03-30） | formal-train-0173（2023-09-11—2023-10-09） |
| 7 | formal-train-0049（2023-03-14—2023-04-12） | formal-train-0181（2023-09-21—2023-10-19） |
| 8 | formal-train-0057（2023-03-24—2023-04-24） | formal-train-0189（2023-10-03—2023-10-31） |


新parquet只构造以上formal Train行；固定dev4直接沿用原parquet。
原reward calibration沿用，不因新增训练日期重新拟合scorer。没有构造或运行formal Valid/Test回合。
这仍是Train内探索，旧screen-dev也不被重新宣称为历史上从未接触过的holdout。

**学习率与日期覆盖是同时改变的共同训练条件。** 本轮先检验这套条件下是否更容易学出收益，
不能把相对旧实验的变化单独归因于学习率或新增日期。

## 同一worker的sampled与greedy验收

SFT在第0轮验收；两RL臂在第2、4、6、8轮验收，每次固定上述4个Train验收窗。
每窗5条：lane0 greedy，lane1—4 sampled；同一次原生trainer validation生成，分别汇总。
trainer自身可能打印5条混合的val均值，本轮主读数只使用按lane拆开的轨迹。

- 训练与sampled验收：temperature1、top_p1、top_k−1、min_p0；
- greedy验收：temperature0，其余相同；
- 全部`enable_thinking=False`、prompt cap8192、response cap4096，沿用训练worker的agent cap256。

这次不混入旧standalone/旧worker收益。每条仍使用原seed42与实际policy_step/sample_index/rollout_n/request_index身份；
M1/M2同轮同窗身份一致。SFT的step0与RL的step2—8种子身份不同，不声称与SFT逐请求完全配对。
比较单位是同一窗口的轨迹均值。sampled先对4个siblings等权平均，再对固定4窗等权平均。

任何轨迹不完整保留原样，对应固定窗口/组的完整收益保持缺失；不补零、不缩小分母或重抽。
保留原strict训练流程：遇到生成cap、未到真实终末、非有限数或基础设施失败即停止当前臂及后续自动执行，
不接旧checkpoint重启，不挑最好轮次提前停止。若发生工程故障，再依据现场证据处理。

## 实际actor更新诊断

两臂每次更新记录**原PPO loss真实收到的advantage、old anchor和pre-update logp**，
包括所有合法/非法action的有效response tokens，排除padding；保留trajectory、step和token身份。
原optimizer更新后，对同批固定输入与既有response做一次无梯度forward，记录post-update logp及变化。
它是条件token评分变化，不是实际生成某交易动作的概率，也不是收益因果证明。

仅增加观测，不修改loss、reward、optimizer、KL或采样；额外forward恢复CPU/CUDA RNG和原模型模式，
共同dropout为0。当前支持范围固定为单GPU、单minibatch、一个PPO epoch、microbatch1。
CPU等价测试核对开关诊断后的参数更新与原指标/RNG一致，再进入真实GPU运行。
原critic同批pre/post拟合探针继续保留。

核心读数为：非法动作实际A的正负、固定response更新后logp漂移、reward与A的区别，
以及这些量与合法率/完整率/收益的共同变化。不用“即时Y/N为正”直接断言错误动作被强化。

## 资源、保存与完成

运行目录`runs/m2-train-coverage-20260915`，新执行身份，启动前固定源码、输入和计划。
用独立systemd user服务与监视器，Linger=yes、Restart=no、stdin为空，客户端断线不影响。
旧文件和模型全部保留；两RL臂仅各保存最终第8轮BF16 actor（约8.23GiB/臂），
中间轮保留轨迹、指标和更新诊断。导出不含critic/Adam/RNG，不支持恢复训练。
当前空闲约32GiB；启动前预留剩余导出各9GiB加8GiB运行余量，足够本轮，无需删除或新增完整checkpoint。

最终按SFT→M1、SFT→M2、M1→M2分别比较两种验收方式的曲线、净收益、Y、成本、
非法率、现金/持仓、股票覆盖与更新信号。两个新因素的效应暂不单独拆分，不扩展采样网格或vLLM排查。
这轮不足以代替A1和正式OOS验证；固定预算结束后写报告，再决定下一步。
