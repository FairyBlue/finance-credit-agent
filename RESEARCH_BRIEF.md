# Research Brief（v0.2，待两人签字冻结）

> 状态：DRAFT。冻结前需要：潘昊 ✍️ ＋ PhD 合作者 ✍️。冻结后本文件只允许追加，不允许修改。

## 研究问题（冻结后不再改）

**在工具调用型 trading agent 中，finance-native step-level credit assignment 能否相比 (a) outcome-only GRPO 和 (b) 使用相同 terminal net return 的 terminal-outcome StepPO control，提高训练稳定性、样本效率与样本外风险调整收益？**

## 核心假设

金融任务的结果信号具有延迟、高噪声、非平稳和风险敏感等特点。通用 StepPO 可以通过 critic 与
step-level GAE 学习 temporal credit，但其方法定义不包含 portfolio state、market regime 或反事实
信息贡献等金融归纳偏置，因此不能先验地区分决策质量与市场噪声。本文关心一条轨迹中的五类步骤：

1. 获取了无效信息的步骤
2. 产生了有效分析的步骤
3. 形成了正确交易方向的步骤
4. 控制了仓位与风险的步骤
5. 因市场噪声而偶然盈亏的步骤

## 方法

### 第一层：finance-native 终局信号（第一版实现）

对一条完整 episode，令 $P_0$ 为初始组合价值，$P_T$ 为最终 session close 按共享账务引擎得到的
组合价值，则

$$
R_{net}=\frac{P_T}{P_0}-1.
$$

$R_{net}$ 已包含冻结的 next-open 成交、commission 和 slippage，但不包含 benchmark adjustment、
volatility normalization 或显式风险罚项。finance-native 终局标量定义为：

$$
A^{fin}
=
\frac{R_{net}-B}{\sigma+\epsilon}
-\lambda_{risk}\cdot\text{RiskPenalty},
$$

其中 $B$ 是同一 episode、同一成本/next-open 账务口径的冻结等权 buy-and-hold benchmark return，
$\sigma$ 是 agent 组合净逐期收益的 population standard deviation（`ddof=0`）。第一版风险项为：

$$
\text{RiskPenalty}
=
\max\!\left(
D_{max},
\max(0,\tau_{max}-1.0),
\max(0,w_{max}-0.30)
\right),
$$

其中 $D_{max}$、$\tau_{max}$、$w_{max}$ 分别是最大回撤、最大单日双边换手和最大单票 close weight；
$\epsilon=0.0006887293802216064$，$\lambda_{risk}=0.9016432557217152$。尽管沿用符号 $A^{fin}$，
它在本文中是完整轨迹结束后计算的 **terminal score/signal**，不是 critic value、return 或直接交给
actor 的 optimizer advantage。

### 第二层：步级训练信号与 advantage 机制 —— ⚠️ 这是本文的真正研究问题

把 $A^{fin}$ 转换为逐步训练信号的机制是开放问题。这里既包含 reward-timeline controls，也包含
直接作用于 GAE 后、whitening 前 advantage 的结构化 residual；二者不得混称为 reward。**如果只做
均匀回填，本工作仍只是 reward shaping，不构成新的 credit-assignment 贡献。**正式设计依次包含：

- **M0 finance-terminal signal control**：不做显式 reward redistribution。对含 $T$ 个实际
  Agent-R1 interaction steps 的完整轨迹，使用

  $$
  r_t^{M0}=
  \begin{cases}
  0, & t<T,\\
  A^{fin}, & t=T.
  \end{cases}
  $$

  M0 与 A2 使用完全相同的 StepPO stack，仅把 terminal scalar 从 $R_{net}$ 换成 $A^{fin}$；它用于
  诊断 finance-native terminal objective 的增量，不是 reward allocation 方法，也不能作为 A3 主方法。

- **M1 uniform immediate-step reward v1**：M1 是 action-agnostic、轨迹总量守恒的 reward
  redistribution 对照。除终局 $A^{fin}$ 和实际轨迹长度外，它不使用 action、portfolio state、
  market regime、步骤质量或反事实证据，具体语义如下：
  - **时序边界与步单位**：$A^{fin}$ 只在 `finance_trajectory` 完整终止后得到；M1 随后回溯式
    重写这条已生成 rollout 的训练 reward timeline。`immediate_step_reward_v1` 中的
    `immediate` 是指这些值作为 StepPO 的逐步 reward 输入，不表示 agent 执行动作时获得在线即时
    反馈，也不会改变生成当前这条 rollout 的行为。$T$ 精确定义为实际保留的 Agent-R1
    interaction/environment steps 数，包括已经实现的 retrieve、compute、buy、sell、hold、abstain 或
    invalid step；它不是 market sessions、token 数、推理文本长度或配置中的最大步数。
  - **分配与守恒**：先把终局信号转换为有限 Python float $v$，令 $q=v/T$；前 $T-1$ 步写入
    $r_t^{M1}=q$，最后一步写入 $r_T^{M1}=v-\sum_{t=1}^{T-1}r_t^{M1}$。最后一步只吸收浮点
    余差，因而不保证所有 step reward bitwise 完全相同，但保证证据层 Python step-credit 向量的
    普通求和精确等于 $v$。正、负信号保持符号，零保持为零；M1 不做 clipping、重新标定或
    action-specific weighting。轨迹越长，每步 reward 的绝对值通常越小，但分配总量不会随 $T$
    膨胀。
  - **输入域与 fail-closed 边界**：M1 核心允许任意有限的正、零、负信号，拒绝 NaN、$\pm\infty$
    和空 steps；A3 flow 另外拒绝截断轨迹、错误终止位置、非 `finance_trajectory` 以及环境 trace 与
    Agent-R1 reward 不一致。$A^{fin}=0$ 在方法定义上合法；wiring smoke 要求 Train/Valid 各出现
    非零信号只是为了证明接线，不是 M1 的算法限制。
  - **进入优化器后的语义**：上述向量只是 reward，不是 return 或直接写给 actor 的 advantage。
    共享 StepPO 管线仍按 trajectory/step 顺序，由 critic value、trajectory-end bootstrap 边界、
    $\gamma$ 和 $\lambda$
    计算 step-level GAE，再进行 step-level whitening 并映射到 response mask 后执行 actor 更新。
    因此实际 optimizer advantages 通常不均匀，也不要求其总和、discounted return 或 float32 trainer
    metric 继续等于 $A^{fin}$；GAE 产生非均匀 advantage 也不会让 M1 获得 action-specific 或因果
    step attribution。
  - **具体例子**：若 $A^{fin}=0.9$、$T=3$，M1 的 immediate rewards 为
    $[0.3,0.3,0.3]$。在仅用于验证语义的合成例子中，若 value 全为 0 且
    $\gamma=\lambda=1$，GAE returns 为 $[0.9,0.6,0.3]$，标准化 advantages 为
    $[1,0,-1]$。这只说明“即时奖励均匀”不等于“optimizer advantage 均匀”，不是正式实验
    超参数或性能结果。
  - **科学角色**：M1 只回答“把 finance-native 终局信号均分到实际 steps，再配合 StepPO 后能达到
    什么表现”，本质上仍是 reward-timing/reward-shaping control。它与 M0 的差异仅为 terminal
    signal 的时间分布；它与 A3-M2 的差异才是待冻结的结构化 residual。M1 本身不能作为 A3 主方法，
    目前通过的也只是 wiring smoke，不是正式消融结果。
- **M2 state-regime anchor residual（当前优先主方法）**：M2 保留 M1 的 $A^{fin}$、immediate
  rewards、critic target、GAE 和 StepPO actor/optimizer，只在同一 policy snapshot、同一 episode/
  session 的 sibling rollouts 中，按 action 前可见的 portfolio state 与 no-future-information market
  regime features 构造粗化 anchor。满足待冻结 preregistration 的 group support 与 action-contrast
  条件时，计算 mean-centered terminal-outcome residual $S_t$，并在唯一一次 step-level whitening
  前组合：

  $$
  X_t=U_t+\eta S_t,\qquad A_t^{M2}=\operatorname{Whiten}(X_t),
  $$

  其中 $U_t$ 是 M1 reward 经 critic/GAE 得到的未白化 advantage；$S_t$ 不进入 critic target，且
  stop-gradient。M2 估计的是 coarse-state conditional relative credit，不是已经识别的单步因果效应。
  anchor、$\eta$、coverage gate、whitening scope 和统计规则必须以单独 preregistration 为准。
- **M3 counterfactual information residual（候选，不自动启用）**：针对 retrieve/compute 等信息步骤，
  在待冻结的 intervention 下遮蔽指定信息并重新生成后续决策，用 decision change 与 terminal outcome
  构造 sampled counterfactual residual。正式采用前必须另行冻结 eligible steps、遮蔽边界、common-
  random-number 规则、距离/质量函数、符号与尺度、多步骤重叠归因、coverage 和预算。由于 M3 会增加
  辅助 rollout，必须另设同 auxiliary schedule、完整资源记账但不向优化器提供有效反事实信号的
  sham control，并在共同 total-token horizon 下比较、另报 GPU-seconds sensitivity。该设计不假定
  real/sham 的实际 token 或 FLOP 天然完全相同；没有这一 control 时，不得把 `M3 − M1` 解释为
  单一机制效应。M3 路线必须同时通过 `M3 − sham` 的 mechanism-specific gate 和既有
  `M3 − M1` superiority gate；前者不能替代后者。

## 三类方法、五个核心正式配置

### 共同记号、step 与完成边界

对 episode $p$ 的第 $i$ 条 rollout，$T_{pi}$ 是实际保留的 Agent-R1 interaction/environment steps
数；每次模型 response 与一次环境交互构成一个 step，包括 retrieve、compute、buy、sell、hold、
abstain 和 invalid action。它不是 market sessions、token 数或 `max_agent_steps`。

所有正式配置只接受到达同一 finance episode terminal 的完整轨迹。达到 step/context 上限仍未终止
时统一 fail closed：不得在某个 arm 静默记为零、在另一个 arm 拒绝，也不得按性能选择是否 retry。

### A1：Outcome-only GRPO

A1 的环境 reward timeline 为：

$$
r_{pit}^{A1}=
\begin{cases}
0, & t<T_{pi},\\
R_{net,pi}, & t=T_{pi}.
\end{cases}
$$

对同一 sampled input-row/prompt occurrence（共享同一 `uid`/group index）和同一 policy snapshot 的
$K=4$ 条 sibling rollouts，GRPO 先把每条完整轨迹的 step/token rewards 求和为 $R_{net,pi}$，再计算
group-relative trajectory advantage：

$$
\widehat A_{pi}^{A1}
=
\frac{R_{net,pi}-\operatorname{mean}_{j\in G_p}R_{net,pj}}
{\operatorname{std}_{j\in G_p}R_{net,pj}+\epsilon_{GRPO}}.
$$

正式 A1 固定 `norm_adv_by_std_in_grpo=true`、$\epsilon_{GRPO}=10^{-6}$，分母使用四条 trajectory
scores 的 sample standard deviation（correction $=1$）。

同一标量随后广播到该轨迹全部 steps 的有效 response tokens。因此 terminal-only 不表示只更新最后
一步；A1 会更新整条轨迹，但没有 critic、没有 GAE，也没有轨迹内部的 time/action-specific credit。
若 sibling rollouts 的 $R_{net}$ 相同，group-relative advantage 会归零。A1 不使用 $B$、$\sigma$、
RiskPenalty 或 $A^{fin}$。

### A2：Terminal-outcome StepPO control

A2 与 A1 使用完全相同的外部 reward 和时间线：

$$
r_{pit}^{A2}=
\begin{cases}
0, & t<T_{pi},\\
R_{net,pi}, & t=T_{pi},
\end{cases}
$$

但把 GRPO stack 换成 learned pre-action critic、step-level GAE、一次 step-level whitening 和 GSPO
actor loss。给定拟由两位合作者共同冻结的 $\gamma,\lambda$：

$$
\delta_{pit}=r_{pit}^{A2}+\gamma V_{pi,t+1}-V_{pit},\qquad
U_{pit}=\delta_{pit}+\gamma\lambda U_{pi,t+1},
$$

终局边界为 $V_{pi,T_{pi}+1}=U_{pi,T_{pi}+1}=0$。未白化的 $U_{pit}$ 经共享 scope 的一次 whitening 后，
每个 step 的 scalar advantage 广播到该 step 的有效 response tokens。

A2 没有 reward model，也不使用 finance-specific intermediate shaping；“通用”只描述 critic/GAE 的
temporal propagation。它可以产生 step-varying advantage，但不保证如此：例如 $V_t=0$ 且
$\gamma=\lambda=1$ 时，terminal-only reward 的未白化 $U_t$ 对所有步骤都等于 $R_{net}$。因此 A2
不能被表述为已经判断了每一步的因果贡献。

### A3：Finance-native controls 与主方法

| 配置 | External training reward | Post-GAE、pre-whitening 增量 | 优化与科学角色 |
|---|---|---|---|
| A1 Outcome-only GRPO | terminal-only $R_{net}$ | 无 | trajectory-relative GRPO + critic disabled + upstream vanilla token-level importance-ratio/clipped PPO actor loss；outcome-only reference |
| A2 Terminal-outcome StepPO | 与 A1 相同的 terminal-only $R_{net}$ | 无 | critic + GAE + GSPO；通用 StepPO method-package control |
| A3-M0 Finance-terminal control | terminal-only $A^{fin}$ | 无 | 与 A2 相同的 StepPO；隔离 terminal signal transformation |
| A3-M1 Uniform redistribution control | 总量守恒的 $A^{fin}/T$ immediate rewards | 无 | 与 M0/M2 相同的 StepPO；reward-timing control 和 M2 matched null |
| A3-M2 Main | 与 M1 相同的 base reward | 待冻结 M2 preregistration 定义的 $\eta S_t$ | finance-native main；唯一 primary 为 `M2 − M1` |

M3 只有在独立 preregistration 与 schedule-matched、resource-accounted sham control 完成后，才能
替代 M2 成为 A3-main；此时正式矩阵会增加 sham arm，不能仍声称是无条件的五配置设计。

截至 2026-09-03，仓库现有 A2 仍实现 dense market-transition reward：净组合单期收益减 benchmark
收益、回撤罚项和换手罚项，并把 `last_reward` 写入下一 observation。它应视为历史实现和可选的
`A2-Dense` supplementary baseline，**不等于上述正式 A2**。此外，launcher 仍把 StepPO arms 的
实际 `rollout.n` 强制为 1，$\gamma,\lambda$ 尚未由项目显式写入 formal config，A1 的未完成轨迹仍
可能静默形成全零总分，A3-M0/M2 尚未实现，M2 preregistration 也未冻结；formal configs 还缺
`training.episodes=441` 与 exact Qwen3-4B revision。正式运行前必须完成 A2 terminal-only 重接线、
A3-M0/M2 实现、共同完成性与 reward-neutral contract、配置/schema/测试及各自 wiring smoke；不得
只改标签后沿用旧行为。

### 相邻对比、可识别内容与声明边界

| 对比 | 可识别的 method-package change | 不能声称 |
|---|---|---|
| `A2 − A1` | 相同 terminal $R_{net}$ 下，完整 StepPO stack 相对 GRPO stack 的增量 | 单独归因于 critic、GAE 或 GSPO 任一组件 |
| `A3-M0 − A2` | 相同 StepPO 与 terminal-only placement 下，$R_{net}\rightarrow A^{fin}$ 的 terminal-objective effect | reward redistribution 或 step attribution |
| `A3-M1 − A3-M0` | 相同 $A^{fin}$/StepPO 下，uniform redistribution 的 reward-timing effect | action-specific 或因果 credit |
| `A3-M2 − A3-M1` | 相同 M1 backbone 下，待冻结 M2 residual 的增量 | $A^{fin}$ 或一般 StepPO 的效果 |
| `A3-main − A1/A2/A3-M0` | 完整系统的实用差异 | 任一设计层的单独因果贡献 |

这是相邻对照阶梯，不是完整 factorial design。RL 训练中的交互与路径依赖意味着上述 effect 不能机械
相加为“总贡献”。除 `A3-M2 − A3-M1` 的待冻结 primary gate 外，其余相邻对比默认是 secondary
descriptive contrasts；若要做正式 superiority claim，必须在查看 Valid 前冻结检验层级与 multiplicity。

### 所有正式配置的共同比较契约

- 每个 episode/policy snapshot 统一生成 `rollout_n=4` 条 sibling trajectories；A1 用于 GRPO group，
  M2 用于 anchor grouping，其余 arms 仍保持相同采样与资源口径。
- A2、A3-M0、A3-M1、A3-M2 使用独立参数但相同 architecture/revision/initialization rule 的 critic，
  并拟在共同审核后把 $\gamma=1,\lambda=1$ 显式写入配置，共享 trajectory-end bootstrap boundary、
  GSPO actor loss、optimizer 和 update schedule。未白化 advantage/residual 必须在 PPO minibatch
  切分前，对完整 logical on-policy rollout batch 中 `sample_mask=true` 的真实 step rows 做唯一一次
  DP-global whitening，使用 sample standard deviation（correction $=1$）与 $10^{-8}$ epsilon。
- Train、Valid 与 Final OOS 都采用 reward-neutral observation contract：训练 reward、arm identity 和
  credit evidence 只进入 trainer/审计日志，不写入 main-rollout 的后续 observation 或 prompt。条件性
  M3 auxiliary branches 只允许出现 preregistration 明定的 real-vs-sham intervention 差异；两侧的
  pre-intervention base context 必须相同，auxiliary prompt/result/RNG 不得回流或推进 main rollout。
- 五臂共享模型 revision、Train/Valid manifests、episode/update 顺序、sampling 参数、seed set、
  checkpoint/evaluation grid 与 retry/completion 规则。所有 arms 的 Valid/Final checkpoint 都由同一个
  frozen scorer 计算 $A^{fin}$，训练 reward 的差异不得改变评测输入或纵轴定义。
- 相同 one-epoch trajectory schedule 不等于相同实际 compute。主资源横轴使用 Train generated response
  tokens；environment transitions、prompt tokens、optimizer updates、GPU-seconds 和峰值显存必须同时
  报告。新增 M0 不得削减 M2/M1 的 seed、trajectory 或统计预算。

模型：Qwen3-4B（调试用 Qwen3-1.7B）。框架：Agent-R1（verl 0.7.0）。Proposal seeds 至少 2 个；
正式主张所需第三 seed 与统计规则必须在 formal rollout 前冻结。

StepPO 的论文/方法语义固定引用 [arXiv:2604.18401v4](https://arxiv.org/abs/2604.18401v4)（2026-06-21）；可执行实现仍独立固定为 Agent-R1 commit `b124aa46534cbf2fb8bc8af11405774984c42ac7`。paper pin 不替代 executable commit pin。

## Go / No-Go（预注册，出结果前签字）

**A3 方法贡献的必要门槛（非充分条件）**：当前优先的 A3-M2 必须在与 A3-M1 相同的模型、数据、
StepPO stack、seed set、超参数选择规则和预注册等预算口径下，通过预注册的 `M2 − M1` superiority
gate。该 gate 的主指标、聚合方式、方向、阈值/不确定性规则及预算单位必须在 formal rollout 前由
两位合作者线下冻结。未通过时，即使优于 A1/A2/M0，也只能报告 $A^{fin}$ / reward-shaping 或完整
系统差异，不得声称 M2 finance-native credit assignment 有增益。M3 若被选择，必须由其独立协议把
schedule-matched、resource-accounted sham 定义为主机制对照，并重新冻结相应 gate；不能自动沿用
M2 的五臂授权，也不能取消既有 `M3 − M1` superiority gate。

**GO（走方法论文）**：A3-M2 必须通过待冻结的 `M2 − M1` fixed-design Valid primary gate，并通过
完整性、coverage、资源和 raw-risk guardrails。A1/A2/M0 的 secondary effect sizes、跨 seed 稳定性及
获准后的单次 Final OOS 方向只作为支持或外部确认，不能替代 primary gate，也不能在 primary 失败后
把 M2 改判为方法贡献。M3 使用其独立 preregistration 中的 `M3 − sham` mechanism gate，并且仍须
通过 `M3 − M1` superiority gate。

**PIVOT（转诊断论文，接 The Losing Winner）**：matched-control primary gate 未通过，或关键完整性/
风险 guardrail 失败。此时论文改为“细粒度 credit 为什么救不了 trading RLVR”——量化预测能力与
盈利能力的断裂点归因。**“能解释断裂”不是 GO 条件，而是 PIVOT 的入口**，防止赛后改标准。

## 与最近邻的差异（一句话版）

> AlphaQuanter learns tool orchestration with trajectory-level optimization; we study finance-specific **step-level** credit assignment under delayed, noisy, risk-sensitive rewards, with a contamination-controlled evaluation protocol.

完整对比见 [PAPER_MATRIX.md](PAPER_MATRIX.md)。

## Phase 2（写给博后看的 roadmap，本阶段不实现）

> After validating finance-native credit assignment in historical replay, we extend the method to hierarchical planning and microstructure execution, building on the TradeMaster / FinWorld ecosystem (EarnHFT / MacroHFT line).
