# Research Brief（v0.1，待两人签字冻结）

> 状态：DRAFT。冻结前需要：潘昊 ✍️ ＋ PhD 合作者 ✍️。冻结后本文件只允许追加，不允许修改。

## 研究问题（冻结后不再改）

**在工具调用型 trading agent 中，finance-native step-level credit assignment 能否相比 (a) outcome-only GRPO 和 (b) 通用 step-level credit（StepPO v4），提高训练稳定性、样本效率与样本外风险调整收益？**

## 核心假设

金融任务的奖励是延迟、高噪声、非平稳、风险敏感的。因此通用 step credit（为 ALFWorld/WebShop 这类可验证低噪声奖励设计）无法区分一条轨迹中的五类步骤：

1. 获取了无效信息的步骤
2. 产生了有效分析的步骤
3. 形成了正确交易方向的步骤
4. 控制了仓位与风险的步骤
5. 因市场噪声而偶然盈亏的步骤

## 方法

### 第一层：finance-native 轨迹信号（占位公式，第一版实现）

$$A^{fin} = \frac{R - B}{\sigma + \epsilon} - \lambda \cdot \text{RiskPenalty}$$

R = episode 收益，B = 同期 benchmark 收益，σ = 同期波动率，RiskPenalty 第一版取 max(最大回撤, 换手超限, 仓位超限)。

### 第二层：步级分配机制 —— ⚠️ 这是本文的真正研究问题，不是实现细节

把 $A^{fin}$ 分配到步骤的机制是开放问题。**如果只做均匀广播，本工作退化为 reward shaping，不构成 credit assignment 贡献**（审稿必杀点，也是与 AlphaQuanter 差异化的根基）。候选机制（按实现难度排序，proposal 至少做出 M2 或 M3 之一）：

- **M1 uniform immediate-step reward v1**：M1 是 action-agnostic、轨迹总量守恒的 reward
  redistribution 对照。除终局 $A^{fin}$ 和实际轨迹长度外，它不使用 action、portfolio state、
  market regime、步骤质量或反事实证据，具体语义如下：
  - **时序边界与步单位**：$A^{fin}$ 只在 `finance_trajectory` 完整终止后得到；M1 随后回溯式
    重写这条已生成 rollout 的训练 reward timeline。`immediate_step_reward_v1` 中的
    `immediate` 是指这些值作为 StepPO 的逐步 reward 输入，不表示 agent 执行动作时获得在线即时
    反馈，也不会改变生成当前这条 rollout 的行为。$T$ 精确定义为实际保留的 Agent-R1
    interaction/environment steps 数，包括已经实现的 retrieve、compute、hold、trade、abstain 或
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
    共享 StepPO 管线仍按 trajectory/step 顺序，由 critic value、终止 mask、$\gamma$ 和 $\lambda$
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
    什么表现”，本质上仍是 reward-shaping control。正式比较时，A3-M1 必须与 A3-main 使用相同的
    $A^{fin}$、模型、数据、StepPO 优化器、seed set、超参数选择规则和预注册预算。只有 A3-main
    按预注册 gate 胜过这一对照，才可能把增益归因于 M2/M3 的非均匀 finance-native credit
    allocation；即使如此，该 gate 也只是必要条件。M1 本身不能作为 A3 主方法，也不能单独支持
    credit-assignment 贡献；目前通过的只是 wiring smoke，不是正式 A3-M1 消融结果。
- **M2 状态分组步级优势**：借 Agent-R1 的 step-MDP，按（portfolio 状态 × 市场 regime）做 GiGPO 式锚点分组，在组内计算相对步级优势——同一市况同一仓位下，不同步骤选择的差异才归因于步骤本身。
- **M3 反事实信息贡献**：对信息获取步骤（retrieve/compute）估计反事实贡献——遮蔽该步骤的返回结果重新 rollout 决策，决策变化幅度 × 结果质量 = 该步 credit。成本高，可采样估计。

## 三类方法、四个正式配置

| 配置 | 优化 | 信号与角色 |
|---|---|---|
| A1 Outcome-only | GRPO | 原始 episode 收益 |
| A2 Generic step credit | Agent-R1 StepPO | 通用逐步奖励 |
| A3-M1 Formal ablation | 与 A3 主方法相同的 StepPO | $A^{fin}$ ＋ M1 uniform immediate-step reward v1；必需正式消融，不具备主方法资格 |
| A3-M2 或 A3-M3 Main | 与 A2/A3-M1 相同的 StepPO | $A^{fin}$ ＋预注册的 M2 或 M3；A3 主方法 |

模型：Qwen3-4B（调试用 Qwen3-1.7B）。框架：Agent-R1（verl 0.7.0）。Seeds：2（proposal 阶段），最终版补第 3 个。

StepPO 的论文/方法语义固定引用 [arXiv:2604.18401v4](https://arxiv.org/abs/2604.18401v4)（2026-06-21）；可执行实现仍独立固定为 Agent-R1 commit `b124aa46534cbf2fb8bc8af11405774984c42ac7`。paper pin 不替代 executable commit pin。

## Go / No-Go（预注册，出结果前签字）

**A3 方法贡献的必要门槛（非充分条件）**：A3-M2/M3 必须在与 A3-M1 相同的模型、数据、StepPO 优化器、seed set、超参数选择规则和预注册等预算口径下，通过预注册的 M1 superiority gate。该 gate 的 superiority statistic（主指标、聚合方式、方向、阈值/不确定性规则及预算计量单位）必须在正式训练前由两位合作者线下冻结；当前不得自行选择数值。未通过该 gate 时，即使优于 A1/A2，也只能报告 $A^{fin}$ / reward-shaping 结果，不得声称 finance-native credit assignment 有增益。

**GO（走方法论文），满足其一即可：**
1. 在通过 M1 superiority gate 的前提下，相同预注册预算下，A3-M2/M3 的 OOS 指标学习曲线显著快于 A1/A2（前 200–300 步即可判定，不要求收敛）
2. 跨 seed 训练稳定性更好（崩溃率 / 曲线方差）
3. OOS Sharpe、最大回撤或 benchmark-adjusted return 更优

**PIVOT（转诊断论文，接 The Losing Winner）：** 1–3 全不成立。此时论文改为"细粒度 credit 为什么救不了 trading RLVR"——量化预测能力与盈利能力的断裂点归因。**注意：'能解释断裂'不是 GO 条件，是 PIVOT 的入口**，防止赛后改标准。

## 与最近邻的差异（一句话版）

> AlphaQuanter learns tool orchestration with trajectory-level optimization; we study finance-specific **step-level** credit assignment under delayed, noisy, risk-sensitive rewards, with a contamination-controlled evaluation protocol.

完整对比见 [PAPER_MATRIX.md](PAPER_MATRIX.md)。

## Phase 2（写给博后看的 roadmap，本阶段不实现）

> After validating finance-native credit assignment in historical replay, we extend the method to hierarchical planning and microstructure execution, building on the TradeMaster / FinWorld ecosystem (EarnHFT / MacroHFT line).
