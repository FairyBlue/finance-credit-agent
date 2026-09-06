# M2 v1 MVP Specification（精简讨论稿）

> 新版实验安排讨论稿见 [M2 实验计划 v2（2026-09-05）](docs/2026-09-05_M2_EXPERIMENT_PLAN_V2.md)。
> 新稿含 scorer 候选、same-gate plain 对照与新的抽样/判读提案，尚未冻结；本文保留原 v1 定义供对照。

> **状态：算法筛选规格，不是最终 preregistration。**
>
> 本文件刻意替换此前的论文级详细草案。当前目标只有一个：尽快判断 M2 是否比 A3-M1 和 A2
> 更有用。正式统计、完整审计与 Final OOS 规则等算法有正信号后再补。

## 0. 本版决定

M2 v1 采用 **action-conditioned sibling residual**，并只作用于每个 market session 唯一的推进决策：

```text
buy / sell / hold / abstain
```

同一 episode、同一 policy rollout batch 的 4 条 siblings，在同一 decision session 天然各有一条
推进决策。因此 v1 不再使用 portfolio、cash、drawdown、prefix finance state、pending order、
credit regime、evidence hash 或 tool depth 做 exact matching，也取消 80% coverage hard gate。

准确名称是：

> **M2-lite: session-aligned sibling action-family credit**

Methods 三句话版：M1 先产生未白化的 $U_t=\operatorname{StepPO\text{-}GAE}(r_t^{M1})$；同一
episode/policy/session 的四条 siblings 按推进 action family 计算
$S_t=\bar Y_{g,a_t}-\bar Y_g$；actor 最终使用
$A_t=\operatorname{Whiten}(U_t+S_t)$，而 critic 保持 M1 不变。

它不是 state×regime estimator，不是 causal/counterfactual estimator，也不覆盖 information-step
credit；`retrieve/compute` 的反事实贡献留给 M3。

## 1. 为什么采纳 action-conditioned 形式

严格说，旧 residual $Y_i-\bar Y_g$ 最终会乘在 $\nabla\log\pi(a_i)$ 上，所以 optimizer 并非完全
“看不到 action”。但新定义更符合我们要表达的语义：同一 sibling group 中，选择同一 environment
action family 的 rollouts 应共享该 action 的平均结果，而不是因后续随机路径获得不同的当前-step
credit。

新旧定义的关系很简单：

- 某个 action 只出现一次时，新旧 residual 相同；
- 多条 rollout 选择同一 action 时，新版先平均它们的 outcome，减少同-action 内噪声；
- 四条 rollout 全选同一 action 时，新版 residual 自然为 0。

代价是：v1 只区分 action family，不区分 buy/sell 的 symbol、target weight，也不区分 retrieve query
或生成该 action 的 reasoning tokens。这是为了使用现有 trace、快速得到算法信号；细粒度 action
以后再做。

## 2. 最小算法定义

对 sibling trajectory $i$，终局 finance outcome 为：

$$
Y_i=A^{fin}_i.
$$

M1 backbone 保持不变：

$$
r^{M1}_{it}=A^{fin}_i/T_i,
\qquad
U_{it}=\operatorname{StepPO\text{-}GAE}(r^{M1}_{it},V_{it}).
$$

M1 最后一步继续吸收浮点余差；$U$ 表示 whitening 前的 GAE advantage。

对一次 sibling batch 的 decision session $d$，定义：

$$
g=(\text{sibling\_group\_uid},d),
\qquad
a_{id}\in\{buy,sell,hold,abstain\}.
$$

令 $I_g$ 是该 group 的四条 rollout rows，$I_{g,a}$ 是其中 action family 为 $a$ 的 rows：

$$
\bar Y_g=\frac{1}{|I_g|}\sum_{i\in I_g}Y_i,
\qquad
\bar Y_{g,a}=\frac{1}{|I_{g,a}|}\sum_{i\in I_{g,a}}Y_i.
$$

若 group 中至少有两个不同 action families：

$$
S_{id}=\bar Y_{g,a_{id}}-\bar Y_g.
$$

否则定义 $S_{id}=0$。$\bar Y_g$ 是按 rollout rows 加权的总均值，不是各 action means 的非加权均值。
不做 action-cell std normalization、ranking、clipping、leave-one-out 或 shrinkage。

最终 actor advantage：

$$
A^{M2}_{it}
=
\operatorname{Whiten}_{step}
\left(
U_{it}+\eta\,\mathbf 1[t\text{ 是 session }d\text{ 的推进决策}]S_{id}
\right),
\qquad \eta=1.
$$

- $S$ 不作为 reward，不再经过第二次 GAE；
- $S$ 只进入 actor advantage，不进入 critic return；
- `retrieve`、`compute`、`invalid` steps 的 M2 residual 固定为 0，但仍保留 M1/GAE；
- A3-M1 是同一 backbone 的 $\eta=0$ 对照。

例子：outcomes 为 `[0, 2, 4, 8]`，actions 为 `[buy, buy, hold, sell]`。全组均值为 3.5，buy
均值为 1，因此 residual 为：

```text
[-2.5, -2.5, 0.5, 4.5]
```

## 3. 最小 grouping 与 diagnostics

唯一 group key：

```text
(sibling_group_uid, decision_session_index)
```

`sibling_group_uid` 只需要保证四条 rollouts 来自同一 episode occurrence、同一 policy snapshot。
action 不进入 key，而是在 group 内用于计算 $\bar Y_{g,a}$。

对完成的 21-session trajectory，每条 rollout 应有 20 个推进决策，因此每个 session group 的
`rollout_id` 应恰好覆盖 `{0,1,2,3}` 四行。缺行、重复 rollout、重复推进决策或 non-finite
$A^{fin}$ 是 wiring/trajectory failure；不靠重新分桶修复。

不再设置 coverage pass/fail。只报告四个便宜的 diagnostics：

1. `action_diversity_rate`：至少出现两个 action families 的 session groups 比例；
2. action-family histogram；
3. non-zero $S$ rate；
4. completion 与 invalid-action rate。

如果 action diversity 很低，M2 会自动接近 M1。这是算法信号弱，而不是触发 anchor 调参的理由。

## 4. 快速 screening experiment

MVP 只比较三个 arms：

| Arm | 定义 |
|---|---|
| A2 | 当前 generic StepPO baseline |
| A3-M1 | $A^{fin}$ + uniform M1 reward + StepPO-GAE |
| A3-M2 | 与 A3-M1 相同，再加入本文 action-conditioned $S$ |

共同设置：

- `rollout_n=4`；
- 相同模型、Train episode order、seed、prompt batch、optimizer updates、generation limits；
- 1.7B 只做 wiring；真正 screening 使用 4B；
- seeds 使用 `[42,43]`，只看方向复现，不做置信区间或显著性检验。

推荐的第一轮工作预算：

- screen-train：Train manifest 最前 32 个 episodes；
- dev-eval：Train manifest 最后 12 个 episodes，和 screen-train 的 calendar sessions 不重叠；
- 每 arm/seed 训练 32×4=128 条 trajectories；
- 三 arms、两 seeds 共 768 条训练 trajectories；
- 每个最终 checkpoint 在 12 个 dev episodes 上 deterministic 单 rollout 评估。

这些是工程筛选值，不是统计功效设计。正式 Valid 和 Final OOS 在 MVP 阶段都不使用。

对 seed $s$：

$$
Score_{a,s}=\frac1{12}\sum_{e\in Dev}A^{fin}_{a,s,e},
$$

$$
D^{M1}_s=Score_{M2,s}-Score_{M1,s},
\qquad
D^{A2}_s=Score_{M2,s}-Score_{A2,s}.
$$

工作判断：

- 两个 seeds 的 $D^{M1}_s$ 和 $D^{A2}_s$ 都为正：`PROMISING`，值得进入完整实验与正式协议；
- 否则：`NOT YET PROMISING`，先看 action diversity、$S$ scale 和训练是否接线正确，再决定改 M2
  或转 M3；
- 该判断只是研发筛选，不是论文 GO/NO-GO，也不允许使用“显著优于”措辞。

除每-seed scores/deltas 外，只记录实际 trajectory/step/token 数和第 3 节四个 diagnostics；不做 AUC、
bootstrap、手续费 sensitivity 或大量 secondary metrics。

## 5. 当前最小实现范围

只实现以下五件事：

1. launcher 让 A2/A3 的实际 `rollout.n=4`；
2. 把 upstream sibling-group UID 与 `rollout_n` 传到 trainer batch；
3. 从现有 `TradingStepTrace.action` 和 `decision_session_index` 提取每 session 的唯一推进决策；
4. 在 batch 层计算 action-family means 与 $S$；
5. 暴露 whitening 前的 StepPO-GAE $U$，执行一次 `Whiten(U+S)`，critic target 保持 M1。

本版明确不实现 portfolio snapshot、regime estimator、canonical action arguments、复杂 evidence
schema、跨 batch pool 或 nearest-neighbor matching。

## 6. 必须先有的测试

1. action-conditioned 例子得到 `[-2.5,-2.5,0.5,4.5]`；
2. 四条相同 action 得到全 0；四条 action 均不同时退化为旧的 $Y_i-\bar Y_g$；
3. 同 action rows 得到相同 $S$，group residual 在数值 tolerance 内按 rows 加权和为 0；
4. 只有推进决策 step 获得 $S$，retrieve/compute/invalid 保持 0；
5. sibling groups 不跨 episode occurrence、policy snapshot 或 session；
6. $\eta=0$ 时 actor advantage 与 M1 一致，任意 $S$ 下 critic target 与 M1 一致；
7. $S$ 不经过第二次 GAE，最终只 whitening 一次；
8. resolved commands 证明 A2、A3-M1、A3-M2 的实际 `rollout.n=4`。

通过这些 tests 和一个 1.7B wiring smoke 后，直接做第 4 节 4B screen，不再增加 protocol 工作。

## 7. 暂缓到 M2 有正信号之后

以下内容从 MVP 删除；如果 M2 screen 为正，再基于现有 `EVAL_PROTOCOL.md` 恢复：

- portfolio/regime/evidence-conditioned anchor；
- action arguments、symbol/target-weight 级 credit；
- retrieve/compute 的 step credit 或 M3 counterfactual；
- 441×4 formal run、A1 与完整四臂比较；
- resource-normalized learning-curve AUC、bootstrap、practical margin；
- seed-specific coverage gate、复杂 failure/retry accounting；
- exact commit/hash、完整 evidence schema 和 formal preflight；
- Final OOS opening、手续费翻倍与论文级结论规则。

在此之前只遵守一个已有边界：不要使用正式 Valid 或 Final OOS 来开发 M2。

## 8. 开工前只需确认的四项

1. action 是否固定为四个 families：`buy/sell/hold/abstain`；
2. 是否同意 M2 v1 只给 advancing decisions 额外 credit，information steps 留给 M3；
3. 是否采用 32 Train episodes + 12 Train-only dev episodes + seeds `[42,43]`；
4. 是否采用“两 seeds 均同时胜过 M1 和 A2”作为 `PROMISING` 的研发筛选规则。
