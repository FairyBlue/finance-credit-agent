# Research Brief（v0.1，待两人签字冻结）

> 状态：DRAFT。冻结前需要：潘浩 ✍️ ＋ PhD 合作者 ✍️。冻结后本文件只允许追加，不允许修改。

## 研究问题（冻结后不再改）

**在工具调用型 trading agent 中，finance-native step-level credit assignment 能否相比 (a) outcome-only GRPO 和 (b) 通用 step-level credit（StepPO/GiGPO 式），提高训练稳定性、样本效率与样本外风险调整收益？**

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

- **M1 均匀广播**：$A^{fin}$ 平均分到每步。⚠️ 仅作为消融基线存在，用于证明"步级分配本身有增益"。
- **M2 状态分组步级优势**：借 Agent-R1 的 step-MDP，按（portfolio 状态 × 市场 regime）做 GiGPO 式锚点分组，在组内计算相对步级优势——同一市况同一仓位下，不同步骤选择的差异才归因于步骤本身。
- **M3 反事实信息贡献**：对信息获取步骤（retrieve/compute）估计反事实贡献——遮蔽该步骤的返回结果重新 rollout 决策，决策变化幅度 × 结果质量 = 该步 credit。成本高，可采样估计。

## 三臂对比

| 臂 | 优化 | 信号 |
|---|---|---|
| A1 Outcome-only | GRPO | 原始 episode 收益 |
| A2 Generic step credit | StepPO/CAPO（arXiv 2604.18401，Agent-R1 同组实现）或 GiGPO | 通用步级优势 |
| A3 Finance-native | 同 A2 的优化器 | $A^{fin}$ ＋ M2/M3 分配 |

模型：Qwen3-4B（调试用 Qwen3-1.7B）。框架：Agent-R1（verl 0.7.0）。Seeds：2（proposal 阶段），最终版补第 3 个。

## Go / No-Go（预注册，出结果前签字）

**GO（走方法论文），满足其一即可：**
1. 相同 rollout 预算下，A3 的 OOS 指标学习曲线显著快于 A1/A2（前 200–300 步即可判定，不要求收敛）
2. 跨 seed 训练稳定性更好（崩溃率 / 曲线方差）
3. OOS Sharpe、最大回撤或 benchmark-adjusted return 更优

**PIVOT（转诊断论文，接 The Losing Winner）：** 1–3 全不成立。此时论文改为"细粒度 credit 为什么救不了 trading RLVR"——量化预测能力与盈利能力的断裂点归因。**注意：'能解释断裂'不是 GO 条件，是 PIVOT 的入口**，防止赛后改标准。

## 与最近邻的差异（一句话版）

> AlphaQuanter learns tool orchestration with trajectory-level optimization; we study finance-specific **step-level** credit assignment under delayed, noisy, risk-sensitive rewards, with a contamination-controlled evaluation protocol.

完整对比见 [PAPER_MATRIX.md](PAPER_MATRIX.md)。

## Phase 2（写给博后看的 roadmap，本阶段不实现）

> After validating finance-native credit assignment in historical replay, we extend the method to hierarchical planning and microstructure execution, building on the TradeMaster / FinWorld ecosystem (EarnHFT / MacroHFT line).
