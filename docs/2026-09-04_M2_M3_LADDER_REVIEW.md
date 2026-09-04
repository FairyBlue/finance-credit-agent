# M2-lite / M3 候选 / 五臂阶梯 独立评审（2026-09-04）

- 评审对象：`M2_V1_PREREGISTRATION.md`、`RESEARCH_BRIEF.md`（v0.2）、`docs/2026-09-02_H100_A3_M1_WIRING_SMOKE_RESULTS.md`、`EVAL_PROTOCOL.md`（v0.1–v1.3）
- 评审方式：多视角独立评审（M2 四透镜：统计/RL 正确性、文献定位、敌意审稿模拟、实验设计与算力；M3 四路独立提案 + 评委排序；阶梯两透镜：识别混杂、资源充分性），共产出 47 条 findings；其中 8 条 critical/high 各派两名独立反驳者做对抗验证，**7 条站住、1 条被驳回**。
- 评审执行：Claude（独立于实现方 Codex 与文档作者）。本文件为归档结论；执行计划见 `docs/claude/2026-09-04_EXPERIMENT_PLAN.md`。
- 状态：评审意见，不构成协议修订。所有涉及冻结值/冻结条款的建议均须按 v0.8 线下流程走 amendment。

## 1. 总判决

| 问题 | 判决 |
|---|---|
| M2-lite 可行性 | **fixable / 值得跑 screening**，但须先修 1 个 critical（A^fin 尺度病理）+ 4 个廉价补丁；论文定位须重写 |
| M3 方向 | **M3-HAC（hold-默认账务反事实残差）为 primary，M3-TPA 为 backup**；MRIC 为 M2 成功后的条件性第二级；PBS 降为可选 control |
| 五臂阶梯 | 骨架优秀，缺 4 项对照/基线；正式预算在单 H100 上按现文不可行，且 INCOMPLETE-即停 + resume 未冻结是比算力更硬的杀手 |

## 2. M2-lite：对抗验证后站住的 critical findings

### 2.1 A^fin 的 σ→0 尺度病理（critical，双反驳者确认，代码级证实）

`src/rewards/finance.py` 的 `finance_advantage` 无 σ floor、无 clipping。近全现金轨迹 σ=0，分母退化为 ε=0.0006887，信号放大 1/ε≈1452 倍。M1 冒烟中 Valid 轨迹 A_fin=−55.288 即精确的 −B/ε（反推 B≈3.81%，与 Valid 窗行情吻合），非偶发异常。

污染链条：(a) 一条 −55 sibling 把 Ȳ_g 拖至约 −13，S 达 ±40，淹没 O(1) 的 U；(b) 极端轨迹的 20+ step rows 拉爆 DP-global whitening 的 batch std，压缩同批全部正常轨迹的 advantage；(c) M1 reward A^fin/T 方差爆炸拖累 critic；(d) screening 指标 mean(A_fin) 被单点决定；(e) 激励病理：跑输 benchmark 时增大 σ 可改善分数，训练激励"亏损中主动加波动"。

根因：§9.1 的 40 条标定代理轨迹全部满仓（σ≈市场波动率），标定域从未覆盖近现金区域；ε 被当成 tie-breaker 而非 floor。

**结论：screening 开跑前必须修 scorer，否则跑出的方向不能外推到修复后协议，等于白烧 GPU。** 首选修复：σ_eff = max(σ_agent, σ_benchmark_episode)（分母外生化，同时封死 σ 操纵通道），配 winsorize A^fin 保险。ε/λ 属已签冻结表，改动须走 amendment。

### 2.2 新颖性：以"新 estimator"立论必死，唯一可辩护的定位是识别结构（critical，站住）

代数事实：四条 sibling 动作互异时 S_id 精确等于（未归一的）GRPO 残差；全同时 S≡0。M2-lite 的准确刻画是 **"按 session 动作分歧门控、按 action family 池化的 GRPO 残差注入 StepPO"**。敌意审稿模拟的定性——"GiGPO anchor 分组退化为 session 索引 + terminal outcome 替代 return-to-go + 叠在 StepPO critic 上"——经对抗验证站住；以算法新颖性立论，主会存活率评估 <15%。

**可反转的真实新颖点（当前文档未写出）**：历史回放中 4 条 siblings 面对逐 bar 完全相同的外生市场路径，组内中心化在构造上**精确消除市场运气这一共同外生冲击**（common-random-numbers 配对识别结构）。ALFWorld/WebShop 不具备此结构（那里外生噪声≈0，分组只是方差缩减；这里外生噪声是主导项，中心化是识别手段），且直接命中核心假设五类步骤的第 5 类。配套论证：portfolio state 连续 → 跨 rollout exact-state 碰撞率≈0 → **GiGPO 移植到本环境 micro 组全为 singleton、退化为纯 trajectory-level GRPO** —— session 对齐是必然而非妥协。一段退化证明 + 实测碰撞率（近零成本）即可成文。

建议论文自称：**"session-aligned, action-family-pooled, group-relative terminal-outcome credit injected into StepPO"**；全文不用 causal/counterfactual 词汇（与预注册 §0 自我边界一致）。定位重构后存活率评估 35–45%。

### 2.3 prefix-divergence 混杂（critical，站住；同题的另一表述被驳回）

siblings 到 session d 时持仓/现金已分叉，S_id 把前后路径差异归入当期 action family；Ȳ_{g,a} 跨不同 prefix 状态池化后乘在特定状态的 ∇log π 上，梯度对 M1 目标有偏；|cell|=1 时无 leave-one-out。被驳回的版本是把它当作"screening 阶段的核心识别缺陷"（screening 是筛选规格、文档已自declare 非因果，攻击对象错位）；站住的版本是：**正式论文层面缺 estimand 定义与 bias 分析**。处理：正式阶段补偏差注记（Y 的外生/内生分解）、持仓分歧度随 session index 的诊断曲线、（若正信号）恢复粗持仓桶 anchor 或前缀分支子集对照；screening 阶段不阻塞。

### 2.4 其余站住的 critical/high（摘要）

- **"finance-native" 名不副实风险（critical，一名反驳者认可、一名部分驳回）**：M2-lite 机制本身不含金融变量，finance-native 兑现在 reward（M0/M1 层），credit 机制是 domain-agnostic 分组。出路二选一：(A) 主张改写为"exogenous-noise-paired credit for replay-based trading agents"（即 2.2 的识别结构定位）；(B) 正信号后正式版恢复 state-regime anchor。
- **dev episodes 有效样本量≈1.5（critical，反驳者解析 manifest 确认）**：最后 12 个 stride-1 episodes 只覆盖 32 个交易日。改 stride≥10 等距抽取，零成本。
- **4B 显存/主存硬阻塞（critical，反驳者查 launcher 代码确认）**：1.7B 已 87.4/94 GiB；4B 双模型 + Adam offload 到 64GB 无 swap 主机按现布局必然崩。screening 前强制 4B 单 step pilot + swapfile。
- **M2−M1 归因不到 action 条件化（critical）**：缺 M2-plain（S=Y_i−Ȳ_g，无 action 池化）对照，primary 只能声称"注入组相对终局信号"层级。M2-plain 是 M2 代码子集，screening +33% 预算。
- **screening 判定与统计（high）**："两 seed 四 delta 全正" null 通过率≈25%，只对大效应有分辨力；判定量改逐 episode 配对差分 median + winsorized mean + win-rate + σ 退化计数；预注册 DIVERSITY-LIMITED 第三出口（熵坍缩→S≡0→静默退化 M1 时判 inconclusive 而非"无信号"）；screening 迭代 append-only 留痕防 forking paths。
- **A2 臂身份（high）**：仓库现存 A2 仍是 dense reward + last_reward 写入 observation 的历史实现；直接用于 screening 则 D^{A2} 比较对象错误且三臂 prompt 不同构。先重接线 terminal-only，或改名 A2-Dense 并声明不外推。
- **完成性不对称（high）**：A1/A2 截断轨迹可静默零分而 A3 fail-closed，违反自家"不得一臂静默记零、另一臂拒绝"条款。screening 版统一为三臂同规则丢弃并记录。

## 3. M3 评审结论

四路独立提案经评委按（与 M2 互补性 ×0.25、单卡可行性 ×0.25、可冻结性 ×0.15、新颖性 ×0.2、独立扛 A3-main ×0.15）排序：

| 排名 | 提案 | 得分 | 一句话 |
|---|---|---|---|
| 1 | **M3-HAC**：hold-默认事后账务反事实残差 | 8.1 | 市场外生 + 账务确定 ⇒ "该 session 不交易会怎样"无需 LLM 重生成：动作换 hold、下游录制动作重放、冻结 scorer 重算，C_id=A^fin(τ)−A^fin(τ^{d→hold})。零 GPU、零方差（有 behavior-fixed 偏差）、不依赖 sibling diversity |
| 2 | **M3-TPA**：trade-level P&L attribution 残差 | 7.6 | Brinson 式归因搬到 step residual；HAC 的记账版近亲，带 clip + benchmark 调整；HAC 病态时的替代路径 |
| 3 | **MRIC**：遮蔽重打分信息 credit | 7.4 | 唯一覆盖 retrieve/compute 信息步的方案（C=g(Δ)·S_id），但乘性依赖 M2 的 S——M2 失败则同死；只能做 M2 成功后的 stacked 第二级 |
| 4 | **M3-PBS**：冻结金融潜函数整形 | 6.5 | Ng 定理保证渐近无效，与自设"reward shaping 不构成贡献"边界冲突；降为可选 densification control |

**推荐**：M3-HAC primary、M3-TPA backup。HAC 与 M2 构成 bias-variance 光谱两端（M2 无偏高方差、依赖 diversity；HAC 有偏零方差、只要下单就非零），恰好覆盖对方失效模式，且直接承接 M2 预注册 §4 "diversity 低则转 M3" 的分支。机制对照用 M3-perm（credit 置换臂）替代重生成式 sham。

合规：M3 槽位定义从 counterfactual information residual 改为 HAC、19.6 sham 条款按"零辅助 rollout"重释为 perm 对照——两者均构成对 RESEARCH_BRIEF 的正式 amendment，**必须在看任何 screening 数字之前**按 v0.8 线下签字。

## 4. 五臂阶梯：缺口与算力现实

骨架（terminal 信号固定、逐层单变量）是项目最强资产，不动。缺口按优先级：

1. **M2-plain 臂**（进 gate 层级，不可省，理由见 2.4）。
2. **frozen zero-shot + buy-and-hold + 全现金挂进 dev/formal 评测网格**：协议 §3 已列但矩阵未接；zero-shot 纯推理近免费，是"训练相对不训练"的必需锚点（The Losing Winner 语境审稿必问）。
3. **GiGPO**：最低要求 = 现有 trace 实测 exact-state 重合率 + related work 退化论证；审稿压力大时移植版第六臂（复用同一份 sibling rollouts，零边际采样）。
4. **A2-Dense** 重实现为 reward-neutral 后升格 formal required supplementary（"手工 dense shaping 就够了"是必然反驳，现无臂回应）。

**算力（critical，站住）**：正式 441×4×5 臂×2–3 seeds ≈ 88–132 万生成 step；以 1.7B batch=1 实测锚（~3.2s/step）外推 4B，单 H100 顺序 2–3 个月、理想批处理 2–4 周且假设零重跑。**更硬的杀手是协议自身**：v1.3 "任一 sibling 截断即整 attempt 作废 + resume 未冻结不许补跑"——截断率哪怕 0.1%，每 arm/seed 期望 ~88 次截断，期望上没有 arm/seed 能一次跑完。两个动作：(a) formal 冻结前先冻结 block 级原子 resume 协议（对外可转为"连重采规则都预注册"的加分项）；(b) 裁剪阶梯按伤识别力从小到大：第三 seed 只给 M1/M2 → M0 降 1 seed → 压缩 checkpoint grid → A1 降 1 seed；**不砍 M1/M2 的 episodes 与 rollout_n=4**。

## 5. 被驳回的批评（留痕）

- "session 对齐不匹配 portfolio state 是 M2-lite 的核心识别缺陷"（novelty 透镜版本）：被两名反驳者驳回——攻击对象错位（M2_V1 自 declare 为筛选规格而非最终预注册，且 §0 已声明非因果）。同一实质问题以"正式论文缺 estimand/bias 分析"的形式在 2.3 站住。

## 6. 评审边界

- 本评审基于 2026-09-04 的 main（`7b64dc8`）文档与代码快照；M2 screening 尚未运行，全部判断先于数据。
- 所有"存活率"为主观评估，不是预测；所有涉及冻结值的修复必须走 v0.8 amendment 流程，本文件不构成批准。
