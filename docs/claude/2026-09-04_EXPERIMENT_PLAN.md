# 后续实验计划（2026-09-04，基于同日评审）

历史草案说明（2026-10-17）：以下排期、scorer、无 gate plain 与 M3-HAC 等为 9 月 4 日提案，不代表当前已采纳合同。实际筛选设计由后续修订确定；当前 plain 使用与 M2 相同的分歧 gate，M3 仍未启用。最新结果及下一步见[文档导航](../README.md)。

- 依据：`docs/2026-09-04_M2_M3_LADDER_REVIEW.md`（对抗验证后的评审结论）
- 性质：执行计划草案。涉及冻结值/冻结条款的条目已标 `[AMENDMENT]`，须按 v0.8 线下流程签字后才生效；其余为工程任务，可直接派发。
- 角色约定：**Codex** = 实现与测试；**合作者** = H100 执行与线下复核；**用户** = 决策与 amendment 授权；**Claude** = 评审与判读。

## 阶段 0：Screening 前置补丁（预计 2–4 天，全部完成才许点火 4B screening）

### P0-1 修 A^fin scorer `[AMENDMENT]`
- σ_eff = max(σ_agent, σ_benchmark_episode)：分母下限取同 episode 冻结 benchmark 路径的日收益 population std；另加 winsorize A^fin ∈ [−5, +5] 保险层。
- 重新生成 reward 标定表（ε 定义随 σ_eff 联动复算），新表走 v0.8 签字。
- 验收：全现金合成轨迹的 |A^fin| 回到 O(1)；"加波动改善分数"通道封死（构造用例断言）；M1 冒烟场景回放 A_fin 不再出现 ±50 量级。
- Owner：Codex 实现 + 用户/合作者签字。

### P0-2 dev/screen-train episode 去重叠
- dev：Train 尾部按 stride≥10 等距抽 12 个 episodes；screen-train：前段 stride 2–4 抽 32 个；两者 calendar 无重叠。
- 验收：dev 覆盖 ≥100 个不同 sessions；manifest 索引与 hash 落盘。
- Owner：Codex。零训练成本。

### P0-3 screening 判定量与出口改造（更新 M2_V1_PREREGISTRATION §4）
- 判定量：逐 episode 配对差分 D_e = A^fin_arm(e) − A^fin_M1(e)，报 median、10% winsorized mean、win-rate；mean(A_fin) 降为 secondary。
- PROMISING = 两 seed 的配对 median(D^M1) 与 median(D^A2) 均 >0；文档明写该规则 null 通过率约 25%、仅授权进入正式冻结、不构成 effect 主张。
- 新增第三出口 **DIVERSITY-LIMITED**：训练后 1/3 的 action_diversity_rate < 20% 时判 inconclusive，触发熵/温度诊断，不判"M2 无信号"、不转 M3。
- σ 退化 episode（σ<σ_benchmark×0.1）计数为必报诊断。
- screening 迭代 append-only 留痕（配置、seed、结果、决策），论文附录披露迭代总数。
- Owner：Codex。

### P0-4 第四臂 M2-plain
- S'_id = Y_i − Ȳ_g，无 action 池化、无门控，其余与 M2 完全同管线（η=1、同 whitening、critic 不变）。
- 声明结构：M2−M1 归因"sibling 组相对终局信号"；M2−M2-plain 归因"action 条件化增量"。
- 附带零成本预检：用 M2 trace 离线统计 S_action≠S_plain 的 step 比例（pooled-cell rate 诊断一并加入 §3：≥2 families 且 max cell ≥2 的 session group 比例）。
- Owner：Codex。screening 预算 3 臂→4 臂（+33%，768→1024 条训练轨迹）。

### P0-5 A2 臂身份重接线
- A2 重接线为 terminal-only R_net（去 dense reward、去 last_reward 写入 observation），与 RESEARCH_BRIEF 正式 A2 定义一致；若时间不允许，则臂名改 A2-Dense 并在判定中声明 D^{A2} 不外推——二选一写死，不得含糊。
- 三臂（现四臂）截断轨迹统一规则：同 episode block 各臂对称丢弃并记录，禁止单臂静默零分。
- γ=1、λ=1 与 DP-global whitening scope 显式写入 screening resolved config，加断言测试。
- 跨臂冻结相同 rollout 采样参数（temperature/top_p 按 Qwen3 推荐，四臂一致）。
- Owner：Codex。

### P0-6 4B 资源 pilot（硬性 go/no-go）
- 一次 4B、rollout_n=4、2 个 21-session episodes、1 个 global step 的实测：GPU 峰值、host RSS 峰值、tokens/s、批处理加速比、截断率。
- 先建 NVMe swapfile ≥64GB（OOM-kill 保险，不改算法语义）。
- 产出新的墙钟外推锚，重排 screening 排期。
- Owner：合作者（H100）。约 1–2 小时。

### P0-7 杂项
- hold/abstain 账务等价性确认：若等价，S 计算内合并为 no-trade family（prompt 层不变），更新 §6 test 1。
- 诊断改 per-global-step 时间序列：action_diversity_rate、决策 token 策略熵、mean|S|、非零 S 率、std(U)/std(S)/corr(U,S)。
- Owner：Codex。

## 阶段 1：4B Screening（P0 全绿后点火，预计 2–4 天墙钟）

- 四臂：A2（terminal-only）/ A3-M1 / A3-M2 / A3-M2-plain × seeds [42,43]，rollout_n=4，screen-train 32 eps + dev 12 eps（P0-2 口径）。
- dev-eval：每 episode 固定 eval-seed 采样 2–4 条 rollout 取均值（弃用确定性单 rollout）；dev 轨迹截断时四臂同剔并报剔除数。
- 免费挂载：frozen zero-shot（4B 同 prompt 同工具）与 buy-and-hold、全现金三条非训练基线跑同一 dev 网格。
- 判读（Claude 参与）：按 P0-3 判定量 + 诊断时间序列，三出口：PROMISING / NOT YET PROMISING / DIVERSITY-LIMITED。

## 阶段 2（与阶段 0–1 完全并行）：M3-HAC 准备（纯 CPU/工程，不占卡）

1. `[AMENDMENT]` M3 槽位定义变更：counterfactual information residual → **M3-HAC**（hold-默认账务反事实残差，C_id = A^fin(τ)−A^fin(τ^{d→hold})，post-GAE pre-whitening actor-only residual，η=1）；19.6 sham 条款重释：零辅助 rollout 时以 **M3-perm**（credit 冻结 RNG 置换臂）承担 mechanism gate。双 gate：M3−M1（superiority）+ M3−perm（mechanism）。**须在看任何 screening 数字前签字。**
2. TradingStepTrace 成交明细审计：symbol/股数/成交价/佣金字段是否完整；缺则补字段（同时利好 M3-TPA backup）。
3. 动作替换重放 API + 下游不可行订单裁剪规则（sell→min(录制量,持仓)、buy→现金约束，舍入随 commit 冻结）。
4. **恒等重放自检为 hard gate**：不替换任何动作时，1e-9 相对容差内复现该轨迹已记录 A^fin；不过则该轨迹 wiring failure。此测试同时覆盖"bug 不报错且让结果变好"最高危区，列入人工审核清单。
5. M3-HAC 预注册草案（镜像 M2 MVP 口径：32+12 episodes、seeds [42,43]、PROMISING 同规则），backup 登记 M3-TPA、条件性第三级登记 MRIC（仅 M2 PROMISING 时以 stacked 模式激活）、PBS 登记为可选 densification control。
- Owner：Codex（实现）+ 用户/合作者（amendment 签字）。

## 阶段 3：Screening 后分支

### 分支 A：M2 PROMISING
1. `[AMENDMENT]` 正式协议冻结包：M2-plain 进 gate 层级（两层归因声明）；M2−M1 primary gate 统计规则（配对、时间分块 bootstrap，引用 EVAL_PROTOCOL §5 框架）；multiplicity 层级；第三 seed 规则（只给 M1/M2）。
2. 正式版 novelty 补件：GiGPO exact-state 重合率实测 + 退化论证（或移植第六臂决策）；prefix-divergence 偏差注记 + 持仓分歧度诊断曲线；粗持仓桶 anchor 或前缀分支子集对照三选一。
3. MRIC 以 stacked 模式（M2M3−M2 adjacent contrast）进入 M3 预注册讨论。

### 分支 B：M2 NOT YET PROMISING
- 按 M2 §4 流程查 diversity/S scale/接线；确认非工程问题后启动 M3-HAC screening（A2/M1/M3 三臂，同预算口径），M2 叙事降级为"gated group-relative injection 无增益"的诚实记录。

### 分支 C：DIVERSITY-LIMITED
- 熵/温度诊断（不属于 M2 算法失败）；同时 M3-HAC screening 可提前启动（HAC 不依赖 diversity，正是该分支的设计答案）。

## 阶段 4：Formal 前置（不阻塞以上任何阶段，但 formal 冻结前必须关闭）

1. `[AMENDMENT]` **block 级原子 resume 协议**：four-sibling block 为原子单位 checkpoint+RNG 精确续跑；截断 block 记账后按冻结规则重采一次、全臂对称；attempt 上限由 4B pilot 实测截断率标定。现行"INCOMPLETE 即停 + 不许补跑"在期望上使任何 arm/seed 都跑不完，必须先修。
2. `[AMENDMENT]` 预算裁剪阶梯（伤识别力从小到大）：第三 seed 只给 M1/M2 → M0 降 1 seed → 压缩 Valid checkpoint grid → A1 降 1 seed；不砍 M1/M2 episodes 与 rollout_n=4；不按臂差异化 episode 数。
3. A2-Dense reward-neutral 重实现，升格 formal required supplementary（2 seeds）。
4. Losing-Winner 式诊断写进 gate：各臂同时报方向准确率与 A^fin，GO 情形须两者同向，否则自动 PIVOT。
5. M2_V1 §7 暂缓清单改为直接引用"EVAL_PROTOCOL §19.7 第 1–10 项全部关闭"；"四臂"字样同步为五配置。
6. Qwen3-4B exact revision pin、`training.episodes=441`、seed_set 展开为 per-seed queue matrix。

## 时间线（乐观口径）

| 周 | 内容 |
|---|---|
| W1（本周） | 阶段 0 全部 + 阶段 2 的 1–4 + 两份 amendment 签字 |
| W1 末–W2 | 4B screening（阶段 1）+ M3-HAC 预注册草案 |
| W2 | screening 判读 → 走分支 A/B/C；formal 前置（阶段 4）amendment 起草 |
| W3+ | 分支对应的正式冻结与 formal 排期（受裁剪阶梯与 resume 协议约束） |

## 论文定位句（评审建议，供 brief 修订参考）

> 中文：在历史回放的金融 agent 训练中，同一 episode 的 sibling rollouts 逐 session 面对完全相同的外生市场路径；session 对齐、action-family 池化的组内终局对比在构造上精确消除市场运气成分——而 exact-state anchoring（GiGPO）在连续 portfolio state 下退化为零信号、风险调整终局分的非可加性使 return-to-go 无定义。我们将该对比作为加性残差注入 StepPO，并以 matched uniform-redistribution null（M1）、plain-residual 对照（M2-plain）与污染受控评测协议隔离其净增量。

写作纪律：全文自称 "session-aligned, action-family-pooled, group-relative terminal-outcome credit"；不用 causal/counterfactual 词汇；causal 版本明确留给 M3-HAC，形成 roadmap。
