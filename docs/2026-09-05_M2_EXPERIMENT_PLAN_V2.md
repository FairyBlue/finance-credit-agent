# M2 实验计划 v2（2026-09-05，讨论稿）

> **2026-09-06 执行授权更新：** 用户确认两位合作者已线下讨论并认为本计划可行，授权按推荐方案
> 实现、验证并尝试 H100 screening，且要求服务不受 SSH 网络中断影响。具体执行值与代码证据见
> [screening 配置](../configs/m2_screening.yaml) 和 [运行手册](2026-09-06_M2_SCREENING_RUNBOOK.md)。
> 下文 DRAFT 描述保留讨论稿历史；本次授权只覆盖 screening，未开放 formal 或 Final OOS。

> **状态：DRAFT / NOT FROZEN。** 本文给出下一轮 M2 screening 的推荐讨论基线，供两位合作者修改。
> 表中的推荐值是提案，尚未执行、标定或获批；本文不改变已经签署的 reward payload，不启动训练。
> 涉及 scorer、训练契约或正式协议的变更，在实施前按 [EVAL_PROTOCOL §14、§19](../EVAL_PROTOCOL.md)
> 记录相应修订。本轮只使用历史 Train 内的数据，正式 Valid 与 Final OOS 不参与本轮筛选。

## 1. 本轮要回答什么

**主要问题：在相同 finance scorer、M1 reward timeline 和 StepPO 训练设置下，M2 residual 是否值得继续研究？**

依次解释三个比较：

1. **M2 − M1：主要筛选比较。** 检验整套 M2 residual 的训练增量。
2. **M2 − M2-plain（same-gate）：机制诊断比较。** 检验相同 diversity gate 下，action-family pooling 的增量。
3. **M2 − A2：实用参照。** 检验完整方案相对 terminal-outcome StepPO 的表现，不替代第一项。

本轮不回答正式统计显著性、完整样本效率、Final OOS 优越性或单步因果贡献。M1 已通过的 wiring
smoke 仍然有效；它没有提供正式性能结果，也不能代替新 scorer、新 `n=4` 配置的接线验证。

### 推荐方案一览

| 项目 | 推荐讨论基线 |
|---|---|
| 训练臂 | A2 / M1 / M2-plain（same-gate）/ M2，共四臂 |
| M2 定义 | 保留 session-aligned、四个 action families、actor-only residual、η=1 |
| Scorer | 优先讨论 benchmark 波动外生分母；在 CPU 标定后、任何 dev 模型评测前确定版本 |
| 数据 | 32 个 screen-train episodes；12 个日期分散的 screen-dev episodes；均来自历史 Train |
| 模型 / seeds | Qwen3-4B / `[42,43]`；1.7B 仅验证接线 |
| 训练 schedule | 每 episode 4 siblings，1 epoch；每 update 2 个 episode occurrences |
| 主训练量 | 每 arm/seed 128 条轨迹、16 个 update rounds；共 1,024 条主训练轨迹 |
| Dev 评测 | 每个最终 checkpoint，12 episodes，各 1 条 greedy rollout |
| 主要研发判断 | 两 seeds 的 mean paired `M2−M1` 均为正，且比较完整有效 → `PROMISING` |
| 额外工作 | HAC 仅作可选 CPU prototype；不作为 M2 screening 前置，也不替换信息 M3 |

与旧 MVP 相比：增加 same-gate plain；分散窗口；明确 A2 terminal-only；把 scorer 选择提前；
把 A2/plain 比较与主要判断分开。上述变化仍待讨论，不表示已经接受了任何新实验结论。

## 2. 四臂和 M2 的精确定义

令完整轨迹的共同终局评测分数为 `Y`，scorer 版本见第 3 节。这里的 `Y` 是 terminal score，
不是直接交给 actor 的 advantage。M1 将其按实际 interaction steps 均分，最后一步吸收浮点余差。

| Arm | External reward timeline | GAE 后、whitening 前的增量 |
|---|---|---|
| A2 | terminal-only `R_net` | 0 |
| M1 | 总量守恒的 `Y/T` | 0 |
| M2-plain（same-gate） | 与 M1 相同 | 推进决策上的 `S_plain` |
| M2 | 与 M1 相同 | 推进决策上的 `S_M2` |

四臂均使用相同 StepPO stack。A2 的历史 dense reward 实现不能用于本表中的 A2；本轮不加入
A2-Dense、M0 或 A1。M0 等正式对照仍按原 formal 阶梯处理，不因本轮省略而被取消。

### 2.1 Group、action 与 residual

Group key 固定为 `(sibling_group_uid, decision_session_index)`；UID 绑定同一 episode occurrence、
同一 pre-update policy snapshot 和同一采样 attempt。每组恰好包含 rollout IDs `{0,1,2,3}`。
每条完整 21-session 轨迹有 20 个推进决策。缺行、重复和 non-finite score 是接线/完成性失败。

保留四个 families：`buy / sell / hold / abstain`。暂不细分 symbol、weight，也不合并 hold/abstain：
两者当前即时账务相同，但后续可见 action history 不完全相同；合并应作为以后独立记录的设计变更。

对 group `g`，令 `Y_i` 为第 i 条 sibling 的终局分数：

$$
\bar Y_g=\frac14\sum_{i\in I_g}Y_i,\qquad
\bar Y_{g,a}=\frac1{|I_{g,a}|}\sum_{i\in I_{g,a}}Y_i,\qquad
H_g=\mathbf1\{|\{a_i:i\in I_g\}|\ge2\}.
$$

两条 residual 分别为：

$$
S^{plain}_{id}=H_g(Y_i-\bar Y_g),\qquad
S^{M2}_{id}=H_g(\bar Y_{g,a_{id}}-\bar Y_g).
$$

总均值按四条 rollout rows 加权，不是 action means 的非加权均值。两臂共享同一个 gate。
均不采用 cell std normalization、ranking、clipping、leave-one-out、shrinkage 或 residual 单独重缩放。

### 2.2 进入 optimizer 的位置

令 `U` 为 M1 rewards 经 critic/GAE 得到的未白化 advantage。M2 和 plain 都使用：

$$
A_{it}=\operatorname{Whiten}_{step}
\left(U_{it}+\eta\mathbf1[t\text{ 为 session 推进决策}]S_{id}\right),\qquad\eta=1.
$$

- `S` stop-gradient，只作用于 actor；不作为 reward、不再经过 GAE、不改变同一 batch 的 critic target。
- retrieve、compute、invalid steps 的 residual 为 0，仍保留 M1/GAE 信号。
- whitening 在 PPO minibatch 切分前，对完整 logical rollout batch 的真实 step rows 只执行一次；
  DP-global、sample std（correction=1）、epsilon=`1e-8`，不包括 padding。
- 各臂 critic 独立训练，初始化和训练规则相同；不同训练路径产生的 critic 数值不要求相同。
- `η=0` 的 M2/plain 必须在同一输入 batch 上恢复 M1。

`M2−plain` 隔离的是相同 gate 下 pooling 这一设计变化。pooling 同时改变 residual 的尺度和它与 U
的相关性，因此这个对照不能进一步单独证明“降方差”或“纯 action 语义”的作用。

## 3. Scorer：推荐变更与 CPU 决策步骤

### 3.1 为什么在训练前讨论

旧 scorer 为 `(R_net−B)/(σ_agent+ε)−λRiskPenalty`。低 agent 波动时会出现大尺度分数，且负超额
收益下可能存在增加波动改善 core 的激励。`max(σ_agent,σ_benchmark)` 只能缓解低波动放大，
其分母仍受策略影响。固定 clip 到 `[-5,5]` 也不能保证移除该激励，并可能压平有用的结果差异。

推荐先选择目标含义，再标定数值；本轮不训练多个 scorer 版本后按 dev 胜负挑选。

### 3.2 推荐候选：`benchmark_scale_v2`

$$
D_e=\max(\sigma_{benchmark,e},\sigma_{floor,cal}),\qquad
Y^{v2}_i=\frac{R_{net,i}-B_e}{D_e}-\lambda_{cal}\,RiskPenalty_i.
$$

`B_e` 和 benchmark 波动来自同一 episode 的冻结等权 buy-and-hold、共享成本和 next-open 账务；
波动使用 20 个净逐期收益的 population std（ddof=0），不年化。`D_e` 对同一 episode 的所有策略
相同，只进入终局 scorer，不写入 rollout observation。dev episode 的 benchmark 路径只在评测时
按该固定公式使用，不用于重新估计任何标定参数。

该候选是“benchmark 波动尺度归一的超额收益 + 显式风险罚项”，不是原 agent-volatility score
的数值等价修补。它移除固定 R/B/Risk 时由 agent σ 引起的分母激励；agent 波动不再被直接罚，
风险约束仍来自原 drawdown / turnover / concentration 项：

$$
RiskPenalty=\max\{D_{max},\max(0,\tau_{max}-1.0),\max(0,w_{max}-0.30)\}.
$$

推荐第一版不 clip Y。若两位合作者更重视保留原目标，可改选 agent-σ floor 方案，但必须另写准确
公式及其剩余激励；两种方案只能在本轮中选一个，不允许跨臂使用不同 scorer。

### 3.3 小型、确定性的标定方案（推荐，尚未计算）

仅使用第 4 节预先指定的 **32 个 screen-train episodes**：

1. 每窗保留旧规则的 40 条代理：20 条单票 buy-and-hold，20 条固定 cyclic midpoint rotation。
2. 每窗增加全现金、10% / 30% / 100% 股票暴露的等权 buy-and-hold，共 44 条；低暴露代理只在
   初始 next-open 建仓，余款留现金，此后持有。仍使用共享费用、舍入和成交规则。
3. 共 `32×44=1,408` 条 CPU 代理轨迹；episode 和代理均等权。推荐规则：

   $$
   \sigma_{floor,cal}=0.1\operatorname{median}_{e\in ScreenTrain}(\sigma_{benchmark,e}),\qquad
   \lambda_{cal}=0.25\frac{\operatorname{mean}_{e,p}|(R_{e,p}-B_e)/D_e|}
   {\operatorname{mean}_{e,p}RiskPenalty_{e,p}}.
   $$

4. 输出按代理类型分列的 core/Y 分位数与极值、风险项、floor 命中比例，以及旧/新 scorer 的
   诊断差异。旧公式只作历史参照，不是本轮模型选择的第二指标。
5. 要求 floor、meanRisk、λ 有限且严格正；零 benchmark 波动用正的 Train floor 处理。
   若标定不满足输入域或团队认为尺度不可用，修订候选后重新记录 CPU 结果，仍不打开 dev 模型分数。

自检包括：全现金、benchmark 同一性、零 benchmark σ、non-finite 输入；固定 R/B/Risk 时改变
agent σ，Y 不变。100% 等权代理的 core 必须为 0，但扣风险后的 Y 不一定为 0。
`0.1` 和 `0.25` 是待讨论的规则，不能据此承诺所有现金分数必为 O(1)。

新标定需要按指定 episode IDs 实现独立版本，不能直接沿用遍历全部 Train 滑窗的旧入口，也不覆盖
旧已签 payload。确定后将 scorer ID、公式、代理定义、参数及输入 IDs/hash 一起记录。
新参数不引用旧 ε/λ。M1/M2/plain 的训练和全部四臂的评测都使用这一共同新 scorer。

## 4. 数据：保持小预算，扩大日期覆盖

按当前 reviewed manifest 的 Train episode 顺序，推荐以下 **零起算索引**：

| 用途 | 精确索引 | Episode ID 范围与步长 | 数量 | 唯一 session 数 | 日历范围 |
|---|---|---|---:|---:|---|
| Screen-train / 新标定 | `0,4,...,124` | `formal-train-0001` 至 `0125`，每次 +4 | 32 | 145 | 2023-01-03 至 2023-08-01 |
| Screen-dev | `209,230,...,440` | `formal-train-0210` 至 `0441`，每次 +21 | 12 | 252 | 2023-11-01 至 2024-10-31 |

这些数量和日期已对当前 manifest 元数据核对；尚未生成或运行新的训练数据集。生成时须断言：
全部 episodes 为 history/Train、每条 21 sessions、两集合无 session 交集、dev 内各窗口无 session
交集；选中 IDs、日期、benchmark 和原 manifest 身份落盘。因果历史特征可使用动作时已可见的
历史数据；不改变现有 as-of 规则。训练按选中 IDs 升序、无 shuffle，四臂和两 seeds 使用相同顺序。

Train 内窗口仍相互重叠，dev 内也不能仅凭无重叠就认定统计独立。该划分用于小规模开发，日期覆盖
更广不等于正式功效设计；不得从覆盖天数直接推导“有效样本量”。

**历史接触边界：** 旧奖励参数曾使用完整历史 Train 标定，其中包含本轮 dev 日期。因此 screen-dev
不是从未参与历史设计的盲测集。新参数只在上述 screen-train 重算，可消除当前参数的直接数值依赖，
不能抹去历史开发接触。正式 Valid 曾用于 wiring smoke，本轮不再用它筛选 M2；Final OOS 保持关闭。

## 5. 训练设置、pilot 与预算

### 5.1 共同训练设置（推荐值）

| 设置 | 值或执行前需完成的记录 |
|---|---|
| Actor / critic | Qwen3-4B，同一 exact revision；具体 revision 在 pilot 前记录，不使用浮动 Hub ID |
| 初始化 | 每臂重新加载相同 base actor；critic 按同一规则初始化，seed 分别为 42/43；不从 pilot 权重续训 |
| 框架 | 继续使用项目已 pin 的 Agent-R1 commit 和 verl 0.7.0 |
| Episodes / rollout | 32 / `rollout.n=4`；同一 episode 的四 siblings 全部生成完才允许更新 |
| Logical batch | 2 episode occurrences = 8 trajectories；共 16 个 critic→actor update rounds |
| PPO | actor/critic `ppo_epochs=1`；mini-batch 配置 2，确认上游乘 n 后覆盖完整 8 条轨迹；micro-batch/GPU=1 |
| GAE / actor loss | γ=1、λ=1、GSPO；whitening 按第 2 节 |
| Learning rates | actor `1e-6`，critic `1e-5`；相同 optimizer、clip 与 KL 配置 |
| Train decoding | `do_sample=true, temperature=1.0, top_p=1.0, top_k=-1`；作为现有 pinned 栈起点，不声称最优 |
| 生成限制 | 沿用当前上限起点：prompt 4096、response 2048、agent steps 256；先以 pilot 验证 |
| Evaluation / checkpoint | 只评估完成第 16 round 后的最终 checkpoint；关闭训练中的 screen-dev validation 和 best-checkpoint 选择 |

表中的 batch 数是 trajectory 层口径；trainer 内部展平后的 step 数可不同。必须通过 resolved config
和真实更新计数验证 16 rounds，不能把 micro-batch backward 数当成 optimizer updates。
所有 main prompts 采用相同 reward-neutral schema：reward、arm ID、credit、critic 输出不进入观察。
RNG 按 run seed、episode occurrence、rollout ID 和 request index 派生；跨臂同规则，不靠执行时序分配。

### 5.2 只做必要的工程验证

1. **CPU：** 第 3 节标定、scorer 回归、第 8 节契约测试及四臂 resolved command 检查。
2. **1.7B wiring：** 四臂均用 `n=4`、新 scorer/flow 和 Train-only 输入，验证完整 rollout、actor/critic
   更新和共同 evaluator。不能沿用旧 `n=1` M1 smoke 代替。
3. **4B pilot：** M1 和 M2 各用最前两个 screen-train episodes、`n=4`、一个完整 update round；
   包含 rollout、critic/actor 更新和 checkpoint 写入；在同一训练 worker 中保留 actor/critic/optimizer
   状态，再做一次 post-update Train rollout，验证 optimizer states 物化后的生成内存及评测接线。
   不能仅以退出训练后新进程加载 checkpoint 的评测代替。pilot 权重随后丢弃。

记录实际 GPU 峰值、host/cgroup 内存与 swap、dtype/offload 布局、tokens/s、各阶段时间、请求及轨迹
截断数。只有实测完成更新且无 OOM/non-finite/完成性错误，才进入主 screening。
两 episode pilot 不能精确估计罕见截断率；零截断也不是整个 schedule 必然完成的保证。

资源不足时先调整共同 micro-batch/offload/vLLM 布局；swap 根据主机条件选择，不预设建 64GB 就能
解决。若需改 logical batch、模型、采样参数或生成限制，先更新共同设计和预算、重做必要 pilot，
再进入 dev 评测；不得只为某个 arm 改设置。4B 不可行时报告资源限制，不把 1.7B wiring 当 screening。

### 5.3 主预算和执行顺序

| 工作 | 计算 | 数量 |
|---|---|---:|
| 主训练 | 4 arms × 2 seeds × 32 episodes × 4 siblings | 1,024 trajectories |
| 最终 checkpoint dev | 4 arms × 2 seeds × 12 episodes × 1 rollout | 96 LLM trajectories |
| Frozen zero-shot dev | 同一未训练 4B actor × 12 episodes × greedy | 12 LLM trajectories |
| 初始 20 股等权 buy-and-hold / cash dev | 2 baselines × 12 episodes | 24 CPU ledger trajectories |
| 新 scorer 标定 | 32 episodes × 44 proxies | 1,408 CPU ledger trajectories |

两个训练 seeds 不改变同一 base actor 的 greedy zero-shot 定义，因此该基线只需评测一份。所有
基线共享日历、账务、费用和 scorer；zero-shot 与训练臂共享 prompt、工具和生成边界，固定组合
基线按预定义规则执行，其中 buy-and-hold 与 B_e 完全一致。zero-shot 推理计入成本，不称免费。

主训练相较旧三臂增加 33.3%；此数不包括 pilot、验证、失败工作、CPU 标定或推理基线。
实际记录 response/prompt tokens、steps、accepted/failed trajectories、optimizer updates、wall-clock
与 GPU-seconds。相同 trajectories 不代表相同 compute，本轮不作 token-normalized superiority 声明。

建议固定队列：seed 42 依次 A2→M1→plain→M2，seed 43 反向 M2→plain→M1→A2。全部最终模型和
zero-shot 的 dev 输出完成或明确失败后再统一判读，不因早期结果增加训练、换 seed 或提前选优。
墙钟排期以 4B pilot 的实际 token 长度和阶段耗时给出区间，不沿用 1.7B 单请求耗时线性承诺。

## 6. 失败、恢复与可判读条件

本轮推荐最小、严格的完成性规则；不引入新的 failure reward，也不把截断轨迹写成零分。

| 情况 | 处理 | 对结果的影响 |
|---|---|---|
| Invalid action，但后来完整终止 | 保留原 trajectory 和代价；invalid step 无 M2 residual | 可评测，另报 invalid rate |
| GPU/进程等基础设施故障 | 只从已验证的完整 update 边界恢复，模型/optimizer/scheduler/data/RNG 状态保持一致 | 恢复证据完整才可继续 |
| agent/context/response 上限导致不完整 | 当前 logical batch 不进 scorer-derived reward、GAE 或 optimizer；当前 arm/seed 标 `INCOMPLETE` | 保留其他已完成 runs；本轮不作完整四臂筛选结论 |
| 96 条主模型 dev 中某个 rollout 不完整 | 保留固定网格中的失败行和原因，不补零、不跨臂同剔、不换 seed 直到成功 | 相关 paired score 缺失；本轮判 `INCONCLUSIVE` |
| Zero-shot / 固定组合基线失败 | 保留失败及原因，不替代缺失分数 | 禁止相关基线比较；不单独使完整的主模型比较失效 |
| UID、trace、scorer 或数据身份错误 | 停止受影响 run，记录 engineering failure | 不解释为算法无效 |

完整轨迹须恰好在最后保留 step 终止、到达最后冻结 session，且无 response truncation/context
overflow；`finish_reason=length` 即使文本可解析也不得接受。一次失败的四 siblings 不与其他
attempt 拼组；一个 logical batch 的两个 episode groups 必须同属当前 policy snapshot。

恢复 checkpoint 至少在完整 critic→actor round 后保存并带完成标记，不从只更新了一半的状态恢复。
推荐每 4 rounds 保存一次，保留最近两个完整恢复点和最终模型；每 cell 最多两次基础设施恢复，
这些执行参数在开跑前一并确定。pilot 同时检查存储空间和保存/读取成本。
若现有栈尚不能证明精确恢复，则停止并保留结果；任何替换 attempt 要先另行记录规则。
重放相同 RNG 可能重现同一次策略截断，它不是重新抽样的理由。失败和重复执行的全部资源保留。

若 pilot 或 screening 表明此规则完成率不足，先讨论共同生成限制或新的训练失败处理目标。之后的
实验作为新版本登记，不能靠删掉难 episode、无限重采或 complete-case 均值把本轮改判成功。
这套小筛选规则不代替 formal 的完整 resume/attempt 预注册，也不要求其他已完成 cells 从零重跑。

## 7. 评测、诊断与决策

### 7.1 固定 eval policy 和主要量

推荐保留 greedy 单 rollout：`do_sample=false, temperature=0, top_p=1, top_k=-1, n=1`，每个最终
checkpoint 在同一 12-episode 网格上评测。该结果衡量 greedy policy，不是随机采样策略的期望。
若团队希望后者，应在本轮开始前改成固定 K 和 eval seeds，并按 K 倍更新预算，不在出结果后补采。

对训练 seed `s`、episode `e`，所有 arms 用同一个已确定的 scorer 得到 `Y_{a,s,e}`。定义：

$$
d^{M1}_{s,e}=Y_{M2,s,e}-Y_{M1,s,e},\qquad
\Delta^{M1}_s=\frac1{12}\sum_e d^{M1}_{s,e}.
$$

同样计算 `M2−plain`、`M2−A2`；逐 episode 配对，不先混合两个 training seeds。每个比较报告
mean、median、win-rate（严格 `d>0`，ties 单列）、最差 episode 及原始 paired rows。
**mean 是唯一主聚合量**；median/win-rate 帮助检查结果是否被少数 episode 支配，不替代 mean。
不预设 winsorization，也不根据结果挑选最好看的聚合方式。

四臂和非训练基线另报 raw net return、benchmark excess、最大回撤、turnover、交易成本、agent σ、
completion/invalid rate 和 tokens。风险恶化时保留主判断原值，并单独记录“不宜直接升级 formal”
的理由；本稿不把尚未定义的风险阈值伪装成量化 gate，也不要求方向准确率必须与收益同向。

### 7.2 最小机制诊断

每个 global update 输出一行，并汇总前/后半程；不要求逐 token 全量概率日志：

- `action_diversity_rate`：`H_g=1` 的完整 session groups 比例；同时报四 family histogram。
- `pooling_opportunity_rate`：至少两 families 且最大 action cell 至少 2 行的 groups 比例。
- `pooling_changed_row_rate`：在同一完整 batch 上离线同时算 plain/M2 后，两 residual 差异超过
  数值容差的推进 row 比例；分母为全部推进 rows。诊断计算不改变实际训练 arm。
- `nonzero_S_rate`、`mean_abs_S`；U/S 的 std 及 corr，分别标清推进 rows 与全部真实 steps 的 scope。
- 最终 whitening std、trajectory/step/token 数、completion/invalid 数；零方差时 corr 记 NA，不填零。

推荐诊断容差 `atol=1e-8, rtol=1e-6`，写入 evaluator 配置；连续尺度也同时保留，避免仅看阈值计数。
低 diversity 只是诊断，不新增未经验证的 20% hard gate。按每个 M2 training seed 分别检查 exposure：
若某 seed 的全部有效训练 batch 都没有超过诊断容差的 S，则该 seed 没有检验到 residual 的实际作用，
本轮报告 `INCONCLUSIVE / NO-RESIDUAL-EXPOSURE`，不能由另一 seed 的非零信号替它放行。
Pooling 支持也要求每个 seed 的 M2/plain 训练 traces 中确实出现过 residual 差异；否则附
`POOLING-NOT-EXERCISED`，不根据极小分数差宣称 pooling 得到支持。

### 7.3 研发判断与后续分支

先检查八个主训练 cells 是否完成计划的 128 条轨迹/16 rounds、96 个 dev rollouts 是否完整、
共同契约与 residual 接线是否有效。失败则为 `INCONCLUSIVE`，附 ENGINEERING / COMPLETION /
RESOURCE 等具体原因；可展示完整子集，但不缩小网格后重新计算本轮 verdict。

对完整有效、两个 M2 seeds 均有 residual exposure 的一轮：

| 结果 | 本轮解释 | 下一步 |
|---|---|---|
| 两 seeds 的 `Δ_M2−M1` 都 >0 | `PROMISING`：整套 M2 值得继续研究 | 结合风险、基线和预算讨论 formal，不自动授权 |
| 上项成立，且两 seeds 的 `Δ_M2−plain` 都 >0，并均有实际 pooling 差异 | pooling 也出现方向一致的支持 | 保留 pooling 作为 formal 待检验机制 |
| M2 胜 M1，但对 plain 不稳定或不为正 | residual package 有方向信号，pooling 增量未获支持 | 讨论更简单的 plain 或 formal 对照，不声称 pooling 已有效 |
| 对 M1 有方向信号，但不胜 A2/zero-shot | matched-control 结果可保留，实用价值仍有限 | 不把它写成整体系统优势 |
| 两 seeds 的 `Δ_M2−M1` 未同时为正 | `NOT YET PROMISING` | 查接线、尺度、diversity 和 episode 差异；不宣称方法已被否定 |

`PROMISING` 没有 practical margin，因此极小正差也可通过；须显示原始幅度及跨 episode 稳定性。
两 seeds 只提供研发方向复现，不提供显著性、固定 null 通过率或论文 GO。低 diversity 不自动触发
调温度/η/分桶直到成功；任何新一轮先登记修改理由、预算和共同配置，保留全部旧结果。

## 8. 实现任务与交付顺序

| 阶段 | 必要工作 | 可供合作者检查的产物 |
|---|---|---|
| A：讨论收敛 | 第 9 节选择、新 scorer 目标、筛选规则；记录必要协议修订 | 本稿修订记录、明确的候选版本 |
| B：CPU 准备 | 32/12 manifest、scorer 标定、四臂 flow/launcher、UID 透传与 residual 注入 | 标定报告、resolved commands、契约测试结果 |
| C：接线与资源 | 1.7B 四臂 wiring；4B M1/M2 pilot；恢复/最终 checkpoint 与 evaluator 验证 | GPU/host 峰值、完成性、更新计数、实际吞吐和预算区间 |
| D：主筛选 | 固定八 cells，最终 checkpoint eval 和非训练基线 | 完整结果网格、paired rows、资源账本、诊断时间序列 |
| E：共同判读 | 按第 7 节记录 verdict、限制和下一轮决定 | 一页 readout，所有尝试的追加记录 |

实现由代码贡献者完成，两位合作者讨论方法与执行资源；自动化检查提供证据，不代替人工决定。
Screening 应有独立的 history-only stage/config，不能把旧 smoke 的特殊豁免或 formal 默认路径套过来；
尤其要关闭 launcher 当前按 update 自动 validation 的行为，避免在训练时反复查看 screen-dev。

必须覆盖的机制/契约测试：

1. `Y=[0,2,4,8]`、actions=`[buy,buy,hold,sell]`：plain=`[-3.5,-1.5,0.5,4.5]`，
   M2=`[-2.5,-2.5,0.5,4.5]`。
2. 四条同 family 时两者均为 0；四条不同 family 时两者相同；两者按 rows 加权和为 0。
3. M2 同 family rows 相等；information/invalid steps 的 S 为 0；η=0 恢复 M1。
4. 同一输入下 critic target 不随 S 改变；S 不经二次 GAE，只 whitening 一次且覆盖完整 logical batch。
5. UID/policy/session 不串组；缺行、重复、non-finite 和不完整响应被拒绝。
6. 四臂实际 `n=4`；A2 terminal-only；相同状态/动作/request seeds 下 model-visible prompts 一致。
7. 新数据入口不加载正式 Valid/Final OOS；无训练中 dev 选模；完整性/恢复规则和最终 evaluator 接线正确。

本轮暂缓：完整 441-episode 阶梯、正式 bootstrap/multiplicity、learning-curve AUC、GiGPO/ECPO/BiPACE
移植臂、细粒度 action/state anchors、手续费 sensitivity、HAC 训练和原信息 M3。进入 formal 时须重新
关闭 [EVAL_PROTOCOL §19.7](../EVAL_PROTOCOL.md) 的完整门禁，并按 scorer/机制变化修订协议。

HAC 可并行做恒等账务 replay 的 CPU prototype，但要明确固定 target-weight 指令还是实际成交股数；
它研究删除交易的账务效应，不能自动替代遮蔽 retrieve/compute 后重新生成策略的信息贡献问题。

## 9. 供两位合作者集中讨论的五个决定

| 决定 | 本稿推荐 | 选择其他方案时必须一同改什么 |
|---|---|---|
| D1：scorer 目标 | 外生 benchmark 分母、不预设 clip；32×44 CPU 新标定 | 若保留 agent σ，明确公式/剩余激励；重新记录参数与共同 evaluator |
| D2：主要问题 | `M2−M1` 为研发主判断；same-gate plain 第四臂；A2 为实用参照 | 若仍要求同时胜 A2 才 PROMISING，要在训练前改表；不能事后合并两种判据 |
| D3：数据/评测 | 本稿 32/12 索引；greedy 单 rollout | 若改随机 eval，预先固定 K/seed 与预算；若改窗口，重做新 scorer 标定 |
| D4：完成性 | 完整网格才判读；基础设施 exact resume；策略截断给 INCONCLUSIVE | 若要带失败也判读，另定义 failure utility/目标及训练语义，不能默认补零/删行 |
| D5：资源 | 4B、batch=2、16 rounds；先 pilot | 若改变 sample/update schedule，各臂共同更改并重新估算；不改成仅某臂多算 |

这些是讨论入口，不要求在阅读本稿时逐项签字。本稿确定并准备实际运行时，才把所选方案、数值、
scorer/model/code 身份、配置及必要修订记录收敛成唯一执行版本。

## 10. 方法定位与讨论来源

推荐表述：

> 在共同历史市场路径下，研究 session-aligned、action-family-pooled terminal-outcome residual
> 对 StepPO 的训练增量，以 uniform-reward matched control 和 same-gate individual-residual
> control 区分整套 residual 与 pooling 的作用。

Session 对齐没有匹配完整 portfolio/prefix，也没有证明市场运气被精确消除或 estimator 无偏。
同 family 仍可能包含不同 symbol/weight。terminal score 非可加并不妨碍定义 return-to-go；本轮
不把 pooling 公式本身宣称为全新 estimator，不用主观论文录用概率作决策依据。

- 现有机制基线：[M2_V1_PREREGISTRATION](../M2_V1_PREREGISTRATION.md)、
  [RESEARCH_BRIEF](../RESEARCH_BRIEF.md)、[EVAL_PROTOCOL](../EVAL_PROTOCOL.md)。
- 输入意见：[2026-09-04 评审](2026-09-04_M2_M3_LADDER_REVIEW.md)、
  [原执行计划草案](claude/2026-09-04_EXPERIMENT_PLAN.md)。本文选择性吸收工程、对照和诊断建议。
- 最近邻：[GiGPO v3](https://arxiv.org/html/2505.10978v3) 包含 anchor 和 similarity grouping；
  [ECPO](https://arxiv.org/html/2606.05885v1) 包含 action pooling/shrinkage/可靠性门控；
  [BiPACE](https://arxiv.org/html/2606.25556v1) 包含状态聚类和 action-conditioned peer baseline。
  正式写作需逐项比较适用条件与实现，不预设这些方法在交易环境必然退化。

## 11. 修订记录

| 日期 | 版本 | 记录 |
|---|---|---|
| 2026-09-05 | v2 discussion draft | 汇总前轮讨论，提出四臂 same-gate screening、外生 scorer 候选与具体 32/12 schedule；尚无新标定或训练结果 |

后续每次讨论记录日期、改动、理由和影响范围；进入实验后，配置、seed、失败、结果和决策追加留痕，
不删除不利尝试。本文尚未获得两位合作者的共同冻结确认。
