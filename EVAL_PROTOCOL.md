# 评测协议（v0.1 起始冻结；当前审核规则以 v0.8 线下人工流程为准）

> 本文件是审稿门票。领域现状：77 篇实证研究里只有 2 篇 split 干净、1 篇报交易成本（Agentic Trading survey, arXiv 2605.19337）。协议本身就是差异化。

## 1. 数据与时间切分

- **数据源（第一版）**：直接复用 StockBench（arXiv 2510.02209，清华 KEG）已策划的数据——行情 + 新闻 + post-cutoff 评测窗口，**跳过自建 point-in-time 新闻管道**。
  - `TODO(人工)`: 克隆 repo 后确认其当前 release 覆盖的股票池与时间窗，填入下表。
- **切分原则**：严格按时间顺序 train → valid → test，禁止随机切分。test 窗口必须晚于所用基模型的 knowledge cutoff（对抗 parametric look-ahead bias）。

| 段 | 时间范围 | 备注 |
|---|---|---|
| Train | `TODO` | |
| Valid | `TODO` | 超参与早停只看这里 |
| Test (OOS) | `TODO` | 必须 > Qwen3 knowledge cutoff；只在最终评测时触碰 |

- **股票池**：10–20 只（proposal 阶段），来自 StockBench 池的子集，选取规则写死（如按流动性 top-N），禁止事后挑选。
- **Availability timestamp**：每条观测（行情/新闻/财报）必须带 `available_at` 字段；env 在时刻 t 只暴露 `available_at <= t` 的数据。

## 2. 交易假设

- 手续费：单边 10 bps（`TODO(人工)`: 确认与 StockBench/FINSABER 设定对齐）
- 滑点：固定 5 bps（第一版；敏感性分析做 0 / 10 / 20 bps）
- 成交：next-open 成交（决策日收盘后决策，次日开盘价成交）——杜绝同 bar 成交的隐性前视
- 无做空、无杠杆（第一版）

## 3. 基线（全部过同一账务引擎）

1. Buy-and-hold（等权股票池）
2. 全现金
3. 冻结 LLM zero-shot（同 prompt 同工具，不训练）
4. AlphaQuanter 对齐配置（复现其开源代码，尽力对齐数据窗）

## 4. Regime 划分

按 test 窗内实现波动率与趋势方向切成至少 3 段（如：上行低波 / 下行 / 震荡），所有结果按 regime 分列报告。`TODO(人工)`: 窗口定好后写死切分日期。

## 5. 指标

**金融**：累计/年化收益、Sharpe、Sortino、最大回撤、换手率、交易成本占比、benchmark-adjusted return。
**Agent**：每 episode 工具调用次数、无效调用率、abstain 比率、token 消耗、平均推理轮数。
**训练**：reward 曲线、跨 seed 方差、崩溃次数（reward 归零/输出重复视为崩溃）。

**统计**：≥2 seeds（proposal 阶段），报均值 ± 区间；时间分块 bootstrap 置信区间；主结论必须在手续费翻倍下方向不变。

## 6. 泄漏测试（代码合并前必须全绿，见 tests/）

| # | 测试 | 断言 |
|---|---|---|
| T1 | 未来数据扰动 | 打乱/破坏 test 期未来价格后，train 期一切特征与检索结果 bit 级不变 |
| T2 | 标签平移 | 标签整体后移一个时间单位后，模型 OOS 性能显著下降（不降 = 特征里有泄漏） |
| T3 | 标准化窗口 | 任何 scaler/统计量只 fit 在 train 窗；代码断言禁止全样本 fit |
| T4 | 检索时间一致性 | 删除未来新闻/财报后，任意历史时点 t 的检索返回完全一致 |

**人工审核范围（AI 写码，人逐行签）**：数据时间对齐、`available_at` 赋值逻辑、标准化窗口、回测成交边界。这四处的 bug 不报错且会让结果变好。

## 7. 冻结记录

| 日期 | 变更 | 审核状态 |
|---|---|---|
| 2026-08-24 | v0.1 初次冻结 | 历史自动归档记录；不构成当前人工签字，按 v0.8 线下复核 |
| 2026-08-24 | v0.2 amendment：冻结数据源/窗口、20 股池、成本、benchmark、sigma、v1 工具边界与 T2 统计口径 | 历史自动归档记录；不构成当前人工签字，按 v0.8 线下复核 |
| 2026-08-24 | v0.3 amendment：冻结 train-only 奖励标定、软阈值、事件 Regime、T2 v2 与唯一 smoke 豁免 | 历史自动归档记录；不构成当前人工签字，按 v0.8 线下复核 |
| 2026-08-24 | v0.4 amendment：旧自动归档确认流程 | 已由 v0.8 废止 |
| 2026-08-25 | v0.6 amendment：G1 改用 Valid shadow test、Final OOS fail-closed、本地结构化运行证据 | 用户明确批准，按本节归档 |
| 2026-08-30 | v0.7 amendment：G1 P1 执行约束 | 两位合作者线下人工审核通过；授权唯一一次 G1 wiring smoke |
| 2026-08-30 | v0.8 amendment：审核/签字只由合作者线下完成 | 两位合作者线下共同审核已完成；记录时间为 2026-08-30T14:46:33+00:00 |
| 2026-09-01 | v0.9 execution record：T2 v2 实现、运行证据与门禁接线 | 自动化三条件通过；不改变 v0.3 预注册口径，不构成人工 reviewer 签字 |
| 2026-09-01 | v1.0 formal manifest record：formal episode 采样、最小审核与 processed 绑定 | 两位合作者已确认；reviewed manifest、benchmark 重放和 `441/60/1` processed rows 均已验证 |
| 2026-09-02 | v1.1 A3 formal comparison amendment：StepPO v4 pin、M1 immediate-reward 正式消融、三类四配置与 M1 superiority gate | M1、比较矩阵与引用 pin 按用户要求落盘；superiority statistic 仍待两位合作者在正式训练前线下冻结 |
| 2026-09-02 | v1.2 execution record：A3-M1 实现、双角色证据门禁与 H100 wiring smoke | 自动化 wiring 验证通过；不改变 v1.1 科学定义，不构成人工 reviewer 签字或正式训练授权 |
| 2026-09-03 | v1.3 draft comparison-identification amendment：terminal-outcome A2、A3-M0、五配置阶梯与统一执行契约 | 用户已批准设计方向；待两位合作者线下共同审核，代码、测试、配置 hash 与 formal H100 preflight 完成前不批准正式训练 |

## 8. v0.2 修订（2026-08-24，用户明确批准）

> 本节是对 v0.1 的留痕修订，不删除或改写上文历史。与第 1–6 节冲突时，以本节为准；
> 未在本节修订的条款继续有效。当前审核与签字流程以第 14 节 v0.8 为准。

### 8.1 复现锁与数据快照

- Agent-R1 submodule 批准锁定到
  `b124aa46534cbf2fb8bc8af11405774984c42ac7`；verl 继续跟随上游 pin `0.7.0`。
- 股票池冻结为 StockBench 全部 20 只，不再抽取子集：`GS, MSFT, HD, V, SHW, CAT, MCD,
  UNH, AXP, AMGN, TRV, CRM, JPM, IBM, HON, BA, AMZN, AAPL, PG, JNJ`。禁止事后删票或换票。
- Train：2023-01-03 至 2024-10-31；Valid：2024-11-01 至 2025-02-28。边界均为完整
  NYSE session；train/valid 日线允许使用 yfinance 或 Stooq 的离线冻结快照。
- Test：StockBench 2025-03-03 至数据实际末日 2025-06-30，20 只股票各 83 个完整 session。
  官方“82 trading days”按 83 个价格/NAV 观测形成的 82 个相邻收益区间解释。整个窗口是一个
  不可再切分的最终 OOS test：不得拿其中任何日期做训练、验证、早停、选模或调参。
  按 XNYS 日历升序序列化为每行 `YYYY-MM-DD<LF>`（含末行 LF）后共 913 bytes，SHA-256 为
  `d272656019d9f7a856444ac903a7a6da9d5377987214d00691600b655df56860`。
- train/valid 不保证晚于 Qwen3 参数知识截止时间，存在参数级污染风险。这是当前领域常见的训练/
  验证做法，不得把其结果表述为严格 OOS；只有上述 post-cutoff StockBench 完整窗口用于最终 OOS
  结论。
- 每个数据快照必须记录实际 source、provider/package version、抓取时间和逐文件 checksum；训练 run
  还必须记录合成的数据版本。因 Yahoo 即使在 `auto_adjust=false` 时仍会回溯缩放历史 split，
  2026-08-24 冻结的 yfinance 1.6.0 快照会反向还原截至该日所有晚于每根 bar 的 split ratio，得到
  当时名义 OHLCV（`repair=false`）。共 10,820 个文件，raw-file manifest SHA-256 为
  `2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90`。StockBench 固定 repo
  commit 与 storage tree，20×83 个 OOS 文件的 raw-file manifest SHA-256 为
  `8b48fc7d50f4ecdbc49e920eb83d72df02b9a56d4365bce7f0db6bb4a342bd1f`。StockBench 自带
  `.parquet.sha256` 是规范化 DataFrame 内容哈希，不得误作 Parquet 原始字节哈希。

### 8.2 v1 可见信息与工具边界

- v1 禁用新闻；新闻只在 Phase 2 经独立 point-in-time 审计后启用。
- `retrieve` 只允许返回三类信息：经 `as_of` 门控的行情；有可审计 `available_at`、确属
  point-in-time 的基本面；仅由截至 `as_of` 的历史行情因果计算的技术指标。
- StockBench 或其他来源的“当前 fundamentals cache”在完成逐字段 PIT 审计前一律视为**非 PIT**，
  不得接入 v1，也不得在论文、README 或 run metadata 中宣称其为 point-in-time。
- 任一允许信息仍须满足 `available_at <= as_of`。日线 bar 最早在对应完整 session 收盘后可见，
  由它产生的决策只能在下一完整 session 开盘成交。
- v1 的 T4 用未来 PIT 基本面/技术观测删除不变性替代 v0.1 的新闻夹具；Phase 2 启用新闻时，
  必须恢复并通过新闻 publication-time 版本的 T4。

### 8.3 交易成本与 finance-native 轨迹信号

- 主配置冻结为单边手续费 10 bps、滑点 5 bps；滑点敏感性集合冻结为 `{0, 10, 20}` bps。
- $B$ 冻结为上述 20 股在同一账务引擎、同一成本与 next-open 边界下的初始等权
  buy-and-hold 组合。
- $\sigma$ 冻结为 agent 组合的逐日收益波动率。$\epsilon$、$\lambda$ 及风险项系数仍须在训练前
  预注册，不能用 test 调整。

### 8.4 T2 监督探针

T2 不以完整 RL 重训作第一版门槛，改为固定的监督式方向预测探针：保持特征、split 和训练流程
不变，仅把每个特征行配对的标签移到再下一个 NYSE session（即 $X_t:y_{t+1}$ 改为
$X_t:y_{t+2}$，不得反向移成截至 $t$ 已知的收益）。指标为最终 OOS 方向准确率；使用时间分块 bootstrap，
block size 固定为 5 sessions，显著性水平 $\alpha=0.05$。通过条件同时为：标签平移后的准确率相对
未平移探针显著下降，且其 95% bootstrap 区间覆盖二元随机方向基线 0.5。否则视为潜在泄漏，
不得发布主结果。

v1 探针冻结为 ridge linear direction probe。每个特征行 $X_t$ 的标签是下一 session 的 20 股
等权横截面平均收益方向 $y_{t+1}$；特征仅含截至当前收盘已知的组合 1 日收益、20 日均值/波动、
当日横截面离散度与上涨比例。平移分支在不跨 split 的共同 $X_t$ 上改配 $y_{t+2}$。
标准化只 fit train，valid 保持隔离，最终指标只在完整 StockBench OOS 计算。

## 9. v0.3 修订（2026-08-24，用户明确批准）

> 本节以追加方式修订 v0.1/v0.2，不删除或改写既有冻结历史；冲突处以本节为准。
> 预注册规则与数值已经确定；其 immutable payload hash 不受 reviewer 元数据影响。
> 两位合作者已按 v0.8 在线下共同复核；`configs/reward_calibration.pending.yaml` 仅记录通用审核角色
> 与时间，不公开个人身份。自动 agent 只代为落盘用户确认的审核结论，不是 reviewer。

### 9.1 Train-only 奖励标定与软阈值

- 所有标定只读取冻结 history 快照的完整 Train 窗 `2023-01-03`～`2024-10-31`，不得读取
  Valid 或 StockBench OOS。$\epsilon$ 使用与正式实验相同的共享账务引擎、成本和 next-open
  成交，连续回放初始等权 20 股 buy-and-hold；对组合日收益计算 20-session 滚动波动率
  （总体标准差，`ddof=0`），取中位数后乘 `10%`。
- 其余系数在 Train 内的 441 个重叠 21-session 窗口上标定。每窗固定 40 条确定性代理轨迹：
  20 条单票 buy-and-hold，以及 20 条按冻结股票顺序循环配对、在零起算第 10 个决策 session
  完整轮换的轨迹；全部 next-open 成交。合计 17,640 条轨迹、352,800 个 step。
- `risk_lambda` 取满足等式的最大系数，使平均风险罚项等于未扣风险前
  `mean(|A_fin core|)` 的 25%；风险暴露是每条轨迹的
  `max(最大回撤, 最大日双边换手超出 1.00 的部分, 最大单票收盘权重超出 0.30 的部分)`
  的均值。`drawdown_coefficient` 与 `turnover_coefficient` 分别取满足等式的最大系数，使各自
  平均罚项等于 A2 未扣罚逐步核心项 `mean(|step core|)` 的 25%；前者暴露为平均 step 回撤，
  后者暴露为平均日双边换手。

| 参数 | 冻结值 |
|---|---:|
| `epsilon` | `0.0006887293802216064` |
| `risk_lambda` | `0.9016432557217152` |
| `drawdown_coefficient` | `0.09659355421265267` |
| `turnover_coefficient` | `0.02095348253089393` |
| `turnover_limit` | `1.0` |
| `position_limit` | `0.3` |

复算统计为：20-session 日收益波动率中位数 `0.006887293802216064`、
`mean(|A_fin core|)=3.0611158447888602`、平均风险暴露 `0.848760256721104`、
`mean(|A2 step core|)=0.008371997449674203`、平均 step 回撤 `0.021668105904983786`、
平均日双边换手 `0.09988789974805483`。冻结载荷的 `freeze_table_sha256` 为
`1b46a81268fe4b45e065c49fde8bf9c0641131676668c8e35e4f884747464430`；该值是排除
`status`、`signature` 和哈希字段本身后的 canonical-JSON 载荷哈希，因此补签不会改变冻结内容身份。
人工审核状态仅以该表和线下团队记录为准；`base.yaml` 只绑定表路径、载荷哈希与数据哈希，不复制
一份可能分叉的状态。

`position_limit=0.30` 与 `turnover_limit=1.00` 均为奖励中的**软阈值**，不是拒单或强制截仓。
仓位按最大单票收盘权重衡量；换手按每个交易日的双边成交额衡量。特别地，
`turnover_limit=1.00` 为双边口径，约等于 `50%` 单边轮换。

### 9.2 T2 v2 预注册与唯一一次旧 OOS 访问留痕

- T2 v2 只允许使用 history 的 Train/Valid：按“股票 × session”合并 20 只股票样本，Train
  拟合、Valid 判定；StockBench OOS 从本修订起对探针、探针调参及功效分析永久关闭。
- 真实管道特征冻结为截至当日收盘因果可见的逐股 1 日收益、20 日收益均值/波动率，以及当日
  横截面离散度与上涨比例。保留 T2 的一期标签后移语义：`X_{i,t}` 的检测目标是同票从
  `t+1` 到 `t+2` 的方向，且 Train/Valid 各自在边界内裁去无法形成该标签的末端行。指标为
  balanced accuracy。
  显著性判定使用按 session 聚类的 5-session moving-block bootstrap（2,000 次，seed `1729`，
  单侧 $\alpha=0.05$）；“检出”定义为 balanced accuracy 的单侧 95% 置信下界高于随机基线
  `0.5`。
- 正控制分两档，均只在内存中临时注入、不得落盘：强档为准确的上述后移未来方向；弱档独立翻转
  `40%`，目标强度约为 `60%` balanced accuracy。弱档翻转集合按 seed `1729` 以及
  `seed<TAB>YYYY-MM-DD<TAB>SYMBOL<LF>` 输入的 SHA-256 阈值规则确定，不依赖库版本或遍历顺序。
  T2 v2 仅在
  **强档检出、弱档检出、真实管道未显著检出**三项同时满足时通过。
- 弱档未检出必须原样保留并报告为探针功效边界，不得删除弱档、只展示强档或将失败改写为通过。
  真实管道“未显著检出”只表示缺乏显著泄漏证据，不等于证明不存在泄漏。

2026-08-24 旧版 T2 对 StockBench OOS 的运行是唯一一次经批准的探针访问：artifact SHA-256
为 `612fe086d380126190aaa23d37a6ee9670eaaf646f993469fc3836ee0c03f347`，共同 OOS 样本数
为 `81`，aligned accuracy 为 `0.530864`，shifted accuracy 为 `0.518519`。该结果不满足旧版
通过条件，且不得用于继续挑选特征或规则；此后 StockBench 仅在最终一次科学评测中开启。

这是 v0.3 记录时的状态：T2 v2 为 **preregistered pending implementation**，因此正式训练门禁
必须直接 fail closed 且不得回退旧 StockBench 探针。当前状态已由第 15 节更新为 `PASSED`；旧 CLI
仍必须在读取配置或 OOS 输入前退出，且不得重跑旧探针。

### 9.3 公开事件锚定的 OOS regime

所有边界都是 `America/New_York` 的完整 XNYS session，首尾日期均包含；跨 session 收益归入
结束日期。边界由公开事件预先锚定，禁止按 OOS 表现再次调整。

| Regime | 日期 | Session 数 | 报告限制 |
|---|---|---:|---|
| `pre_tariff_announcement` | `2025-03-03`～`2025-04-01` | 22 | 常规分段报告 |
| `tariff_crash` | `2025-04-02`～`2025-04-08` | 5 | 仅描述性报告，不得附显著性声明 |
| `post_tariff_pause` | `2025-04-09`～`2025-06-30` | 56 | 常规分段报告 |

尤其是 `tariff_crash` 只有 5 个交易日；不得为该段报告 p 值、显著性区间或任何显著性结论。

### 9.4 A1/Qwen3-1.7B 接线 smoke 与人工审核

- 唯一 T2 豁免是当前 A1、`Qwen/Qwen3-1.7B`、四个短 episode 的 wiring smoke。它不是正式
  训练，不产生科学结论，也不能把豁免扩展到其他 arm、模型、数据集或训练配置。门禁绑定完整
  merged config 的 canonical SHA-256
  `9ad12d73e0db7b6213c602b1fe04ec7841e97e0fab11fda0652594551da02c9d`，任一配置漂移均失去豁免。
- 已授权把冻结数据快照复制到 H100、安装 GPU extra 并运行该 smoke；实际启动前仍须具备 H100
  主机访问方式、字节一致的 `configs/episode_manifest.smoke.reviewed.yaml`，并生成配置所指向的
  smoke train/valid/test processed Parquet。
- 两位合作者已于 2026-08-30 线下完成人工审核，范围覆盖数据时间对齐、`available_at`、train-only
  标准化、next-open 边界、拆股反向还原及成交量还原逻辑；当前 v0.7 G1 smoke 获准点火。

## 10. v0.4 历史自动归档规则（2026-08-24；已由 v0.8 废止）

- 本节只保留流程演进记录，不再授予任何自动 agent 人员签字或审核身份。
- v0.1～v0.3 的数据、奖励数值、Regime、T2 条件、smoke 豁免和冻结载荷哈希不因流程废止而改变。
- 当前有效规则见第 14 节：审核由合作者线下完成；在 reviewer/time 未记录前保持 pending。

## 11. v0.5 Smoke 采样与本地文件同步修订（2026-08-25，用户明确批准）

- Smoke 采样不再是待确认项：Train 窗最前 36 个 XNYS session 连续、无重叠地拆成 4 个
  9-session episode；Valid 窗最前 9 个 session 组成 1 个 episode；Test 保留完整 83 个
  session，作为 1 个不可拆分 episode。该规则同时写入 `configs/smoke_1p7b.yaml`。
- 取消 W&B 登录、在线记录和验收要求。Agent-R1 logger 固定为 console-only；运行哈希、rollout、
  reward、optimizer update 和退出状态保存在 `logs/run_queue` 的时间戳日志中。
- 数据快照使用 rsync 从 Mac 同步到 H100；运行后将 `logs/run_queue`、`outputs` 和 `artifacts`
  用 rsync 同步回本地，scp 仅作为 rsync 不可用时的备用方式。传输不引入新的人工签字环节。
- 由于明确加入完整采样规则并将 tracking 改为 local files，v0.3 的 merged-config SHA-256
  `9ad12d73e0db7b6213c602b1fe04ec7841e97e0fab11fda0652594551da02c9d` 被替代。新的唯一
  A1/Qwen3-1.7B wiring-smoke merged-config SHA-256 为
  `211a7189e746415be34f82e8ee8dab067b3bc8f6e33b243ad98595e8dfa6c411`。
- 本修订不改变奖励参数、Regime、T2 v2 条件、最终 OOS 边界或“smoke 不产生科学结论”的限制。

## 12. v0.6 G1 Shadow Test 与直接更新证据修订（2026-08-25，用户明确批准）

> 本节替代 v0.5 中“Smoke Test 使用完整 83-session Final OOS”及纯文本日志条款。官方
> Train、Valid、Final OOS 定义不变；这里只改变不产生科学结论的 G1 接线 smoke。

### 12.1 G1 专用采样

- Smoke Train 仍为官方 Train 的前 36 个 XNYS session，连续、无重叠地拆成 4 个
  9-session episode。
- Smoke Valid 为官方 Valid 的 `valid_dates[0:9]`，即 `2024-11-01`～`2024-11-13`。
- Smoke Shadow Test 改为同一官方 Valid 的 `valid_dates[9:18]`，必须精确对应以下 9 个 session：
  `2024-11-14, 2024-11-15, 2024-11-18, 2024-11-19, 2024-11-20, 2024-11-21,
  2024-11-22, 2024-11-25, 2024-11-26`。日历结果漂移时立即失败，不得换日。
- Shadow Test 仅检查数据加载、split、评估、本地日志和正常退出接线。不得用于研究结论、模型或
  checkpoint 选择、超参数选择、arm 比较、收益/Sharpe 结论或论文结果表。
- Manifest、processed rows 和运行记录必须同时冻结：
  `split_role=smoke_shadow`、`source_split=valid`、`purpose=wiring_only`、
  `scientific_use_allowed=false`、`checkpoint_selection_allowed=false`、`final_oos=false`。

### 12.2 Final OOS fail-closed

- Smoke resolved config 必须把 StockBench source 设为禁用，Train/Valid/Shadow Test 全部绑定
  history。启动时不读取、不核验、不做 StockBench overlap audit。
- 任一 source/path/provenance 指向 StockBench 或 Final OOS，任一日期进入
  `2025-03-03`～`2025-06-30`，或生成后的 Parquet 与 reviewed manifest provenance 不一致，
  preflight 和 launcher 都必须失败。不得只按 `smoke_test.parquet` 文件名判断来源。
- 官方 83-session StockBench Final OOS 继续作为不可拆分的最终科学评测窗口；其边界、Regime 和
  报告规则不因本次 smoke 修订改变。

### 12.3 本地运行记录与 G1 验收

- W&B 继续禁用。每次非 dry-run 启动创建 `runs/<run_id>/`，至少含命令、resolved config、Git
  状态、环境、数据清单、events、metrics、summary、checkpoints 和 artifacts。失败也必须保留
  `summary.json`。
- G1 不要求保存完整 1.7B checkpoint，但必须找到具有有限非零梯度的 trainable parameter，记录
  optimizer step 前后样本、gradient norm、delta L2/max，以及 backward、optimizer step、global
  update 计数。进程返回 0 但未检测到有限非零参数变化，仍判为失败。
- `--preflight-only` 只执行配置、Git/submodule、依赖/GPU/model、数据/manifest/provenance/hash、
  reward、输出、logger 和 OOS 隔离检查，不启动训练。成功时输出 `PREFLIGHT_PASSED`、
  `No training was started.` 和 `Final OOS not accessed.`。
- G0 在本节、配置/schema、旧 hash 废止、新 hash、OOS 隔离测试及全量本地验证完成后可为
  `PASSED`。G1 在 H100 inputs 与 preflight 完成但尚未真实 smoke 时为 `IN_PROGRESS`；只有真实
  smoke 同时满足 rollout、reward、optimizer 更新、干净退出与可复现记录后才为 `PASSED`。
- dry-run、preflight、模型加载、仅 rollout、本地 Mac 测试或外部历史报告均不能写成 G1 passed。

v0.5 的 merged-config SHA-256
`211a7189e746415be34f82e8ee8dab067b3bc8f6e33b243ad98595e8dfa6c411` 已废止。新的唯一
A1/Qwen3-1.7B wiring-smoke merged-config SHA-256 为
`6c58a822d3a6ff2751fbbfcda9451c9ad3eae9ffe81acb24d3417c3de40fe0ed`。

以下是 v0.6 落地当时的历史状态：G0 为 `PASSED`，G1 因尚未具备 H100 GPU extra、本地模型、
reviewed manifest 和 processed 文件而为 `BLOCKED`。该历史状态已由第 13、14 节以及 2026-08-30
H100 preflight 实测更新；真实 smoke 是否通过仍只能由本次 G1 运行证据决定。

## 13. v0.7 G1 P1 执行约束（2026-08-30，线下人工审核通过）

两位合作者已于 2026-08-30 线下人工审核并通过本节，授权在最终 `main` commit 的完整 H100
preflight 通过后启动唯一一次 A1/Qwen3-1.7B G1 wiring smoke。本授权不适用于正式 A1/A2/A3
训练；在 v0.7 当时不解除 T2 v2 阻断，也不允许访问 Final OOS。T2 后续通过状态见第 15 节。

- 唯一 smoke 模型固定为 `Qwen/Qwen3-1.7B` revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。preflight 只接受本地 exact snapshot，最终
  actor/critic 命令不得保留模型 repo ID；训练环境强制
  `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- Smoke merged-config canonical SHA-256 更新为
  `2ec544cb4e539cdbc2d3301b255384d3fdd770f5801f8d0ba3194081ac39d844`。它取代 v0.6 的
  `6c58a822...`，但不改变采样日期、reward、Regime 或 Final OOS 边界。
- `training.episodes=4` 同时绑定 processed train 行数和上游
  `data.train_max_samples=4`；布尔值、零、负数、浮点数和字符串均不得作为 episode cap。
- `retrieve` 每次最多 3 条，结果 JSON 最多 1 KiB，recent events 最多 3 条；用户控制文本以
  Unicode-safe 的有界 ASCII 摘要进入状态。模型仅保留当前两条消息，真实 tokenizer 门禁覆盖
  Train/Valid、计划 ASCII、三次最大 Unicode retrieve、多持仓混合路径及 Unicode unknown-tool
  错误路径，并同时检查 prompt limit 与 prompt + response 的模型上下文上限。
- `retrieve` 只匹配最终可见的 compact evidence，不得根据随后会被删除的原始字段命中。
- 非 dry-run 必须由外部传入 `--expected-commit <40-char SHA>`，并核对 HEAD、tracked tree、
  Agent-R1 submodule 和实验相关未跟踪/ignored 代码或配置。reviewed manifest 可作为明确允许的
  线下输入。
- `summary.global_step` 已删除，因为它过去只是 optimizer probe 记录数的别名。当前准确字段为
  `observed_update_record_count`；G1 的有效更新证据是 `backward_count >= 1`、
  `optimizer_step_count >= 1`、finite gradient 和 non-zero parameter delta。
- `scripts/run_queue.sh` 在所有参数下 fail closed，且不得启动 `uv` 或训练。P1 验收只允许本文档
  记录的 direct launcher；正式训练在 v0.7 当时仍由 T2 v2 单独阻断，现状态见第 15 节。
- 每个 preflight 阶段完成后立即写入 `artifacts/preflight_evidence.json`；后续阶段失败不得抹掉
  已完成的 Git、模型、数据、manifest、processed rows、prompt budget、最终命令及环境证据。

## 14. v0.8 线下人工审核流程修订（2026-08-30）

- 数据、manifest、配置和协议的审核/签字只由两位合作者线下完成，并在双方私有记录中留下
  reviewer 与时间；项目不引入数字签名、加密交接、W&B 或第三方审批服务。
- 自动 agent 可以复算 hash、生成报告和指出差异，但不得作为 reviewer、签字人或授权主体。
- 用户已于 2026-08-30 确认两位合作者完成线下共同审核并批准 v0.7；仓库只记录通用 reviewer
  `two project collaborators (offline joint review)` 和确认时间，不记录个人身份。
- `configs/reward_calibration.pending.yaml` 的 immutable payload 与
  `freeze_table_sha256=1b46a81268fe4b45e065c49fde8bf9c0641131676668c8e35e4f884747464430`
  保持不变；当前状态为 `signed`，signature 元数据不参与 immutable payload hash。
- 人工审核门禁已完成。最终 `main` exact commit 再次通过完整 H100 preflight 后状态为
  `READY_TO_IGNITE`；只有真实 smoke 满足 rollout、reward、backward、optimizer update、非零参数
  变化、干净退出和完整本地证据后，才能写 `G1 PASSED`。
- 本节只替代人员审核流程，不改变 v0.7 的 exact model/config、数据边界、episode cap、prompt、
  evidence、queue 或当时有效的 T2 阻断条件；该条件随后由第 15 节的 `PASSED` 结果解除。

## 15. v0.9 T2 v2 实现与执行记录（2026-09-01）

> 本节记录 v0.3 已预注册 T2 v2 的实现与实际运行结果，不修改数据窗口、特征、标签、bootstrap、
> seed、显著性阈值、正控制强度或三条件通过规则。自动 agent 生成代码、复算结果和落盘哈希，
> 不充当 v0.8 所定义的人工 reviewer。

- 实现只读取冻结 history Train/Valid；运行证据明确记录 `stockbench_oos_accessed=false`。Train 在
  20-session 因果滚动窗口和 split 内两期标签边界裁剪后为 439 sessions、8,780 个股票-session
  样本；Valid 在 split 内末端裁剪后为 78 sessions、1,560 个样本。
- 线性探针沿用 `ridge_linear_direction_v1`：L2 penalty 为 `1e-6`，截距不惩罚，均值与总体标准差
  只在 Train 拟合。正控制作为额外一列仅在内存追加到同一真实特征矩阵；零收益按非负方向处理。
  这些实现细节纳入 protocol payload，SHA-256 为
  `c26ab1e4648320e01933df63f36ac1498c58833b387bc24712d30f33aa35bdb7`。
- 严格运行结果如下；“检出”仍表示单侧 95% 下界严格高于 `0.5`：

| 分支 | Balanced accuracy | 单侧 95% 下界 | 判定 |
|---|---:|---:|---|
| 真实管道 | `0.49935736777842044` | `0.49445947047260524` | 未显著检出 |
| 强正控制 | `1.0` | `1.0` | 检出 |
| 40% flip 弱正控制 | `0.595495148126727` | `0.5756388102707657` | 检出 |

- 三条件联合判定为 `PASSED`。聚合证据保存在 `configs/t2_v2_result.json`，原始字节 SHA-256 为
  `aff81d0af5fb458768abd741264ce312a7c0767884069824a7e309286ea27436`；正控制特征和翻转数组没有
  落盘。launcher 会重新校验 artifact 字节哈希、protocol hash、history manifest、样本/session
  数、bootstrap 元数据及三条件语义，任一漂移继续 fail closed，且绝不回退旧 StockBench 探针。
- T2 v2 这一项阻塞已经解除。真实管道未显著检出仍只表示“没有显著泄漏证据”，不是无泄漏证明；
  本记录也不自动批准正式训练。formal reviewed manifest 已于本日完成；正式实验仍须满足干净
  代码身份、数据与模型 preflight，以及 A3 主结论所需 M2/M3 等独立条件，Final OOS 仍保持关闭
  至最终科学评测。

## 16. v1.0 Formal Reviewed Manifest（2026-09-01）

> 用户批准采用最小化 formal manifest 流程：只保留机器可复算的数据/split/benchmark/row 绑定和
> 两位合作者的内容确认，不增加数字签名、第三方审批或复杂同步。两位合作者已完成内容确认；自动
> agent 只执行确定性晋级和复算，不作为 reviewer。

- Formal Train/Valid 沿用第 9.1 节已经冻结的 21-session horizon，按 XNYS session、stride 1
  生成重叠窗口：Train 461 sessions 形成 441 个 episode，Valid 80 sessions 形成 60 个 episode。
  每个 episode 的 20 个相邻收益区间与奖励标定 horizon 一致，且在两个 split 内独立生成。
- Final OOS 继续是 `2025-03-03`～`2025-06-30` 的单个不可拆分 83-session episode，不产生内部
  选择窗口。Train/Valid 每个 episode 的预算为 `20 × (1 advance + 2 information) = 60` steps，
  不超过 formal `max_agent_steps=256`。
- candidate 由确定性生成器和共享 next-open 账务引擎生成。candidate 原始字节 SHA-256 为
  `4fe7da766d214f6fa14e71e080a537759354caf2de842237d244a332d19dbc98`；benchmark series SHA-256 为
  `a86507384e6806db84066a0cd286cc143664ecddb3c2b851f63562ea45b117b8`。两次独立冷启动生成结果
  byte-for-byte 相同。
- 生成前重新核验 history `10,820` 文件和 StockBench `1,660` 文件的 raw manifest，分别精确匹配
  `2612e707...a76c90` 与 `8b48fc7d...42bd1f`；每只股票分别为 541/83 sessions。
- 两位合作者已核对三个 split 的 source/日期、`441/60/1` episode 数、`21/21/83` session 长度、
  20 股 universe、benchmark series hash 和 candidate raw-byte hash。晋级只修改 status 并重命名为
  `configs/episode_manifest.reviewed.yaml`；reviewed raw SHA-256 为
  `4a8ff4f3ad8c9305c03aab17f44ebda5c4d493c22b6610a67a70ba238a5a54a3`。逆向还原 status 后 raw hash
  精确回到 candidate hash，证明其余字节未变。
- processed Train/Valid/Test 已生成并核对为 `441/60/1` rows，文件 SHA-256 分别为
  `87e9de90...21b33b`、`5e088b15...bacbcf`、`92b6e574...f692b`。Train/Valid 的 launcher 逐行绑定
  检查通过，Test 由 builder 验证为单个完整 OOS row；从冻结 raw bars 独立重放 benchmark 仍得到
  `a8650738...b117b8`。该重放只验证冻结基准，不运行 agent、不产生 Final OOS 性能或调参信号。

## 17. v1.1 A3 正式比较与 M1 消融（2026-09-02）

> 本节以追加方式澄清早期“三臂”和“M1 smoke/ablation only”的简写；不改写历史正文，不批准或
> 启动任何正式训练，也不开放 Final OOS。与早期简写冲突时，以本节为准。

- StepPO 的论文/方法语义固定引用
  [`arXiv:2604.18401v4`](https://arxiv.org/abs/2604.18401v4)（2026-06-21）。可执行实现继续独立固定为
  Agent-R1 commit `b124aa46534cbf2fb8bc8af11405774984c42ac7`；paper pin 不替代 executable
  commit pin，版本号也不构成对未固定代码的授权。
- M1 冻结为 **uniform immediate-step reward v1**：对一条实际含 $T$ 个 Agent-R1 interaction
  steps 的完成轨迹，在终局计算 $A^{fin}$ 后，将每步即时奖励设为
  $r_t^{M1}=A^{fin}/T$；最后一步吸收浮点余差，使 $\sum_t r_t^{M1}=A^{fin}$ 精确成立。
  这些数值是输入 StepPO 的 immediate rewards，不是直接或均匀的 optimizer advantages；StepPO
  随后通过 step-level GAE 计算实际 advantage。
- 正式设计保持 A1 outcome-only、A2 generic step credit、A3 finance-native 三类方法，但最小比较
  矩阵包含四个配置：`A1`、`A2`、`A3-M1` 以及采用预注册 M2 或 M3 的 `A3-main`。A3-M1 是必需
  的正式消融，必须按正式 Train/Valid、模型、StepPO 优化器、seed set、超参数选择规则和预算运行；
  它不能被表述为 A3 主方法。A3-main 必须实现 M2 或 M3，不能以 M1 冒充。
- finance-native credit-assignment 主张的必要但非充分条件是：A3-main 在与 A3-M1 相同的上述条件和
  预注册等预算口径下，通过预注册的 M1 superiority gate。若未通过，即使 A3-main 优于 A1/A2，
  也只能报告 $A^{fin}$ / reward-shaping 结果，不得声称新的逐步 credit assignment 有增益。
- superiority statistic 的主指标、聚合方式、优劣方向、阈值或不确定性规则，以及预算计量单位，仍待
  两位合作者在线下于正式训练前共同冻结；本文不自行选择数值。在该统计规则落盘并完成所需 M2/M3
  实现、测试、干净执行身份和完整 H100 preflight 前，正式比较继续 fail closed。
- 所有四个配置的 checkpoint 必须在同一次 Final OOS 开启前冻结；A3-M1 的正式消融身份不构成提前
  查看 Final OOS、选择 A3-main 或追加调参的许可。

## 18. v1.2 A3-M1 实现与 Wiring Smoke 执行记录（2026-09-02）

> 本节只记录 v1.1 M1 定义的代码实现与 wiring-only H100 验证，不改变正式比较矩阵、预算、
> superiority gate 或 Final OOS 边界。自动 agent 执行实现、测试和证据复算，不充当 v0.8 所定义的
> 人工 reviewer；本记录也不批准正式训练。

- A3-M1 已按 `immediate_step_reward_v1` 实现：完整轨迹结束后，把有限 `A_fin` 均匀分配到实际
  Agent-R1 interaction steps，最后一步吸收浮点余差以保证精确守恒。该向量进入共享 StepPO
  reward timeline，实际 optimizer advantage 继续由 step-level GAE 计算；M2/M3 仍显式
  `NotImplementedError`，不能回退或伪装为 M1。
- 完整轨迹会写入 step-credit JSONL；launcher 要求 Train/Valid 信号有限、非零、完整、精确守恒，
  并要求与 deterministic rollout-seed evidence 的 trajectory identity SHA-256 一致。A3-M1 的
  optimizer 门禁还要求分别存在 `role=critic` 和 `role=actor` 的有限非零参数更新记录，仅有退出码
  或单一角色更新都不能通过。
- 相关实现提交依次为 `dca22c5a317e1ce059f78255c544b031387feacf`、
  `d99e7d64a63c0285217f6605e0cc54cd5208e704` 和
  `fb07e727b8077c9f25684ad3bfa5222feef84035`。最终 wiring-only 配置固定
  `rollout_n=1`、`train_batch_size=1`、`total_training_steps=1`，让 critic 参数常驻 H100、critic
  optimizer 继续 offload，并把 forward-only reference 的存储 dtype 设为 BF16。这些资源项不作为
  正式 A3-M1 超参数。
- 单卡 H100 actual run
  `20260902T070540291149Z-smoke-a3-m1-qwen3-1p7b-fb07e727` 在 exact clean commit
  `fb07e727b8077c9f25684ad3bfa5222feef84035` 上得到 `status=PASSED`、
  `exit_code=returncode=0`、`training/global_step=1`、`final_oos_accessed=false`。它记录 1 条
  20-step Train 轨迹与更新前后各 1 条 Valid 轨迹，共 63 steps；三条 `A_fin`/credit 都有限非零且
  `maximum_absolute_conservation_error=0.0`。
- seed 与 credit 两侧的 trajectory identity SHA-256 同为
  `526755cdbf85e401bbd02dce5ad87d9b13833652d9260a9e16aa9080fb4eed4d`；optimizer evidence 包含
  critic、actor 各 1 条 finite non-zero update record，peak allocated GPU memory 为
  `93,829,628,928` bytes。原始 `summary.json` SHA-256 为
  `99329dbf4c624a611a06505af96164fbdbace5acf221b186ebcf5cb27d908fdf`；完整证据和限制见
  [`docs/2026-09-02_H100_A3_M1_WIRING_SMOKE_RESULTS.md`](docs/2026-09-02_H100_A3_M1_WIRING_SMOKE_RESULTS.md)。
- 协议定义的 G1 仍专指 A1/Qwen3-1.7B wiring smoke，并已于 2026-08-31 独立通过；本节的 A3-M1
  run 是后续 method-specific wiring smoke，不能称为 G1，也不能解释为 A3-M1 正式消融、A3-main
  或项目全部门禁已经通过。M2/M3、M1 superiority statistic、正式执行身份与 formal H100
  preflight 仍须完成，Final OOS 继续关闭。

## 19. v1.3 五配置识别与统一执行契约修订草案（2026-09-03；DRAFT / NOT FROZEN）

> 本节一旦按第 14 节完成两位合作者线下共同审核并明确标为 `FROZEN`，将以追加方式替代第 17 节中
> “四配置”以及 A2 `generic step credit` 的未展开简写，不删除或改写历史记录。在此之前，第 17 节
> 仍是最近一次已落盘设计记录，但其未决统计门禁与本节新增实现缺口共同阻止 formal training。本节
> 记录的是 prospective scientific design 与实施门禁，不表示当前代码已经实现，也不批准 formal
> training 或开放 Final OOS。本节新增的 M3 sham 要求只会加强、不会替代第 17 节既有的 M1
> superiority 必要门槛。
>
> 用户已批准采用 A3-M0 与下述设计方向；按第 14 节流程，仍须两位合作者线下共同审核本节、方法
> preregistration、固定实现 commit、resolved-config hashes、正式模型 revision 和资源授权，之后才能
> 把状态改为 `FROZEN`。自动 agent 不充当 reviewer 或授权主体。

### 19.1 五个核心配置

令 $P_0$ 为任何交易发生前的初始组合权益，$P_T$ 为最终 session close 由同一 next-open 账务引擎
得到的组合权益；$P_T$ 已反映冻结的 slippage 与 commission，因此下式整体是 net-of-cost return：

$$
R_{net}=\frac{P_T}{P_0}-1.
$$

对一条含 $T$ 个实际 Agent-R1 interaction steps、合法到达 environment terminal 的轨迹，A1 与 A2
使用完全相同的 external reward timeline：

$$
r_t^{raw}=
\begin{cases}
0, & t<T,\\
R_{net}, & t=T.
\end{cases}
$$

$R_{net}$ 不做 benchmark subtraction、volatility normalization、drawdown/turnover/position penalty
或 $A^{fin}$ 变换。StepPO 只消费该 $r_t$ 并估计 advantage，不生成 reward。

M0/M1/M2 共用的 terminal scorer 必须严格复现第 8.3、9.1 节和已签字 calibration payload：

$$
A^{fin}
=
\frac{R_{net}-B}{\sigma+\epsilon}
-\lambda_{risk}
\max\!\left(
D_{max},
\max(0,\tau_{max}-1.0),
\max(0,w_{max}-0.30)
\right).
$$

$B$ 是同一 episode、同一成本/next-open 账务口径的冻结等权 buy-and-hold benchmark return；$\sigma$
是 agent 组合净逐期收益的 population standard deviation（`ddof=0`）；$D_{max}$、$\tau_{max}$、
$w_{max}$ 分别为最大回撤、最大单日双边换手和最大单票 close weight。数值继续为
$\epsilon=0.0006887293802216064$、$\lambda_{risk}=0.9016432557217152$，并绑定已签字 calibration
payload 的 `freeze_table_sha256=1b46a81268fe4b45e065c49fde8bf9c0641131676668c8e35e4f884747464430`。
v1.3 不重标定或改变这些值。

A1 对同一 sampled input-row/prompt occurrence、同一 pre-update policy snapshot 的四条 rollout replicas
分别聚合完整 trajectory score，再执行 `norm_adv_by_std_in_grpo=true` 的 group-relative normalization。
分母固定使用四个 trajectory scores 的 sample standard deviation（correction $=1$）与
$\epsilon_{GRPO}=10^{-6}$；所得单一 trajectory advantage 广播到该轨迹全部 steps 的有效 response
tokens。A1 不启用 critic 或 GAE。

经两位合作者共同审核后，v1.3 formal 核心矩阵拟改为：

| 配置 | External training reward | Post-GAE、pre-whitening 增量 | Advantage/optimizer 与科学角色 |
|---|---|---|---|
| A1 Outcome-only GRPO | terminal-only $R_{net}$ | 无 | trajectory-relative GRPO、无 critic/GAE、upstream vanilla token-level importance-ratio/clipped PPO actor loss；outcome-only reference |
| A2 Terminal-outcome StepPO | 与 A1 完全相同的 terminal-only $R_{net}$ | 无 | critic + step-level GAE + GSPO；generic temporal-credit method-package control |
| A3-M0 Finance-terminal control | terminal-only $A^{fin}$ | 无 | 与 A2 相同的 StepPO；terminal signal transformation control |
| A3-M1 Uniform redistribution control | 总量守恒的 $A^{fin}/T$ immediate rewards | 无 | 与 M0/M2 相同的 StepPO；reward-timing control / M2 matched null |
| A3-M2 Main | 与 M1 相同的 base reward | 待冻结 M2 preregistration 定义的 $\eta S_t$ | 同一 StepPO，只增加 M2 residual；finance-native main method |

A3-M0 的 reward timeline 拟冻结为：

$$
r_t^{M0}=
\begin{cases}
0, & t<T,\\
A^{fin}, & t=T.
\end{cases}
$$

A3-M1 继续沿用第 17、18 节的 `uniform immediate-step reward v1`：前 $T-1$ 步写入
$A^{fin}/T$，最后一步吸收 Python/binary64 证据向量的浮点余差，使普通求和精确等于 $A^{fin}$。
M0、M1 与 M2 必须使用同一个 terminal scorer；$A^{fin}$ 是 terminal signal，不是 optimizer
advantage。

历史 `generic_step` A2 使用逐 session 的净组合收益减 benchmark、drawdown 与 turnover 罚项，属于
hand-designed dense finance reward。第 9.1 节对该 reward 系数的冻结继续作为历史记录保留，但该实现
不进入 v1.3 核心矩阵。若未来运行，必须另命名为 `A2-Dense` supplementary baseline，并另行冻结
预算、统计层级和 multiplicity；不得用旧实现冒充 terminal-outcome A2。

### 19.2 相邻对比与声明边界

五配置形成以下 prospective staircase：

1. `A2 − A1`：在相同 terminal $R_{net}$ 下比较完整 StepPO stack 与 GRPO stack；不能单独归因于
   critic、GAE 或 GSPO 任一组件；
2. `A3-M0 − A2`：在相同 StepPO 与 terminal-only placement 下比较
   $R_{net}\rightarrow A^{fin}$ 的 terminal-objective effect；不能声称 step reward redistribution；
3. `A3-M1 − A3-M0`：在相同 $A^{fin}$/StepPO 下比较 uniform redistribution 的 reward-timing
   effect；不能声称 action-specific 或因果 credit；
4. M2 的唯一 primary mechanism contrast 为 `A3-M2 − A3-M1`；
5. A3-main 相对 A1/A2/M0 的比较只作为 whole-system secondary contrasts，不能替代 M1 gate。

该设计不是完整 factorial design。RL 优化存在交互和路径依赖，以上相邻 contrasts 不能机械相加为
总贡献。除 M2 的 matched-control primary gate 外，其余相邻 contrasts 默认只报告 effect size、区间和
方向；若要宣称正式 superiority，必须在打开任何 Valid 结果前扩展检验层级和 multiplicity 规则。
M2 的 coverage/support gate 只决定 M2 是否 feasible，不得反向使 A1/A2/M0/M1 的数值失效。

### 19.3 共享 StepPO、rollout 与完整性规则

A2、A3-M0、A3-M1 和 A3-M2 使用独立参数副本，但必须共享：

- 相同 critic architecture、initial revision、initialization rule/seed、optimizer、target construction
  rule、mask 与 update schedule；各 arm 因 reward timeline 不同而产生的 target 数值不要求相同；
- 显式写入 resolved config 的 $\gamma=1$、$\lambda=1$，不得依赖上游默认值；
- 相同 GSPO actor loss、clip/KL regularization、step ordering 与 trajectory-end bootstrap boundary；
  未白化 GAE 与 M2 residual 必须在 PPO minibatch 切分前，对完整 logical on-policy rollout batch 中
  `sample_mask=true` 的真实 step rows 做唯一一次 DP-global step whitening，使用 sample standard
  deviation（correction $=1$）与 $10^{-8}$ epsilon；
- reward model 关闭，training reward 与 reference-policy KL loss 分离。

$\gamma=1$ 避免仅因 interaction depth 对同一有限期 finance outcome 施加额外时间折扣；$\lambda=1$
使完整轨迹的 Monte-Carlo residual 成为共享主规格。二者是待线下共同审核的 prospective design
choice，不是对 v1.2 smoke 实际超参数的新解释。

给定该 arm 的 external step reward，统一 GAE 语义为：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\qquad
U_t=\delta_t+\gamma\lambda U_{t+1},
$$

其中 $V(s_t)$ 是生成当前 action 前的 critic value，terminal 后 $V(s_{T+1})=U_{T+1}=0$。critic
target 使用 $U_t+V(s_t)$；actor 使用唯一一次 whitening 后的 step advantage。若
$\gamma=\lambda=1$ 且 $V(s_t)=0$，terminal-only A2 的未白化 $U_t$ 在所有 steps 上均等于
$R_{net}$；不得把 StepPO 描述为自动产生非均匀或因果 step credit。

A1、A2、A3-M0、A3-M1 与 A3-M2 的 formal Train 拟统一为
`actor_rollout_ref.rollout.n=4`。同一 episode occurrence 与同一 pre-update actor snapshot 必须先完整
生成 IDs 恰为 `{0,1,2,3}` 的四条 sibling rollouts，再允许任何 critic/actor update；不得跨 optimizer
version、attempt 或 resume boundary 拼组，也不得用重复 trajectory 伪造 siblings。A3-M1 已通过的
`n=1` smoke 继续是有效历史 wiring evidence，但不满足 formal v1.3 contract。

所有 arms 共享同一个 complete-trajectory predicate。可进入 scorer、GAE、critic 或 actor batch 的
轨迹必须在最后一个保留 step 恰好一次返回 `done=true`，且精确到达 episode 的最后冻结 session。
达到 agent/context/response 上限或异常退出而未终止时，可以保留已经产生的 raw environment/evidence
记录，但整个 four-sibling block 必须在进入 scorer-derived training reward、GAE、critic/actor batch
或 optimizer 前 hard fail；对应 formal attempt 标记 `INCOMPLETE` 并停止。不得记零、使用 partial
NAV、强制虚构 terminal step、丢弃后补采或按性能选择 retry。单个 invalid action 是 policy outcome；
只要之后完整终止便可保留。失败前已经消耗的 token、environment calls 与 GPU time 仍计入资源账本。
任何 `finish_reason=length`、response truncation 或 context overflow 都视为不完整，即使截断文本碰巧能
被 parser 解析为一个 action；不得按 arm 或 action 内容选择接受。基础设施 exact-resume 条件与每 seed
允许的 attempt 上限仍须在 formal rollout 前唯一冻结，未冻结时不允许以 replacement attempt 继续。

### 19.4 Reward-neutral observation 与共同评测

Train、Valid 和 Final OOS 的 **main rollouts** 全部采用同一个 reward-neutral model-visible schema；
条件性 M3 与 M3-sham 的 main rollouts 也必须满足本节全部可见性约束。给定相同 pre-action
environment state、action history 与正常 tool results，下列字段只能写入 trainer-side evidence，不能
进入 main-rollout 的下一 observation、prompt、tool result 或 recent-events context：

- `last_reward`、`reward_mode`、arm/configuration identity；
- $R_{net}$、$A^{fin}$、M0/M1/M2/M3/sham credit、counterfactual target/intervention/auxiliary
  outcome；
- critic value、return、raw/whitened advantage 或只供 scorer 使用的 latent finance state。

共同初始 checkpoint 必须通过 main-rollout prompt-identity oracle：以相同 frozen states/actions/request
seeds 重放时，五个 M2 核心 arms 每一步的 model-visible prompt IDs byte-for-byte 相同；若选择 M3，
M3 与 M3-sham 的 main rollouts 也必须加入同一 oracle。训练后由策略导致的 state/action 分叉是实验
结果，不是 schema 差异。

M3 auxiliary branches 采用独立契约：real 与 sham 的 pre-intervention base context、target selection 和
RNG derivation 必须相同，唯一允许的 model-visible 差异是独立 preregistration 明定的 masked
intervention 与 identity/placebo intervention。auxiliary target、prompt、output、environment state 和 RNG
均不得写回、推进或改变 main rollout；因此不要求 real/sham 的 post-intervention auxiliary prompt IDs
相同，但必须由 evidence 精确定位并验证其唯一差异。

五臂共享模型 revision、Train/Valid manifests、episode permutation、sampling 参数、seed set、
update boundaries、checkpoint grid、decoding/evaluation policy 和 retry/completion 规则。Valid 与经授权
后唯一一次 Final OOS 均在完整 rollout 后由同一 frozen scorer 计算 $A^{fin}$；不得使用各 arm 的
training reward 作为跨臂评测纵轴。raw net return、benchmark-adjusted return、Sharpe、maximum
drawdown、turnover、成本、completion/truncation/invalid rate 与 trajectory/token length 作为共同
guardrails 报告。

### 19.5 样本与资源预算

每个核心 arm/seed 的主 Train schedule 拟采用同一 441-episode Train manifest、每 episode 四条 sibling
rollouts、fixed one epoch，即每 arm/seed 计划 `441 × 4 = 1,764` 条完整 main trajectories。新增 M0
后，五臂每 seed 的计划量从 `7,056` 增至 `8,820` 条（增加 `1,764`，即 sample schedule 增加 25%）；
这不意味着 GPU/FLOP 成本恰好增加 25%。M0 不得挤占 M1/M2 的 seed、trajectory、Valid 或统计预算。
成功的每个 arm/seed 必须恰好接受全部 `1,764` 条 main trajectories；任何 incomplete sibling block
都会使该 attempt 标为 `INCOMPLETE`，不得以较少 accepted rows 生成结果。

相同 sample schedule 不等于相同实际 compute。每个 arm/seed 必须累计 Train generated response
tokens、prompt/prefill tokens、environment calls、main/auxiliary trajectories、interaction steps、
optimizer updates、actor/critic forward-backward tokens、wall-clock、GPU-seconds 和峰值显存；任何已执行
但 aborted/discarded 的工作也不得从账本删除。主学习曲线使用所有 arms 在 Train-only no-update pilot
后都可覆盖的绝对 generated-response-token horizon；environment transitions 与 GPU-seconds 作为
sensitivity axes。具体 grid、horizon、margin 与统计脚本 hash 仍须在方法 preregistration 中冻结。

### 19.6 M3 条件性 schedule-matched、resource-accounted control

五配置矩阵是当前 M2 路线的核心设计。若 A3-main 改为会生成额外 counterfactual branches 的 M3，
则必须在任何 M3 rollout 前通过独立 preregistration 增加 `A3-M3-sham`：它与真实 M3 使用相同的
$A^{fin}$、M1 backbone、StepPO、main-rollout schedule、target-step selection、auxiliary branch 数、
generation/scorer schedule、RNG namespace、硬件与完整资源记账，但采用不会删除或加入语义信息的
identity/no-op intervention，且不给 optimizer 提供有效 counterfactual residual。

M3 与 sham 的 main-rollout RNG stream 必须相同且与 auxiliary RNG namespace 隔离，auxiliary branch
的长度、失败或调度不得推进或改变 main request seeds。sham auxiliary outputs 只供资源和 identity
审计；其 post-GAE residual 必须严格为零并从 optimizer input 排除，任何非零值都使 preflight 失败。

本节只确立“若运行 M3，必须先有 auxiliary schedule 与资源记账一致的 sham”这一设计原则，不声称
当前已经得到实际 token/FLOP 完全相同的可执行 control。M3/sham 必须遵守第 19.3～19.5 节的
rollout、完整性、可见性与评测规则。资源横轴使用 $G_{total}=G_{main}+G_{aux}$，非 M3 arms 的
$G_{aux}=0$。相同 schedule 不代表实际 token/FLOP 天然相等，因此还须报告 GPU-seconds
sensitivity。

M3 的 mechanism-specific primary contrast 必须是 `A3-M3 − A3-M3-sham`，且 M3 仍必须通过第 17
节既有的 `A3-M3 − A3-M1` superiority gate；两项都是必要条件，前者不能替代后者。只有 M3 胜过
普通 M1 不能排除额外 inference/scheduling 路径的影响，只胜过 sham 也不能证明优于 M1 matched
null。具体 intervention、residual 公式、sham identity oracle、两道 gate 的层级/multiplicity、resource
horizon 与统计规则仍由单独 M3 protocol 冻结；本节不授权一个尚未定义的 M3 或 sham run。
若 M3 替代 M2，核心矩阵为 A1、A2、M0、M1、M3、M3-sham，共六个 main-training arms，即每 seed
计划 `6 × 1,764 = 10,584` 条 main trajectories，另加 M3 与 sham 的 auxiliary branches；该预算及两道
gate 的 multiplicity 必须单独授权。若 M2 与 M3 都运行，则为七个 main-training arms，不能沿用五臂
或六臂预算。

### 19.7 当前实现状态与 formal 门禁

截至本修订记录时，以下事项尚未全部实现，因此科学定义与 executable state 必须明确区分：

1. 当前 A2 仍为 `reward_mode=generic_step` dense finance reward，arm metadata 还把 reward 与
   advantage 混写为 `generic_step_advantage`；A1 command 也尚未显式锁定
   `norm_adv_by_std_in_grpo=true`、GRPO epsilon 与 vanilla actor-loss identity；
2. A3-M0 尚无独立 arm/flow/config/evidence schema 或测试；
3. launcher 仍对 StepPO arms 强制实际 `rollout.n=1`，且项目 formal config 未显式写入本节的
   $\gamma,\lambda$/whitening contract；尚未实现 formal 全臂 `n=4` contract；
4. A1/A2 尚未具备与 A3 相同的 complete-trajectory hard gate；
5. environment model-visible state 仍含 `last_reward`，reward-neutral prompt identity 尚未实现；
6. formal configs 仍缺 `training.episodes=441` 和 exact Qwen3-4B revision；
7. 五臂 resource ledger、common token horizon、checkpoint mapper 与统一 evaluator 尚未实现；正式
   seed 数、Valid rollout/eval seeds/decoding、grid、$G^*$、margin、统计脚本和 retry/attempt 规则仍待
   两位合作者冻结；
8. M2/M3 仍未完成 executable implementation，M3-sham 也未实现；
9. `M2_V1_PREREGISTRATION.md` 仍是未冻结讨论稿，其中“四臂”、A2、M0、预算及相关 TBD 必须与
   本节同步后再交由两位合作者审核；文中的“四条 sibling rollouts”仍保持为四，不得误改成五；
10. 方法统计规则、fixed commit/config hashes、test-first acceptance suite、五个核心 arms 的 `n=4`
    command/regression/wiring evidence 与完整 formal preflight 尚未完成；若走 M3，还须增加 M3/sham
    的对应测试、wiring 和 resource-isolation evidence。

上述任一门禁未完成时，不得启动 formal A1/A2/A3-M0/A3-M1/A3-M2 training；条件性 M3/M3-sham
同样不得启动。已有 A1 G1 `n=2` smoke 与 A3-M1 `n=1` smoke 只保留为历史 wiring evidence，均不
满足 formal `n=4` contract，也不解除 v1.3 门禁；Final OOS 继续关闭。
