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
