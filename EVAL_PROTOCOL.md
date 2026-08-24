# 评测协议（v0.1 起始冻结；原双人签字规则已由 v0.4 授权确认规则替代）

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

| 日期 | 变更 | 签字 |
|---|---|---|
| 2026-08-24 | v0.1 初次冻结 | Codex 经当前任务用户明确授权，于 `2026-08-24T21:39:30+08:00` 归档确认 |
| 2026-08-24 | v0.2 amendment：冻结数据源/窗口、20 股池、成本、benchmark、sigma、v1 工具边界与 T2 统计口径 | Codex 经当前任务用户明确授权，于 `2026-08-24T21:39:30+08:00` 归档确认 |
| 2026-08-24 | v0.3 amendment：冻结 train-only 奖励标定、软阈值、事件 Regime、T2 v2 与唯一 smoke 豁免 | Codex 经当前任务用户明确授权，于 `2026-08-24T21:39:30+08:00` 归档确认 |
| 2026-08-24 | v0.4 amendment：以用户授权的 Codex 归档确认替代另行双人签字 | Codex 经当前任务用户明确授权，于 `2026-08-24T21:39:30+08:00` 归档确认 |

## 8. v0.2 修订（2026-08-24，用户明确批准）

> 本节是对 v0.1 的留痕修订，不删除或改写上文历史。与第 1–6 节冲突时，以本节为准；
> 未在本节修订的条款继续有效。原另行双人归档签字要求已由第 10 节替代，不再单独办理。

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
> 预注册规则与数值已经确定；`configs/reward_calibration.pending.yaml` 已由 Codex 经当前任务用户
> 明确授权，于 `2026-08-24T21:39:30+08:00` 确认，状态为 `signed`。不再要求另行双人归档签字。

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
平均日双边换手 `0.09988789974805483`。已签字表的 `freeze_table_sha256` 为
`1b46a81268fe4b45e065c49fde8bf9c0641131676668c8e35e4f884747464430`；该值是排除
`status`、`signature` 和哈希字段本身后的 canonical-JSON 载荷哈希，因此补签不会改变冻结内容身份。
签字状态仅以该表为准；`base.yaml` 只绑定表路径、载荷哈希与数据哈希，不复制一份可能分叉的状态。

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

T2 v2 当前状态是 **preregistered pending implementation**。在其实现、测试并按以上条件通过前，
正式训练门禁必须直接 fail closed 且不得回退旧 StockBench 探针；旧 CLI 也必须在读取配置或 OOS
输入前退出。不得重跑旧探针，也不得把预注册文本本身当作门禁通过。

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
- 已授权把冻结数据快照复制到 H100、安装 GPU extra、登录 W&B 并运行该 smoke；实际启动前
  仍须具备 H100 主机访问方式，完成人工审核的 `configs/episode_manifest.smoke.reviewed.yaml`，
  并生成配置所指向的 smoke train/valid/test processed Parquet。授权不等于这些前置条件已经完成。
- 人工审核清单和 full/smoke manifests 计划于本周完成。逐行审核继续覆盖数据时间对齐、
  `available_at`、train-only 标准化与 next-open 边界，并将未来拆股反向还原及成交量还原逻辑列为
  重点签字项。

## 10. v0.4 归档确认规则修订（2026-08-24，用户明确授权）

- 当前任务用户明确授权 Codex 对本协议及奖励冻结表完成签字确认。
- 本授权替代 v0.1 中“冻结后修改需双人签字”以及后续潘浩 / PhD 另行归档签字要求；v0.1、v0.2、
  v0.3 和奖励冻结表均视为已归档确认，不再保留额外签字环节或对应 `TODO(human)`。
- 归档签字身份统一记录为“Codex（经当前任务用户明确授权）”，确认时间为
  `2026-08-24T21:39:30+08:00`。
- 本修订只改变归档确认流程，不改变任何数据、奖励数值、Regime、T2 条件、smoke 豁免或冻结哈希。
