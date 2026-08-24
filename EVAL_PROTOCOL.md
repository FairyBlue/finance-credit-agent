# 评测协议（v0.1 —— 代码开始前冻结，冻结后修改需双人签字并留痕）

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
| `TODO` | v0.1 初次冻结 | 潘浩 / PhD |
| 2026-08-24 | v0.2 amendment：冻结数据源/窗口、20 股池、成本、benchmark、sigma、v1 工具边界与 T2 统计口径 | 用户在本任务中明确批准；潘浩 / PhD 归档签字 `TODO(human)` |

## 8. v0.2 修订（2026-08-24，用户明确批准）

> 本节是对 v0.1 的留痕修订，不删除或改写上文历史。与第 1–6 节冲突时，以本节为准；
> 未在本节修订的条款继续有效。正式双人归档签字仍按第 7 节记录。

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
