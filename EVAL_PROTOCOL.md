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
