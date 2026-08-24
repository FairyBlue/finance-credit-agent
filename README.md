# finance-credit-agent

Research scaffold for testing whether finance-native **step-level credit assignment** improves a
tool-using trading agent over outcome-only GRPO and generic StepPO. The project composes with
[Agent-R1](https://github.com/AgentR1/Agent-R1) as a locked submodule, keeps every local change under
`src/`, and treats look-ahead leakage prevention as the first executable requirement.

The three experimental arms are:

| Arm | Upstream optimizer | Reward/credit signal |
|---|---|---|
| A1 | GRPO (`adv_estimator=grpo`) | Original episode return, emitted only at trajectory end |
| A2 | Agent-R1 StepPO (`adv_estimator=gae`, critic + GSPO loss) | Generic per-environment-step reward |
| A3 | StepPO | `A_fin` plus a strategy-selected step allocation |

A3's M1 uniform broadcast is implemented only as a smoke-test/ablation. M2 state×regime grouping and
M3 sampled counterfactual information contribution have explicit interfaces and deliberately raise
`NotImplementedError`; a final A3 scientific comparison must implement at least one of M2/M3.

## Reproducibility pins

- Agent-R1 gitlink: `b124aa46534cbf2fb8bc8af11405774984c42ac7` (2026-07-21). It is the latest
  verified default-branch commit and contains StepPO; the only tag, `v0.1.0`, does not. Upstream has
  no GitHub release, so human confirmation of this untagged stable point remains required.
- verl: exactly `0.7.0`, following Agent-R1. Do not upgrade it independently.
- StockBench candidate source: [`ChenYXxxx/stockbench`](https://github.com/ChenYXxxx/stockbench),
  repo commit `ce8b2b3483590646ad3b650ac8221f43f76fd091`, storage tree
  `f9553f735d6bd2d797541bec519cbd8028adee40`. The dataset has no tag/release.

Every training command injects `agent_r1_commit`, `repo_commit`, and `data_version` into the Hydra
configuration recorded by W&B. Non-dry training refuses a dirty/uncommitted repository by default.

## Project map

```text
configs/                  frozen assumptions, arm configs, and Agent-R1 flow recipes
src/data/                 StockBench-only offline loader and mandatory as_of gate
src/env/                  next-open ledger, BaseTool registrations, TradingEnv plugin
src/rewards/              A_fin and M1/M2/M3 strategy interface
src/agents/               three training entrypoints, hash capture, Agent-R1 flows
src/evaluation/           shared metrics/baselines and three standard figures
tests/                    leakage, accounting, reward, environment, and metric specifications
scripts/run_queue.sh      one-GPU sequential experiment queue
scripts/watchdog.py       collapsed-reward/repeated-output process watchdog
third_party/agent-r1/     read-only locked submodule
```

`RESEARCH_BRIEF.md` and `EVAL_PROTOCOL.md` are frozen inputs and are not generated or rewritten by
the code.

## CPU quick start

```bash
git submodule update --init --recursive
uv sync
uv run pytest
uv run python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml --dry-run
```

The default `uv sync` is CPU-only. GPU libraries live exclusively in the `gpu` optional extra; see
[`SETUP.md`](SETUP.md) for the H100 installation order.

## Data safety model

`StockBenchAdapter.get_observations(as_of)` is the only public observation read method. It always
enforces `available_at <= as_of`; no raw/all-data accessor exists. Real-data ingestion reads only:

- `storage/parquet/{TICKER}/day/*.parquet` for daily OHLCV/VWAP; and
- `storage/cache/news_by_day/{TICKER}/*.json` for news, using its integer UTC epoch.

It never invokes StockBench's network-capable `DataHub`. Indicators, financials, query caches, and
company-action caches remain unavailable pending a point-in-time audit. Daily parquet sidecar
SHA-256 values are verified during ingestion. StockBench code is
Apache-2.0, but Polygon/Finnhub-derived data has no clearly separate redistribution license in the
repository; keep data external under ignored `data/` and obtain a human/legal confirmation.

Current leakage-test status:

| Test | Executable status |
|---|---|
| T1 future-price corruption | Implemented; historical retrieval and numeric feature bytes invariant |
| T2 shifted labels | Strict xfail integration skeleton until metric/significance threshold is frozen |
| T3 train-only statistics | Implemented; full-sample fits raise `FullSampleFitError` |
| T4 future-document deletion | Implemented; historical retrieval bytes invariant |

## Preparing audited episodes

After resolving every data/reward TODO in `configs/base.yaml`, copy
`configs/episode_manifest.example.yaml`, provide timezone-aware close/open timestamps and an aligned
frozen benchmark series, then run:

```bash
uv run python scripts/prepare_episodes.py --manifest /path/to/reviewed_manifest.yaml
```

The builder rejects random/overlapping splits and writes the Agent-R1 parquet inputs under ignored
`data/processed/`. The same `PortfolioLedger` powers trained arms, equal-weight buy-and-hold, all
cash, frozen zero-shot schedules, and any AlphaQuanter-aligned target schedule.

## Training and evaluation

```bash
# Inspect exact upstream command and the three recorded hashes without a GPU.
uv run python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml --dry-run

# H100 smoke run after data/TODO review and W&B login.
uv run python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml

# Serial A1 → A2 → A3 queue on CUDA device 0.
CUDA_VISIBLE_DEVICES=0 scripts/run_queue.sh
```

The watchdog terminates only the active queued PID after eight logged zero-reward updates or eight
identical model-output records, writes the reason to `logs/run_queue/watchdog_events.jsonl`, and lets
the queue continue.

`src/evaluation/` computes cumulative/annualized return, Sharpe, Sortino, maximum drawdown, turnover,
cost ratio, benchmark-adjusted return, tool-call count/success, invalid-call and abstain rates, token
consumption, reasoning rounds, and regime columns. It also produces the three standard figures:
reward-vs-step, tool-call success, and normalized equity vs buy-and-hold.

## TODO(human) — required sign-off

1. **StockBench pool/window/data version:** candidate upstream pool is `GS, MSFT, HD, V, SHW,
   CAT, MCD, UNH, AXP, AMGN, TRV, CRM, JPM, IBM, HON, BA, AMZN, AAPL, PG, JNJ`. The project page
   says 2025-03-03…2025-06-30 and 82 sessions, while the immutable tree contains 83 daily bars per
   ticker. Freeze the 10–20 symbol rule, exact train/valid/test windows, and input checksum manifest.
2. **Agent-R1 commit:** approve the proposed lock
   `b124aa46534cbf2fb8bc8af11405774984c42ac7`; it is newer than `v0.1.0` specifically because the
   tag lacks StepPO.
3. **Costs and reward constants:** confirm 10 bps one-way commission and 5 bps base slippage against
   StockBench/FINSABER; freeze benchmark `B`, sigma's volatility source, epsilon, lambda,
   drawdown/turnover coefficients and turnover/position limits.
4. **Mandatory line review:** sign off data time alignment and `available_at` assignment in
   `src/data/stockbench.py`, the fit window in `src/data/transforms.py`, and the next-open execution
   boundary in `src/env/accounting.py`. Do not treat the project as leakage-complete before this
   review and T2 activation.
