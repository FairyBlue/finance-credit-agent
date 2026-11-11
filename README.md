# finance-credit-agent

An open research log and working implementation for training multi-step tool agents on
financial evidence, verification, and recovery tasks. The research question is whether better
step-level advantages and training-sample allocation can improve actual learning under a limited
interaction budget.

This repository includes the implementation, dated experiment plans, working demonstrations,
cost accounting, and results through **November 11, 2026**. The current experiments have not
established a reliable learning or transfer advantage from the tested credit methods.

## What is implemented

- Financial-report tool environments with retrieval, calculation, evidence checks, conditional
  decisions, and recovery from invalid requests; historical trading environments also include
  point-in-time observations and a shared execution ledger.
- A Qwen3-4B supervised baseline and real RL rollout/update pipelines, including RLOO,
  step-focused updates, invalid-output feedback, MC state baselines, and action-Q continuations.
- Frozen run contracts, model/source identity receipts, paired comparisons, token and GPU-time
  accounting, and CPU tests for the implemented mechanisms.
- Concrete execution examples, including [paired financial-report decisions](runs/m3-financial-evidence-20261030-v2/examples/paired_demo.md),
  plus [historical mechanism investigations](docs/history/CREDIT_ORACLE_DIAGNOSTICS.md).

## Results and current status

The full financial-workflow SFT reached **82/100 Dev and 72/100 on its original held-out tasks**.
These are financial-report tool tasks, not evidence of market trading profits. See the
[task, baselines, and limitations](runs/m3-financial-evidence-20261030-v2/CONCLUSION.md).

The later development-time sampling panel contains 32 tasks from 16 reports, with two samples
per task. Its results are separate from the original 100-task evaluation above:

| Configuration | Complete success / 64 | Evidence |
|---|---:|---|
| Original strong SFT | 43 | [Reference sampling probe](runs/agentic-rl-transfer-probe-20261105-v1/REPORT.md) |
| Narrow-coverage RLOO | 42 | [Reference sampling probe](runs/agentic-rl-transfer-probe-20261105-v1/REPORT.md) |
| V3: wider training coverage | 42 | [Report and conclusion](runs/agentic-rl-coverage-20261105-v3/CONCLUSION.md) |
| V4: calculate-response updates | 41 | [Report and conclusion](runs/agentic-rl-step-focus-20261110-v4/CONCLUSION.md) |
| V5: include invalid-output feedback | 44 | [Report and conclusion](runs/agentic-rl-invalid-feedback-20261110-v5/CONCLUSION.md) |
| V6: action-Q with an extra continuation | 42 | [Report and conclusion](runs/agentic-rl-action-q-20261111-v6/CONCLUSION.md) |

These are limited development comparisons with reused controls and different generation costs,
not independent confirmation or a strict equal-compute benchmark. The V5 follow-up scored
80/100 on Dev against SFT 82/100, so the original SFT remains the base model.
In V6, all six batches, 192 parent exposures, 24 updates, and 64 evaluations completed;
188 of 192 extra same-action continuations returned the same reward as the original successor,
and only 16 target raw advantages changed. The tested configuration is closed without a gain;
this does not establish that RL or credit assignment is ineffective in general.

Earlier [MC learning comparisons](runs/agentic-rl-mc-20261030-v1/REPORT.md) and the
[V2 learning study](runs/agentic-rl-learning-20261030-v2/REPORT.md) also retain their actual
results and costs, including training-panel improvement without established report transfer.
The new financial Test and historical market Final OOS remain unopened in the current series.

## Where to start

- [Current status](docs/CURRENT_STATUS.md): completed experiments, decisions, and evidence links.
- [Financial research brief](RESEARCH_BRIEF.md): the current research direction and evidence boundaries.
- [Research Brief 2.0](RESEARCH_BRIEF2.0.md): a future RAGEN-based alternative; no experiments have started.
  Financial exploration remains the current priority.
- [Document index](docs/README.md): plans, historical summaries, and archived originals.
- [Latest result snapshot](docs/results/2026-11-11-public-update/README.md): the selected V3–V6 reports,
  summaries, and receipts included in this update.
- [Setup](SETUP.md) and [maintenance](docs/COLLABORATION.md): installation and repository workflow.

The repository is [FairyBlue/finance-credit-agent](https://github.com/FairyBlue/finance-credit-agent).
Selected small result artifacts are tracked at their original `runs/` paths. Model weights,
optimizer states, raw datasets, and full rollout collections stay outside the source checkout;
this repository alone does not restore a complete training run.

## Historical trading-credit design and implemented mechanism

The original project tested whether finance-native **step-level credit assignment** improves a
trading agent over outcome-only GRPO and terminal-outcome StepPO. The following sections describe
that historical design, not the current research plan. It composes with the locked
[Agent-R1](https://github.com/AgentR1/Agent-R1) submodule through local extensions.
The [research brief](RESEARCH_BRIEF.md) remains a draft; the [evaluation protocol](EVAL_PROTOCOL.md)
and [M2 screening specification](M2_V1_PREREGISTRATION.md) define their respective boundaries.
The screening specification is not a final preregistration.

The proposed formal ladder is:

| Arm | External training reward | Role |
|---|---|---|
| A1 | Terminal net return | Outcome-only GRPO reference |
| A2 | The same terminal net return | Critic + GAE + GSPO StepPO control |
| A3-M0 | Terminal financial score `A_fin` | Terminal-objective control |
| A3-M1 | Conserved immediate rewards `A_fin / T` | Reward-timing control and required M2 baseline |
| A3-M2 | The same M1 base rewards, plus actor-only post-GAE `eta * S` | Candidate credit method |

This is a proposed formal comparison, not a completed five-arm result. Current Train screening
uses the separately calibrated `benchmark_scale_v2` score `Y`, not the formal ladder's original
`A_fin`. Terminal-only A2 and M1/M2 screening are implemented; legacy dense A2 is a separate
supplementary baseline. State/regime conditioning and counterfactual M3 remain research proposals.

**Implemented M2 pools session-aligned siblings by action family.** Four trajectories from the
same episode occurrence and policy snapshot are aligned by advancing session, then grouped by
`buy`, `sell`, `hold` or `abstain`:

```text
S_t = mean(Y of siblings choosing action family a_t) - mean(Y of all four siblings)
A_t = Whiten_step(GAE(M1 base rewards, critic values) + eta * S_t)
```

The disagreement gate requires at least two legal advancing action families. Residuals are zero on
retrieve, compute, invalid and no-disagreement steps; their base GAE signal remains. `S` enters
neither rewards nor critic targets and is not passed through GAE again. The current plain control
uses centered individual `Y` under the same gate and StepPO stack; it is not standard GRPO.
See [sibling_credit.py](src/rewards/sibling_credit.py) and
[screening_runtime.py](src/agents/screening_runtime.py).

The grouping matches neither portfolio state nor concrete stock/target-weight orders. Its relative
terminal-outcome signal is not an identified causal contribution. The latest CPU probe additionally
shows that credit rankings depend on the evaluation horizon and fixed continuation rule; one-day
returns cannot replace the original financial `Y` without changing the target.

## Completed stages and limits

- Historical G1 and A3-M1 wiring smokes passed; they establish real rollout/optimizer wiring,
  not formal performance. See the [H100/M1 summary](docs/history/FOUNDATIONS_AND_DIAGNOSTICS.md).
- Common initialization, early learning comparisons, 4B seed replication and resource recovery
  are consolidated in [initialization and 4B learning](docs/history/INITIALIZATION_AND_4B_LEARNING.md).
- The [R5 grid](docs/history/RECOVERY_AND_ACTION_POOLING.md) completed training but retained
  29 incomplete greedy evaluations. The subsequent
  [common recovery SFT acceptance](docs/history/RECOVERY_AND_ACTION_POOLING.md) passed
  1,200/1,200 fixed acceptance trajectories. It resolved the tested rejection loops, not all invalid
  requests or general performance issues.
- The latest pooling comparison used the new common SFT/observation version. Its results cannot be
  spliced with R5, and its complete execution does not imply a method advantage.

Dated backup snapshots remain under [September 17](docs/results/2026-09-17-backup/README.md) and
[October 4](docs/results/2026-10-04-backup/README.md). Selected small `runs/` reports and result receipts are tracked; the migration snapshot lists them.
Most run outputs, data and model artifacts remain local evidence and are not included in a source checkout. Historical capacity and service statuses
are snapshots; check actual resources and the selected run's receipts before execution.

## Reproducibility and data boundaries

- Agent-R1: `b124aa46534cbf2fb8bc8af11405774984c42ac7`; verl: exactly `0.7.0`.
  Do not edit `third_party/` or upgrade the GPU stack independently.
- StepPO method reference: [arXiv:2604.18401v4](https://arxiv.org/abs/2604.18401v4).
  The paper pin does not replace the executable commit pin.
- New engineering and training use **Qwen3-4B**. Earlier 1.7B smokes remain historical evidence.
- Train/Valid use frozen daily-history snapshots; Final OOS uses the pinned StockBench source.
  `StockBenchAdapter.get_observations(as_of)` enforces `available_at <= as_of`.
- Shared `PortfolioLedger` accounting applies next-open execution, commission and slippage to
  agents and baselines alike. News and unaudited current-fundamentals caches remain disabled.
- The reviewed formal manifest contains 441 Train, 60 Valid and one indivisible 83-session Final
  OOS episode. T2 v2 passed on history Train/Valid; this does not prove absence of leakage or
  authorize formal training. Legacy OOS probes remain disabled.
- Training records model/data/code identity and structured evidence under `runs/<run_id>/`.
  A run's exact frozen sources and receipts govern its identity; a later `main` commit is not
  automatically validated. Formal training requires clean code and complete preflight.

Full identities, scorer semantics, review requirements and OOS rules remain in
[EVAL_PROTOCOL.md](EVAL_PROTOCOL.md), [SETUP.md](SETUP.md) and
[H100_COMPATIBILITY.md](docs/H100_COMPATIBILITY.md).

## Local setup and project map

```bash
git submodule update --init --recursive
uv sync --frozen
uv run --no-sync pytest
```

The default environment is CPU-only. Use [SETUP.md](SETUP.md) for the frozen GPU environment.
Current jobs must use their own frozen 4B plan and contract; old 1.7B smoke commands are historical.
`scripts/run_queue.sh` remains fail-closed disabled.

| Path | Purpose |
|---|---|
| `configs/` | Arm configs, frozen data assumptions and flow recipes |
| `src/data/`, `src/env/` | Point-in-time adapters, observations and shared trading ledger |
| `src/rewards/` | Formal/scoped screening scores, M1 rewards and sibling residuals |
| `src/agents/` | Training, acceptance, resource coordination and evidence |
| `src/evaluation/` | Shared metrics and baselines |
| `tests/` | Leakage, accounting, mechanisms and runtime checks |
| `docs/` | Current reports, phase summaries and retained frozen contracts |
| `runs/` | Selected tracked key results; other execution evidence stays local |

For any resumption of the historical trading-credit experiment, freeze the selected main method, matched controls, resource accounting and
M2−M1 superiority statistic under the existing protocol. Beating A1/A2 without beating M1 cannot
establish a credit-assignment contribution. M3 requires its own preregistration and schedule-matched
sham control. Current work instead follows the [long-term RL Brief](RESEARCH_BRIEF.md) and the
[current status and experiment decisions](docs/CURRENT_STATUS.md).
