# finance-credit-agent

Start with the [unified research plan](RESEARCH_BRIEF.md) and
[current status](docs/CURRENT_STATUS.md), then use the
[document index](docs/README.md) for results and background. Completed plans and detailed
reports are consolidated by topic; exact historical originals are in the [archive](docs/archive/README.md).

The long-term question is how to estimate useful step-level advantages and allocate limited
training samples for multi-step tool agents in an Agent-R1-style RL setting. Financial evidence,
verification and recovery provide the current task environment. The October 30 financial
workflow SFT reached 82/100 Dev and 72/100 held-out tasks. A subsequent paired probe found some
local information effects but did not pass its preset training gate; no four-arm training followed.
The [completed first MC cycle](docs/2026-10-30_AGENTIC_RL_THREE_STEP_PLAN.md) connected independent
MC state baselines to real updates and compared ordinary RLOO, uniform MC and one allocation
candidate from the same strong SFT. Dev success was 82/100 for SFT,
80/100 for RLOO, 82/100 for uniform MC and 81/100 for priority MC. The preset gate stopped
further confirmation; the new Test was not evaluated. See the [research review](runs/agentic-rl-mc-20261030-v1/RESEARCH_REVIEW.md).
Step-credit learning benefits remain unproven; the historical market Final OOS remains unopened.

The [three-step plan v2](docs/2026-10-30_AGENTIC_RL_THREE_STEP_PLAN_V2.md) is complete. High-LR
RLOO improved the selected Train panel from 41/64 to 59/64, with report holdout still 22/32
and Dev 80/100 versus SFT 82/100. The final targeted MC comparison scored 40/64 against
ordinary RLOO 42/64; both used 96 parent trajectories and 12 updates, with 14 MC baselines.
A [supplementary holdout sampling check](docs/2026-11-05_HELDOUT_SAMPLING_PROBE.md) also completed:
high-LR RLOO 42/64 against SFT 43/64. These limited results establish local learning responsiveness,
but no transfer or MC learning gain. The [complete report](runs/agentic-rl-learning-20261030-v2/REPORT.md)
and [research review](runs/agentic-rl-learning-20261030-v2/RESEARCH_REVIEW.md) preserve scope and costs.
The combined generation cost was 40,053,793 tokens and GPU-stage wall time about 2.91 hours,
inside the original 44M/12-hour total allowance. New Test and historical market Final OOS remain unopened.

The Brief holds long-term goals; dated plans in `docs/` hold rolling tasks and budgets. Review
progress against the Brief after each completed three-step cycle or when key evidence changes.

Installation is in [SETUP.md](SETUP.md); repository maintenance is described in
[COLLABORATION.md](docs/COLLABORATION.md). The private repository is
[FairyBlue/finance-credit-agent](https://github.com/FairyBlue/finance-credit-agent).
Selected key reports and result receipts are included in the
[November 5 migration snapshot](docs/results/2026-11-05-backup/README.md).

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
separate [rolling three-step plan v2](docs/2026-10-30_AGENTIC_RL_THREE_STEP_PLAN_V2.md).
