# finance-credit-agent

**New collaborator:** read
[`docs/EXPERIMENT_HANDOFF.md`](docs/EXPERIMENT_HANDOFF.md) first. It is the self-contained handoff
for the research motivation, current gate status, frozen inputs, clean-clone setup, H100 G1 and
A3-M1 wiring smokes, acceptance evidence, troubleshooting, and the work that remains afterwards.

For day-to-day work between the two maintainers, see
[`docs/COLLABORATION.md`](docs/COLLABORATION.md). GitHub automatically publishes a readable report
when a pull request changes and when a change reaches `main`.

Research scaffold for testing whether finance-native **step-level credit assignment** improves a
tool-using trading agent over outcome-only GRPO and generic StepPO. The project composes with
[Agent-R1](https://github.com/AgentR1/Agent-R1) as a locked submodule, keeps every local change under
`src/`, and treats look-ahead leakage prevention as the first executable requirement.

The three method classes use four formal configurations:

| Configuration | Upstream optimizer | Reward/credit signal and role |
|---|---|---|
| A1 | GRPO (`adv_estimator=grpo`) | Original episode return, emitted only at trajectory end |
| A2 | Agent-R1 StepPO (`adv_estimator=gae`, critic + GSPO loss) | Generic per-environment-step reward |
| A3-M1 | The same StepPO setup as A3-main | `A_fin` plus M1; required formal ablation, not an eligible main method |
| A3-M2 or A3-M3 | The same StepPO setup as A2/A3-M1 | `A_fin` plus the preregistered finance-native allocation; A3 main method |

M1 is frozen as **uniform immediate-step reward v1**. For a completed trajectory with `T` realized
Agent-R1 interaction steps, the terminal `A_fin` is retroactively assigned as immediate rewards
`r_t = A_fin / T`; the final step absorbs floating-point residual so the rewards sum exactly to
`A_fin`. These values are StepPO inputs, not direct or uniformly broadcast optimizer advantages;
StepPO subsequently computes advantages with step-level GAE. A3-M1 must receive a formal,
budget-matched run as the required reward-shaping ablation, but it cannot serve as A3-main. M2
state×regime grouping and M3 sampled counterfactual information contribution retain explicit
interfaces and deliberately raise `NotImplementedError`; A3-main must implement and preregister at
least one of them.

The A3 runtime fails closed on empty or truncated trajectories and records one JSONL credit record
per completed trajectory. The launcher validates finite `A_fin`, the actual rewritten step rewards,
exact conservation, non-zero Train/Valid signals, and one-to-one trajectory identity agreement with
the deterministic rollout-seed evidence. This evidence contract is shared by M1 and future M2/M3;
mechanism-specific checks remain separate.

The A3-M1 implementation and its single-H100 wiring smoke have now passed. The accepted run executed
one real trainer step with a critic update, an actor update, and post-update validation; all three
Train/Valid trajectories conserved `A_fin` exactly and Final OOS remained closed. See
[`docs/2026-09-02_H100_A3_M1_WIRING_SMOKE_RESULTS.md`](docs/2026-09-02_H100_A3_M1_WIRING_SMOKE_RESULTS.md)
for the exact code identity, implementation changes, evidence hashes, and limitations. This is not a
formal A3-M1 result and does not unblock A3-main.

A finance-native credit-assignment claim additionally requires A3-main to pass a preregistered M1
superiority gate under the same model, data, StepPO optimizer, seed set, hyperparameter-selection
rules, and budget accounting as A3-M1. The exact superiority statistic remains to be frozen by the
two collaborators before formal training; no value is selected here. Beating A1/A2 without beating
A3-M1 supports at most an `A_fin`/reward-shaping result, not the credit-assignment claim.

## Reproducibility pins

- Agent-R1 gitlink: `b124aa46534cbf2fb8bc8af11405774984c42ac7` (2026-07-21). This untagged
  default-branch commit contains StepPO and was explicitly approved in the 2026-08-24 protocol
  amendment; the only tag, `v0.1.0`, does not contain StepPO.
- StepPO paper/specification pin: [arXiv:2604.18401v4](https://arxiv.org/abs/2604.18401v4)
  (2026-06-21). This citation pin fixes the referenced method semantics; it does not replace the
  executable Agent-R1 commit pin above.
- verl: exactly `0.7.0`, following Agent-R1. Do not upgrade it independently.
- StockBench OOS source: [`ChenYXxxx/stockbench`](https://github.com/ChenYXxxx/stockbench),
  repo commit `ce8b2b3483590646ad3b650ac8221f43f76fd091`, storage tree
  `f9553f735d6bd2d797541bec519cbd8028adee40`. Its 1,660-file OOS raw-manifest SHA-256 is
  `8b48fc7d50f4ecdbc49e920eb83d72df02b9a56d4365bce7f0db6bb4a342bd1f`.
  The 83-line canonical XNYS session list has SHA-256
  `d272656019d9f7a856444ac903a7a6da9d5377987214d00691600b655df56860`.

Every training command injects `agent_r1_commit`, `repo_commit`, and `data_version` into the Hydra
configuration and writes structured local evidence under `runs/<run_id>/`. Formal training refuses
a dirty/uncommitted repository; the exact hash-bound G1 smoke records dirty state but is intended to
run from a clean GitHub checkout.

## Project map

```text
configs/                  frozen assumptions, arm configs, and Agent-R1 flow recipes
src/data/                 frozen daily-data adapters and mandatory as_of gate
src/env/                  next-open ledger, BaseTool registrations, TradingEnv plugin
src/rewards/              A_fin and M1/M2/M3 strategy interface
src/agents/               three training entrypoints, hash capture, Agent-R1 flows
src/agents/credit_evidence.py A3 step-credit persistence and fail-closed trajectory summary
src/evaluation/           shared metrics/baselines and three standard figures
tests/                    leakage, accounting, reward, environment, and metric specifications
scripts/run_queue.sh      fail-closed disabled queue; G1 uses the direct launcher
scripts/watchdog.py       collapsed-reward/repeated-output process watchdog
scripts/fetch_history.py  freeze yfinance/Stooq train+valid inputs and checksums
scripts/build_smoke_manifest.py generate the deterministic history-only G1 manifest draft
scripts/build_formal_manifest.py generate the frozen 441/60/1 formal manifest draft
scripts/freeze_benchmarks.py generate shared-ledger benchmark NAV + checksum
configs/episode_manifest.smoke.candidate.yaml frozen G1 candidate; reviewed copy stays local/ignored
configs/smoke_a3_m1_1p7b.yaml hash-bound A3-M1 wiring-only resource configuration
configs/episode_manifest.reviewed.yaml collaborator-reviewed 441/60/1 formal manifest
scripts/run_t2_probe.py   disabled legacy OOS entrypoint retained for audit history
scripts/run_t2_v2_probe.py run the frozen history-only T2 v2 probe and write bound evidence
configs/t2_v2_result.json repository T2 v2 aggregate result bound by SHA-256
scripts/verify_h100_stack.py pinned-version and real FlashAttention CUDA acceptance
runs/                     ignored local preflight/training evidence, one directory per run
third_party/agent-r1/     read-only locked submodule
```

`RESEARCH_BRIEF.md` and `EVAL_PROTOCOL.md` are frozen inputs and are not generated by code.
`EVAL_PROTOCOL.md` preserves the substantive v0.1 protocol and its dated amendments. V0.3 freezes
the train-only reward rule, event-anchored regimes, T2 v2, and the narrow G1 exemption; v0.6 replaces
the old full-OOS smoke with the Valid-derived shadow test and requires structured local and direct
optimizer-update evidence; v0.7 defines the exact-model, prompt, evidence and validation-identity
gates used by G1 and was approved offline by both collaborators on 2026-08-30; v0.9 records the
history-only T2 v2 implementation and its 2026-09-01 execution evidence; v1.0 records the reviewed
formal manifest, v1.1 freezes the four-configuration A3/M1 comparison, and v1.2 records the A3-M1
implementation and successful wiring smoke without authorizing formal training. No agent or digital
signature substitutes for human review. New protocol changes require another recorded amendment.

## CPU quick start

```bash
git submodule update --init --recursive
uv sync --frozen
uv run --no-sync pytest
uv run --no-sync python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml --dry-run
```

The default `uv sync` is CPU-only. GPU libraries live exclusively in the `gpu` optional extra; see
[`SETUP.md`](SETUP.md) for clean-clone commands and
[`docs/H100_COMPATIBILITY.md`](docs/H100_COMPATIBILITY.md) for the pinned wheel/hash contract and
H100 evidence boundary.

## Data safety model

`StockBenchAdapter.get_observations(as_of)` is the only public observation read method. It always
enforces `available_at <= as_of`; no raw/all-data accessor exists. The frozen protocol uses:

- an offline yfinance 1.6.0 snapshot (Stooq is the allowed fallback) for the full-session train and
  validation windows, with actual provider, version, retrieval time, and checksums recorded; and
- `storage/parquet/{TICKER}/day/*.parquet` from the pinned StockBench tree only for the indivisible
  2025-03-03…2025-06-30 final OOS window.

V1 disables news. `retrieve` is limited to gated market data, audited point-in-time fundamentals,
and technical indicators computed causally from already available historical prices. Existing
“current fundamentals” caches are **not** considered PIT and remain disabled unless every field and
its `available_at` semantics pass an audit. The project never invokes StockBench's network-capable
`DataHub` during a run. Daily Parquet sidecars are verified with StockBench's canonical
DataFrame-content hash; data-version manifests independently hash raw Parquet bytes. StockBench
code is Apache-2.0, but Polygon/Finnhub-derived data has no clearly separate redistribution license
in the repository; keep data external under ignored `data/` and obtain a human/legal confirmation.

Train/valid data may overlap Qwen3 pretraining and therefore carries parametric-contamination risk.
That development setup follows common practice in the field but is not claimed as strict OOS; only
the untouched post-cutoff StockBench window supports final OOS claims.

Current leakage-test status:

| Test | Executable status |
|---|---|
| T1 future-price corruption | Implemented; historical retrieval and numeric feature bytes invariant |
| T2 shifted labels | V2 passed on frozen history Train/Valid; bound artifact is revalidated by the formal gate |
| T3 train-only statistics | Implemented; full-sample fits raise `FullSampleFitError` |
| T4 future-document deletion | Implemented; historical retrieval bytes invariant |

## Preparing audited episodes

Yahoo retroactively split-scales historical `Close` even when `auto_adjust=false`. The frozen
yfinance snapshot therefore reverses every later split ratio through 2026-08-24 and rounds
de-adjusted volume to the nearest share to reconstruct contemporaneous nominal OHLCV
(`repair=false`). It has 10,820 files (541 XNYS sessions for each of
20 tickers) and raw-manifest SHA-256
`2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90`. To reproduce it:

```bash
uv sync --frozen --extra gpu --extra data
uv run --no-sync python scripts/fetch_history.py --provider yfinance
```

The deterministic train-only reward calibration has produced
`configs/reward_calibration.pending.yaml`; its values are `epsilon=0.0006887293802216064`,
`risk_lambda=0.9016432557217152`, `drawdown_coefficient=0.09659355421265267`,
`turnover_coefficient=0.02095348253089393`, `turnover_limit=1.0`, and `position_limit=0.3`.
The immutable payload is frozen and both collaborators completed their offline joint review on
2026-08-30; the table records a generic reviewer role and timestamp without publishing personal
identity. No automated agent acts as reviewer. Base config binds only its path and immutable payload
hash. It uses overlapping 21-session train windows and 40
fixed trajectories per window (20 single-stock holds plus 20 cyclic midpoint rotations). Each
penalty is calibrated at no more than 25% of its specified core's mean magnitude. The two limits
are soft reward thresholds: turnover is measured daily and bilaterally, so `1.00` is approximately
a 50% one-way rotation; position is maximum single-name close weight.

Re-verify that review metadata did not alter the frozen payload or train inputs:

```bash
uv run --no-sync python scripts/calibrate_reward.py \
  --config configs/base.yaml \
  --check configs/reward_calibration.pending.yaml
```

The formal manifest generator deterministically resolves 441 overlapping 21-session Train episodes,
60 overlapping 21-session Valid episodes, and one indivisible 83-session Final OOS episode. The
freezer writes equal-weight benchmark NAV through the shared `PortfolioLedger`:

```bash
uv run --no-sync python scripts/build_formal_manifest.py \
  --config configs/base.yaml \
  --output /tmp/finance-credit-agent-formal-manifest-draft.yaml
uv run --no-sync python scripts/freeze_benchmarks.py \
  --draft /tmp/finance-credit-agent-formal-manifest-draft.yaml \
  --output /tmp/finance-credit-agent-formal-manifest-candidate.yaml

# The committed reviewed manifest was promoted with a status-only change after
# both collaborators confirmed the minimal review fields.
uv run --no-sync python scripts/prepare_episodes.py \
  --manifest configs/episode_manifest.reviewed.yaml

# The repository contains the frozen smoke candidate approved offline on 2026-08-30.
# Copy it byte-for-byte to the ignored reviewed path; it never contains Final OOS.
cp configs/episode_manifest.smoke.candidate.yaml \
  configs/episode_manifest.smoke.reviewed.yaml
cmp configs/episode_manifest.smoke.candidate.yaml \
  configs/episode_manifest.smoke.reviewed.yaml
uv run --no-sync python scripts/prepare_episodes.py --config configs/smoke_1p7b.yaml \
  --manifest configs/episode_manifest.smoke.reviewed.yaml

# Do not rerun the legacy T2 probe against StockBench OOS. T2 v2 uses history Train/Valid only.
```

The reviewed manifest raw SHA-256 is
`4a8ff4f3ad8c9305c03aab17f44ebda5c4d493c22b6610a67a70ba238a5a54a3`; its benchmark series
SHA-256 is `a86507384e6806db84066a0cd286cc143664ecddb3c2b851f63562ea45b117b8`.
The local processed result was verified at 441/60/1 rows. See
[`docs/2026-09-01_FORMAL_MANIFEST_RESULTS.md`](docs/2026-09-01_FORMAL_MANIFEST_RESULTS.md) for the
candidate-to-reviewed proof and artifact hashes.

The builder rejects random/overlapping splits and writes the Agent-R1 parquet inputs under ignored
`data/processed/`. The smoke shadow is the nine Valid sessions from `2024-11-14` through
`2024-11-26`, carries wiring-only provenance, and cannot access Final OOS. At launch, every
train/valid row—and every smoke shadow test row—must exactly match the committed manifest,
including dates, opens, benchmark NAV, horizon, and the complete allowed `env_kwargs` key set. The
launcher also recomputes every reviewed benchmark NAV from the frozen raw bars before training. The
same `PortfolioLedger` powers trained arms, equal-weight buy-and-hold, all cash, frozen zero-shot
schedules, and any AlphaQuanter-aligned target schedule.

The legacy real T2 run used its one approved StockBench OOS access and is retained only as historical
evidence: 81 samples, `0.530864` aligned accuracy, `0.518519` shifted accuracy, and artifact SHA-256
`612fe086d380126190aaa23d37a6ee9670eaaf646f993469fc3836ee0c03f347`. StockBench remains closed to
probe reruns, feature selection, and power analysis.

T2 v2 passed on 2026-09-01 using only frozen history Train/Valid data. It pooled per-stock samples,
fit on Train, decided on Valid, detected both ephemeral positive controls, and did not significantly
detect the real pipeline. The real/strong/weak balanced accuracies were respectively `0.499357`,
`1.0`, and `0.595495`; the repository aggregate artifact is
[`configs/t2_v2_result.json`](configs/t2_v2_result.json), with raw-byte SHA-256
`aff81d0af5fb458768abd741264ce312a7c0767884069824a7e309286ea27436`. The formal launcher recomputes
the frozen protocol identity and three-condition verdict from that artifact and fails closed on any
drift. This clears the T2 blocker without proving absence of leakage or authorizing formal training;
see [`docs/2026-09-01_T2_V2_RESULTS.md`](docs/2026-09-01_T2_V2_RESULTS.md). The legacy CLI still exits
before reading configuration or OOS input.

For both sources, the raw manifest includes only the approved symbols and XNYS sessions. Relative
POSIX paths are sorted by UTF-8 bytes; each line is
`<path><TAB><lowercase raw-file sha256><LF>`, including the final LF. History paths are relative to
`data/history/storage`; StockBench paths are relative to `data/stockbench`, hence begin with
`storage/`. A non-dry launcher recomputes both manifests before starting Agent-R1, and every rollout
worker independently recomputes its source manifest on first load before caching the adapter.

## Training and evaluation

```bash
export FCA_VALIDATION_COMMIT=REPLACE_WITH_EXACT_40_CHAR_COMMIT

# Inspect the provisional Hub-ID command and three hashes without a GPU.
# The authoritative final command is written only after preflight resolves the exact local snapshot.
uv run --no-sync python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml --dry-run

# H100-only preflight; performs no training and never accesses Final OOS.
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml --preflight-only \
  --expected-commit "$FCA_VALIDATION_COMMIT"

# H100 smoke run with structured local evidence under runs/<run_id>/.
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --expected-commit "$FCA_VALIDATION_COMMIT"
```

`scripts/run_queue.sh` is disabled fail-closed for every argument and never invokes `uv`. Until its
stage-aware process-group and signal semantics are redesigned and reviewed, use only the direct
launcher commands above. T2 v2 and the formal manifest gate have passed, but formal runs still
require clean code/data identity, complete preflight, and the remaining arm-specific scientific work.

The protocol-defined G1 is the A1/Qwen3-1.7B smoke and is already `PASSED` at commit `2d48bf6...`.
The A3-M1 wiring smoke is a separate follow-up and is also `PASSED` at commit `fb07e727...`; neither
run is formal scientific training.

`src/evaluation/` computes cumulative/annualized return, Sharpe, Sortino, maximum drawdown, turnover,
cost ratio, benchmark-adjusted return, tool-call count/success, invalid-call and abstain rates, token
consumption, reasoning rounds, and regime columns. It also produces the three standard figures:
reward-vs-step, tool-call success, and normalized equity vs buy-and-hold.

The frozen OOS regimes contain 22 sessions (`2025-03-03`…`2025-04-01`), 5 sessions for the
tariff-crash segment (`2025-04-02`…`2025-04-08`), and 56 sessions (`2025-04-09`…`2025-06-30`).
The five-session tariff-crash metrics are descriptive only and must carry no significance claim.

## Remaining required work

1. **A3 scientific implementation:** A3-M1 implementation and wiring validation are complete.
   Implement and preregister M2 or M3 as A3-main, retain A3-M1 as the required formal budget-matched
   ablation, and freeze the M1 superiority statistic before any formal training. M1 cannot be
   presented as the main method, and A3-main cannot support a credit-assignment claim unless it
   passes that preregistered gate.
2. **Disabled modalities:** keep current fundamentals and news disabled unless a later protocol
   amendment defines and reviews their point-in-time fields.
3. **Human line review:** use the offline checklist in `docs/EXPERIMENT_HANDOFF.md` for future-split
   reversal/volume handling, data timing, train-only transforms, and next-open execution. No digital
   signature or separate cryptographic signing workflow is required.
4. **Formal-run readiness:** regenerate the ignored processed Parquet from the committed reviewed
   manifest on the execution host, freeze a clean commit, verify the frozen history and model
   identities on the H100 host, run complete preflight, and preserve `runs/` evidence. G1 has already
   passed and must not be represented as formal scientific training.
