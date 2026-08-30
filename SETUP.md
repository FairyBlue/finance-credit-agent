# H100-NVL setup

These steps target one Linux x86_64 H100-NVL (96 GB) and CPython 3.10.20. The pinned GPU set is
PyTorch 2.7.0+cu128, vLLM 0.9.2, flash-attn 2.8.0.post2, verl 0.7.0, Transformers
4.53.2, PyArrow 20.0.0, and datasets 4.0.0. See [`docs/H100_COMPATIBILITY.md`](docs/H100_COMPATIBILITY.md) for the
wheel hash, evidence boundary, and upgrade policy.

## 1. Driver, repository, and submodule

Confirm that the NVIDIA driver supports the selected CUDA runtime, then clone recursively:

```bash
nvidia-smi
git clone --recurse-submodules \
  https://github.com/HaoPan036/finance-credit-agent.git
cd finance-credit-agent
git submodule update --init --recursive
git -C third_party/agent-r1 rev-parse HEAD
```

The last command must print:

```text
b124aa46534cbf2fb8bc8af11405774984c42ac7
```

Agent-R1 has no installable project metadata. Do not run an editable install and do not patch it;
training launchers add the locked directory to `PYTHONPATH`.

## 2. uv and CPU environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10.20
uv venv --python 3.10.20
uv sync --frozen
uv run --no-sync pytest
```

This path installs no torch, vLLM, flash-attn, or verl and is the supported leakage/accounting test
environment.

## 3. GPU optional extra

The lock encodes torch's CUDA 12.8 index and the CPython 3.10/Linux x86_64 prebuilt flash-attn
wheel. Install the complete frozen environment in one sync:

```bash
uv sync --frozen --extra gpu --extra data
uv pip check --python .venv/bin/python
uv run --no-sync python scripts/verify_h100_stack.py
```

No CUDA Toolkit or system `nvcc` is required because flash-attn is not built from source. Do not
independently upgrade any GPU component or use `uv run` without `--no-sync` after stack acceptance.
The verifier runs a real FlashAttention CUDA kernel; passing it is not evidence that G1 passed.

## 4. Frozen data snapshots

Train/valid use a materialized daily-history snapshot. The primary provider is yfinance 1.6.0 and
Stooq is the allowed fallback. Never fall back silently or read live data during an experiment:
record the provider actually used, its version, UTC retrieval time, and a checksum manifest under
the matching `data.sources.history` fields in `configs/base.yaml` and the reviewed episode manifest.
The inclusive full-session windows are 2023-01-03…2024-10-31 for train and
2024-11-01…2025-02-28 for validation.

For the approved G1 smoke, copy the already frozen history snapshot into `data/history/storage/` and
verify its manifest; do not replace it with a fresh network download. Only when intentionally
creating a new, separately reviewed snapshot should an H100 operator run:

```bash
uv sync --frozen --extra gpu --extra data
uv run --no-sync python scripts/fetch_history.py --provider yfinance
```

Yahoo split-scales historical `Close` even with `auto_adjust=false`. The audited materializer
downloads corporate actions through 2026-08-24, reverses every split ratio later than each bar, and
rounds de-adjusted volume to the nearest share. This reconstructs contemporaneous nominal OHLCV
while keeping `repair=false`. The 10,820-file raw
manifest SHA-256 is `2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90`.
A Stooq snapshot must be generated into a clean location, receive an equivalent price-basis audit,
and be recorded as a different data version; never mix providers.

### Formal-only Final OOS input

StockBench is the final OOS data input, not a G1 smoke prerequisite, runtime dependency, or
submodule. Do not copy it onto the G1 host; prepare it only for the later formal/T2-approved stage:

```bash
git clone https://github.com/ChenYXxxx/stockbench.git data/stockbench
git -C data/stockbench checkout --detach ce8b2b3483590646ad3b650ac8221f43f76fd091
git -C data/stockbench rev-parse HEAD
git -C data/stockbench ls-tree HEAD storage
```

The storage tree must be `f9553f735d6bd2d797541bec519cbd8028adee40`. Its 20 tickers each contain
83 daily observations from 2025-03-03 through 2025-06-30, yielding 82 adjacent return intervals.
The ascending `YYYY-MM-DD<LF>` XNYS session list (final LF included) is 913 bytes with SHA-256
`d272656019d9f7a856444ac903a7a6da9d5377987214d00691600b655df56860`.
Treat that entire window as one indivisible final OOS test: no date may be reused for validation,
early stopping, selection, or tuning. Compute the aggregate SHA-256 over the exact frozen input
manifest. The audited 1,660-file value is
`8b48fc7d50f4ecdbc49e920eb83d72df02b9a56d4365bce7f0db6bb4a342bd1f`. Upstream
`.parquet.sha256` sidecars hash canonical DataFrame contents; the aggregate manifest independently
hashes raw Parquet bytes. The exact aggregate includes only the approved symbols and XNYS sessions:
sort manifest-root-relative POSIX paths by their UTF-8 bytes, then append one
`<path><TAB><lowercase raw-file sha256><LF>` line per file, including the final LF. History uses its
storage directory as manifest root. StockBench uses the repository clone as manifest root, so each
of its 1,660 input paths starts with `storage/`; the resulting UTF-8 manifest is 179,114 bytes.
The launcher checks both source manifests before dispatch, and each rollout worker repeats the raw-byte
check on first source load before it caches a read-only adapter for that process.

Keep all snapshots offline during a run. V1 disables news. Do not ingest or describe an existing
“current fundamentals” cache as point-in-time: fundamentals are allowed only after a field-level PIT
audit establishes immutable `available_at` timestamps. Technical indicators must be computed only
from prices already available at the query's `as_of` time.

The train-only calibration table at `configs/reward_calibration.pending.yaml` fixes
`epsilon=0.0006887293802216064`, `risk_lambda=0.9016432557217152`,
`drawdown_coefficient=0.09659355421265267`, `turnover_coefficient=0.02095348253089393`,
`turnover_limit=1.0`, and `position_limit=0.3`. Its immutable payload hash is frozen. Both
collaborators completed their offline joint review on 2026-08-30; the table records only a generic
reviewer role and timezone-aware timestamp, not personal identities. No automated agent acts as
reviewer. Base config binds its path and immutable payload hash. Calibration uses
overlapping 21-session train windows and 40 deterministic trajectories per window. Each coefficient
is capped at 25% of its designated core's mean magnitude. Position `0.30` and daily turnover `1.00`
are soft reward thresholds, not hard execution limits; turnover is bilateral, so `1.00` is
approximately a 50% one-way rotation.

Verify the frozen payload against the current train inputs without rewriting or signing it:

```bash
uv run --no-sync python scripts/calibrate_reward.py \
  --config configs/base.yaml \
  --check configs/reward_calibration.pending.yaml
```

The repository contains the history-only smoke candidate with four short train episodes. Review it
offline, copy it byte-for-byte, and generate the three derived smoke Parquet files. Future-split
reversal and de-adjusted volume are explicit review priorities:

```bash
# After offline review, copy the frozen candidate byte-for-byte; do not edit its NAV values.
cp configs/episode_manifest.smoke.candidate.yaml \
  configs/episode_manifest.smoke.reviewed.yaml
cmp configs/episode_manifest.smoke.candidate.yaml \
  configs/episode_manifest.smoke.reviewed.yaml
uv run --no-sync python scripts/prepare_episodes.py --config configs/smoke_1p7b.yaml \
  --manifest configs/episode_manifest.smoke.reviewed.yaml

# Do not run scripts/run_t2_probe.py against StockBench again.
# T2 v2 is preregistered to use history train/valid only and is pending implementation.
```

The full formal manifest is a later, T2-approved workflow and is not needed for G1:

```bash
cp configs/episode_manifest.example.yaml configs/episode_manifest.draft.yaml
uv run --no-sync python scripts/freeze_benchmarks.py \
  --draft configs/episode_manifest.draft.yaml \
  --output configs/episode_manifest.reviewed.yaml
uv run --no-sync python scripts/prepare_episodes.py \
  --manifest configs/episode_manifest.reviewed.yaml
```

The one approved legacy T2 access used 81 StockBench OOS samples (`0.530864` aligned vs `0.518519`
shifted) and produced artifact SHA-256
`612fe086d380126190aaa23d37a6ee9670eaaf646f993469fc3836ee0c03f347`. StockBench is closed to
further probe use. T2 v2 will pool per-stock rows, fit on history train, decide on history valid, and
use two ephemeral controls: exact future direction and a 40%-flip future direction (approximately
60% balanced-accuracy strength). Both controls must be detected and the real pipeline must not be
significantly detected. Keep any weak-control miss as a power-boundary result; non-detection is not
proof of absence. V2 remains preregistered pending implementation, so formal launch fails closed
without a legacy fallback; `scripts/run_t2_probe.py` itself exits before reading any configuration
or OOS input. Only the exact, complete hash-bound A1/Qwen3-1.7B wiring-smoke configuration is exempt,
and it cannot support a scientific claim.

## 5. Local-file tracking, rsync, and smoke verification

Copying the frozen history snapshot to H100, installing the GPU extra, and running the
A1/Qwen3-1.7B wiring smoke are authorized. W&B is disabled; structured local evidence is written
under `runs/<run_id>/`. The run still requires H100 host access, a human-reviewed
`configs/episode_manifest.smoke.reviewed.yaml`, and all configured smoke train/valid/test processed
Parquet paths. This authorization does not mark those prerequisites complete. Formal training still
requires a clean, reviewed configuration and a passing implementation of T2 v2; data files remain
ignored.

```bash
export FCA_VALIDATION_COMMIT=REPLACE_WITH_EXACT_40_CHAR_COMMIT

# CPU-safe provisional command/config/hash validation. This is not the final trainer command.
uv run --no-sync python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml --dry-run

# H100 real-tokenizer prompt budget: no model inference or training.
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/verify_context_budget.py \
  --config configs/smoke_1p7b.yaml

# H100 preflight only: no training and no Final OOS access.
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml --preflight-only \
  --expected-commit "$FCA_VALIDATION_COMMIT"

# Verify the external plugin imports against the locked submodule.
PYTHONPATH="$PWD:$PWD/third_party/agent-r1" uv run --no-sync python - <<'PY'
from agent_r1.env import AgentEnv, ToolEnv
from src.env.trading import TradingEnv
assert issubclass(TradingEnv, ToolEnv)
assert "trading" in AgentEnv._registry
print("TradingEnv plugin registered")
PY

# One-GPU Qwen3-1.7B outcome-only smoke with structured local evidence.
# Do not run until the exact P0 validation commit passes stack, tests,
# CONTEXT_BUDGET_PASSED, and PREFLIGHT_PASSED on H100.
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --expected-commit "$FCA_VALIDATION_COMMIT"
```

Set `FCA_VALIDATION_COMMIT` to the exact committed SHA being validated; non-dry launch fails if HEAD,
tracked files, the Agent-R1 submodule, or experiment code/config overlays differ. The queue is
disabled for all arguments and never invokes `uv`; formal training also remains blocked by T2 v2.
`run.seed` is propagated to the main data/actor/critic/vLLM random sources, but distributed GPU
execution is not promised to be bitwise deterministic.

The inherited `tracking.sync` list in `base.yaml` documents the later formal workflow; it is not an
executable G1 transfer command. For G1, copy only history into the host and copy only `runs/` back.
Do not sync `data/stockbench`, `outputs`, or `artifacts` for the smoke.

The run record must contain non-empty `reproducibility.agent_r1_commit`,
`reproducibility.repo_commit`, and `reproducibility.data_version`; the first must equal the submodule
SHA above. Logged rollout, reward, positive backward/optimizer/global-update counts, and a finite
non-zero sampled parameter delta are the GPU smoke acceptance condition. CPU verification cannot
certify CUDA kernels, vLLM rollout, or H100 memory headroom.

Use rsync as the primary file transport; set placeholders for the SSH alias and remote project path.
These commands do not delete the destination and can be resumed:

```bash
export H100_HOST=<ssh-host-alias>
export H100_PROJECT_DIR=<absolute-project-directory-on-h100>

# Mac -> H100: history-only smoke input. Do not expose StockBench to G1.
rsync -a --partial --info=progress2 data/history/storage/ \
  "$H100_HOST:$H100_PROJECT_DIR/data/history/storage/"

# H100 -> Mac: local run evidence after smoke.
rsync -a --partial --info=progress2 \
  "$H100_HOST:$H100_PROJECT_DIR/runs/" runs/
```

If rsync is unavailable, copy the same configured paths with scp. After copying inputs, the existing
raw manifests remain the scientific identity check; no W&B account or manual cryptographic signature
is required.
