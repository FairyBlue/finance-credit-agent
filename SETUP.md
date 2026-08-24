# H100-NVL setup

These steps target one Linux x86_64 H100-NVL (96 GB), Python 3.10, and CUDA 12.x. The pinned GPU set
is PyTorch 2.7.0 from the CUDA 12.8 index, vLLM 0.9.2, flash-attn 2.8.0.post2, and verl 0.7.0.

## 1. Driver, repository, and submodule

Confirm that the NVIDIA driver supports the selected CUDA runtime, then clone recursively:

```bash
nvidia-smi
git clone --recurse-submodules <finance-credit-agent-url>
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
uv python install 3.10
uv sync --frozen
uv run pytest
```

This path installs no torch, vLLM, flash-attn, or verl and is the supported leakage/accounting test
environment.

## 3. GPU optional extra

flash-attn needs torch visible while its extension is built, so install in this order:

```bash
uv sync --frozen
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv sync --frozen --extra gpu --no-build-isolation-package flash-attn
```

Do not upgrade verl independently; Agent-R1 explicitly requires `verl==0.7.0`. vLLM may otherwise
replace torch with its preferred build, so verify the resolved runtime immediately:

```bash
uv run python - <<'PY'
import torch, verl, vllm
import flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("verl", verl.__version__)
print("vllm", vllm.__version__)
print("flash_attn", flash_attn.__version__)
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory > 90 * 1024**3
PY
```

If the optional-extra sync resolves a different torch build, stop and reconcile the vLLM/torch
pair; do not force-install through a broken resolver state.

## 4. Frozen data snapshots

Train/valid use a materialized daily-history snapshot. The primary provider is yfinance 1.6.0 and
Stooq is the allowed fallback. Never fall back silently or read live data during an experiment:
record the provider actually used, its version, UTC retrieval time, and a checksum manifest under
the matching `data.sources.history` fields in `configs/base.yaml` and the reviewed episode manifest.
The inclusive full-session windows are 2023-01-03…2024-10-31 for train and
2024-11-01…2025-02-28 for validation.

```bash
uv sync --extra data
uv run python scripts/fetch_history.py --provider yfinance
```

Yahoo split-scales historical `Close` even with `auto_adjust=false`. The audited materializer
downloads corporate actions through 2026-08-24, reverses every split ratio later than each bar, and
rounds de-adjusted volume to the nearest share. This reconstructs contemporaneous nominal OHLCV
while keeping `repair=false`. The 10,820-file raw
manifest SHA-256 is `2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90`.
A Stooq snapshot must be generated into a clean location, receive an equivalent price-basis audit,
and be recorded as a different data version; never mix providers.

StockBench is the final OOS data input, not a runtime dependency or submodule:

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

Resolve the remaining reward `TODO(human)` values in `configs/base.yaml`. Create and commit a full
reviewed manifest at the exact path configured in base; create a separate smoke manifest with four
short train episodes (at most nine sessions per train/valid episode). Generate strict chronological
inputs into their separate processed files:

```bash
cp configs/episode_manifest.example.yaml configs/episode_manifest.draft.yaml
uv run python scripts/freeze_benchmarks.py \
  --draft configs/episode_manifest.draft.yaml \
  --output configs/episode_manifest.reviewed.yaml
uv run python scripts/prepare_episodes.py \
  --manifest configs/episode_manifest.reviewed.yaml

cp configs/episode_manifest.example.yaml configs/episode_manifest.smoke.draft.yaml
uv run python scripts/freeze_benchmarks.py --config configs/smoke_1p7b.yaml \
  --draft configs/episode_manifest.smoke.draft.yaml \
  --output configs/episode_manifest.smoke.reviewed.yaml
uv run python scripts/prepare_episodes.py --config configs/smoke_1p7b.yaml \
  --manifest configs/episode_manifest.smoke.reviewed.yaml

uv run python scripts/run_t2_probe.py --output artifacts/t2_probe.json
```

The real T2 command currently exits non-zero because the frozen unshifted control is not
significantly above chance (`0.530864` aligned vs `0.518519` shifted). Preserve that failure until a
dated protocol amendment is approved; do not tune against StockBench OOS. Main launchers recompute
and enforce this gate; only the wiring-only smoke run is exempt.

## 5. W&B and smoke verification

Commit the reviewed configuration and generated code before a real run; the launcher rejects a
dirty repository by default. Data files remain ignored.

```bash
uv run wandb login

# CPU-safe command/config/hash validation.
uv run python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml --dry-run

# Verify the external plugin imports against the locked submodule.
PYTHONPATH="$PWD:$PWD/third_party/agent-r1" uv run python - <<'PY'
from agent_r1.env import AgentEnv, ToolEnv
from src.env.trading import TradingEnv
assert issubclass(TradingEnv, ToolEnv)
assert "trading" in AgentEnv._registry
print("TradingEnv plugin registered")
PY

# One-GPU Qwen3-1.7B outcome-only smoke.
CUDA_VISIBLE_DEVICES=0 uv run python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml
```

The W&B run config must contain non-empty `reproducibility.agent_r1_commit`,
`reproducibility.repo_commit`, and `reproducibility.data_version`; the first must equal the submodule
SHA above. A reward curve appearing in W&B is the GPU smoke acceptance condition. CPU verification
cannot certify CUDA kernels, vLLM rollout, or H100 memory headroom.
