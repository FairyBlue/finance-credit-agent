# H100-NVL setup

These steps target one Linux x86_64 H100-NVL (96 GB), Python 3.10, and CUDA 12.x. There are two
deliberately separate environment paths:

1. **Pinned/reproducible:** PyTorch 2.7.0 from the CUDA 12.8 index, vLLM 0.9.2,
   flash-attn 2.8.0.post2, and verl 0.7.0. Use this path for reported experiments.
2. **Local compatibility:** clone an already working Agent-R1 micromamba environment to unblock
   tests and an H100 smoke run. This path inherits the source environment's unpinned pip state and
   is not an experiment lock.

Do not combine results from the two paths without recording the complete package versions. The
local compatibility path must pass a real one-GPU rollout before it is considered training-capable.

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

## 2. Pinned uv CPU environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --frozen
uv run pytest
```

This path installs no torch, vLLM, flash-attn, or verl and is the supported leakage/accounting test
environment.

## 3. Pinned uv GPU optional extra

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

## 4. Local micromamba compatibility path

Use this only when an existing `agentr1` environment has already exercised the GPU on the same
host. A micromamba clone copies conda-managed packages, but it does **not** copy packages installed
with pip. Reinstall the source environment's pip snapshot before adding this project's dependencies:

```bash
cd /path/to/finance-credit-agent

micromamba create -n finance-credit --clone agentr1 -y
micromamba run -n agentr1 python -m pip freeze > /tmp/agentr1-pip-freeze.txt
micromamba run -n finance-credit python -m pip install -r /tmp/agentr1-pip-freeze.txt
micromamba run -n finance-credit python -m pip install -e . --group dev
```

The editable install above applies only to `finance-credit-agent`; do not install or modify the
locked `third_party/agent-r1` submodule. Activate and verify the copied runtime:

```bash
micromamba activate finance-credit
export PYTHONPATH="$PWD:$PWD/third_party/agent-r1${PYTHONPATH:+:$PYTHONPATH}"

python -m pip check
python -m pytest -q
python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml --dry-run

python - <<'PY'
import importlib.util
import torch
import verl
import vllm

assert torch.cuda.is_available()
properties = torch.cuda.get_device_properties(0)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("verl", verl.__version__)
print("gpu", properties.name, "memory GiB", round(properties.total_memory / 1024**3, 2))
print("flash_attn installed", importlib.util.find_spec("flash_attn") is not None)
assert properties.total_memory > 90 * 1024**3
PY
```

The compatibility environment verified on 2026-08-24 contained Python 3.10.20,
torch 2.9.0+cu128, vLLM 0.12.0, and verl 0.7.0 on an NVIDIA H100 NVL with 93.09 GiB visible
memory. It had neither flash-attn nor `nvcc`. On that host, `pip check`, the full CPU test suite,
the launcher dry-run, the TradingEnv plugin import, and the Agent-R1 trainer import passed. This is
pre-smoke evidence only: it does not certify vLLM rollout, CUDA kernels, backward training, or memory
headroom, and it must not replace the pinned environment for reported experiments.

## 5. StockBench offline snapshot

StockBench is data input, not a runtime dependency or submodule:

```bash
git clone https://github.com/ChenYXxxx/stockbench.git data/stockbench
git -C data/stockbench checkout --detach ce8b2b3483590646ad3b650ac8221f43f76fd091
git -C data/stockbench rev-parse HEAD
git -C data/stockbench ls-tree HEAD storage
```

The storage tree must be `f9553f735d6bd2d797541bec519cbd8028adee40`. Keep StockBench in offline
mode; this project does not call Polygon/Finnhub and does not ingest financial/indicator caches.

Resolve the `TODO(human)` values in `configs/base.yaml`, create a reviewed manifest from
`configs/episode_manifest.example.yaml`, and generate strict chronological inputs:

```bash
uv run python scripts/prepare_episodes.py --manifest /path/to/reviewed_manifest.yaml
```

## 6. W&B and smoke verification

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

For the activated local micromamba path, run the same checks without the `uv run` prefix:

```bash
wandb login

python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml --dry-run

PYTHONPATH="$PWD:$PWD/third_party/agent-r1" python - <<'PY'
from agent_r1.env import AgentEnv, ToolEnv
from src.env.trading import TradingEnv
assert issubclass(TradingEnv, ToolEnv)
assert "trading" in AgentEnv._registry
print("TradingEnv plugin registered")
PY

CUDA_VISIBLE_DEVICES=0 python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml
```

The non-dry smoke requires the reviewed TODO values, processed train/validation parquet files, model
access, and W&B authentication. Do not describe the local compatibility environment as verified
until this command produces a reward curve without import, rollout, CUDA-kernel, or OOM failures.

The W&B run config must contain non-empty `reproducibility.agent_r1_commit`,
`reproducibility.repo_commit`, and `reproducibility.data_version`; the first must equal the submodule
SHA above. A reward curve appearing in W&B is the GPU smoke acceptance condition. CPU verification
cannot certify CUDA kernels, vLLM rollout, or H100 memory headroom.
