# H100 兼容性与 clean-clone 复现

更新日期：`2026-08-27`

当前状态：`PARTIALLY_READY`。本文记录的兼容性移植后来已在 H100 NVL 通过软件栈、真实
FlashAttention kernel 与 preflight。最新 prompt-budget 修复 commit `e42fdda` 也通过完整 H100
preflight；真实 rollout/optimizer/G1 仍未运行。详见
[`2026-08-28_H100_G1_FIX_TEST.md`](2026-08-28_H100_G1_FIX_TEST.md)。

## 1. 身份与证据边界

| 项目 | 值 |
|---|---|
| 旧 H100 验证日期 | `2026-08-26` |
| 旧 H100 验证基线 | `6ba699acce32feca8ae2c84152200b3153e3fabd` 加未提交兼容补丁 |
| 本次实际移植基线 | `9e4bfe3c2d9d6ef36fbd684efcc78c2b1c600cfd` |
| 本次移植提交 | `bcb28e890c0a6cde48f1d5ea2366798a9ff3590e` |
| Agent-R1 gitlink | `b124aa46534cbf2fb8bc8af11405774984c42ac7` |

旧环境已经证明指定软件组合可以在一张 H100 NVL 上运行真实 FlashAttention CUDA kernel。
它没有证明本次移植提交、端到端 rollout、optimizer update 或 G1 smoke 已通过。

## 2. 冻结的 H100 运行栈

| 项目 | 固定值 | 当前移植的约束位置 |
|---|---:|---|
| 平台 | Linux x86_64 | FlashAttention marker 与验收脚本 |
| Python | CPython `3.10.20` | `.python-version` 与验收脚本 |
| PyTorch | `2.7.0+cu128` | CUDA 12.8 显式 uv index 与 lock |
| PyTorch CUDA runtime | `12.8` | PyTorch wheel 与验收脚本 |
| vLLM | `0.9.2` | `gpu` extra |
| verl | `0.7.0` | `gpu` extra，与 Agent-R1 一致 |
| flash-attn | `2.8.0.post2` | 预编译 direct wheel |
| Transformers | `4.53.2` | Linux x86_64 `gpu` extra |
| PyArrow | `20.0.0` | 项目核心依赖 |
| datasets | `1.2.1` | GPU 栈传递依赖；验收脚本检查 import 与版本 |

PyArrow 20 是兼容性上限，不是随意选择：`datasets==1.2.1` 仍使用从 PyArrow 21 起删除的
`PyExtensionType`。Transformers 4.53.2 是 vLLM 0.9.2/Agent-R1 已验证组合，不能让 resolver
自动漂移到 5.x。

## 3. FlashAttention wheel 合同

Linux x86_64、CPython 3.10 使用：

```text
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

SHA-256：

```text
8d7502817aacefb906a06d991b5c5777ecbc541aeefd76fb46466aaa787f0034
```

`uv.lock` 中该 package 必须只有上述 direct wheel，不能出现 flash-attn sdist。该 wheel 仅适用于
CPython 3.10、Linux x86_64、PyTorch 2.7、CUDA 12 ABI、C++11 ABI TRUE；不能用于其他组合。

## 4. 为什么不安装 nvcc

旧 H100 主机没有系统 `nvcc`，但 PyTorch 2.7.0+cu128 自带执行已编译 GPU 程序所需的 CUDA
12.8 runtime。`nvidia-smi` 显示的是 driver 的最高 CUDA 兼容能力，不是本项目 runtime，也不
代表安装了 CUDA Toolkit。

当前使用预编译 FlashAttention wheel，所以不需要 CUDA Toolkit 或 `nvcc`。只有以下情况才建立
新的环境升级任务：

- 改用 Python 3.11/3.12、非 x86_64 或不同 C++ ABI，且没有匹配 wheel；
- 升级 PyTorch、CUDA ABI 或 flash-attn；
- 修改 CUDA extension 源码；
- 新增必须在主机编译的自定义 CUDA kernel。

不要通过安装任意版本 `nvcc` 来掩盖 wheel、PyArrow 或 Transformers 不兼容。

## 5. clean clone 安装流程

当前 lock 已完整编码 torch 2.7.0+cu128 的 CUDA 12.8 index 和 FlashAttention direct wheel，
因此不再需要先 `uv pip install torch` 再 sync。H100 上从空目录执行：

```bash
git clone --recurse-submodules \
  https://github.com/HaoPan036/finance-credit-agent.git
cd finance-credit-agent
git checkout main
git pull --ff-only origin main
git submodule update --init --recursive

uv python install 3.10.20
uv venv --python 3.10.20
uv sync --frozen --extra gpu --extra data
uv pip check --python .venv/bin/python
uv run --no-sync python scripts/verify_h100_stack.py
```

最后一条命令完成版本、CUDA、GPU、imports 和真实 FlashAttention kernel 验收。环境验收后，所有
实验命令使用 `uv run --no-sync ...`，防止运行时隐式改变已验收环境。

## 6. H100 验收脚本

`scripts/verify_h100_stack.py` fail closed 检查：

- CPython 3.10.20、Linux x86_64；
- torch 2.7.0+cu128 与 `torch.version.cuda == "12.8"`；
- CUDA 可用，设备名包含 `H100`；
- vLLM、verl、flash-attn、Transformers、PyArrow、datasets 的 import 和精确版本；
- FlashAttention CUDA kernel 输出 shape 为 `(1, 16, 4, 64)`；
- kernel 输出全部 finite。

通过时输出 `"status": "H100_STACK_VERIFIED"`。这只证明运行栈和 kernel，不等于 G1 passed。

## 7. seed 与可复现性

launcher 将 YAML 的 `run.seed` 传入：

```text
data.seed
actor_rollout_ref.actor.data_loader_seed
actor_rollout_ref.actor.fsdp_config.seed
+actor_rollout_ref.rollout.engine_kwargs.vllm.seed
critic.data_loader_seed                  # A2/A3
critic.model.fsdp_config.seed            # A2/A3
```

测试使用非默认值 123，确认最终上游命令来自 YAML。seed 控制主要数据、actor、critic 和 vLLM
随机源，但分布式 GPU 调度、并行归约与某些 CUDA kernel 仍可能产生差异，因此不承诺 bitwise
deterministic；复现判断应结合 seed、Git、lock、数据和模型 revision。

verl 的上游依赖图仍可能让 `wandb` package 出现在 lock 中，但本项目不启用它：
tracking backend 固定为 `local_files`，上游命令只使用 console logger，preflight 会拒绝任何
wandb 参数。lock 中的传递 package 不等于重新启用 W&B。

## 8. queue 安全语义

允许的 smoke 调用：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_queue.sh configs/smoke_1p7b.yaml
```

正式队列必须显式授权：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_queue.sh --formal \
  configs/arm_a1.yaml configs/arm_a2.yaml configs/arm_a3.yaml
```

无参数或未带 `--formal` 的正式配置会在调用 uv 前退出非零。所有配置先完成 preflight；全部通过
后才启动第一项训练。任一 preflight 或训练失败都会保留非零退出码并停止后续任务。当前 T2 v2
尚未实现，所以正式 preflight 仍应 fail closed。

## 9. G1 规模与验收边界

不要修改 hash-bound `configs/smoke_1p7b.yaml` 来增加 `total_training_steps`。G1 规模由 4 行 train
candidate、batch size 2、epoch 1 和 processed-data preflight 共同约束；调度细节可能影响 update
次数，不断言恰好两次。

G1 至少要求：

- `optimizer_step_count >= 1`；
- `global_step >= 1`；
- gradient finite；
- parameter update detected 且 delta finite/non-zero；
- Final OOS 未访问。

环境验收、模型加载或 FlashAttention kernel 成功都不能替代这些证据。

## 10. 升级禁区与维护门禁

不要单独升级 torch、vLLM、verl、flash-attn、Transformers、PyArrow 或 Python。任何运行栈变更
必须作为一个整体重新 lock，并至少完成：

```bash
uv lock --check
uv sync --frozen --extra gpu --extra data
uv pip check --python .venv/bin/python
uv run --no-sync python scripts/verify_h100_stack.py
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts runtime_hooks
uv run --no-sync mypy src tests
git diff --check
```

同时检查非默认 seed 回归、queue fail-closed 和 W&B residue。不得使用旧 micromamba
torch 2.9/vLLM 0.12 环境产出本项目结果。

## 11. 当前证据

| 检查 | 旧基线 + 本地补丁 | 本次移植提交 |
|---|---|---|
| FlashAttention H100 kernel | `PASSED`，2026-08-26 | `PASSED`，2026-08-27 实机复核 |
| 端到端 rollout / optimizer update | 未执行 | 未执行 |
| G1 preflight-only | 未执行 | `PASSED`，`e42fdda` exact commit，未启动训练 |
| G1 smoke | 未执行 | 未执行 |
| 当前 pytest | 旧结果仅供历史参考 | `146 passed`（H100） |
| Ruff / mypy | 旧结果仅供历史参考 | 通过；mypy 检查 45 个 source files |
| lock / local sync / pip check | 旧结果仅供历史参考 | lock 通过，236 packages；clean-clone 重装仍未测 |

项目状态仍为 `PARTIALLY_READY`：preflight 通过不等于 G1 通过，且 v0.7 协议变更仍待线下审核。
