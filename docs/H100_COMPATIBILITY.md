# H100 兼容性与 clean-clone 复现

更新日期：`2026-08-30`

文档定位：本文是长期维护的 H100 版本、安装和证据合同。具体实验必须使用对应 runbook 指定的
精确 commit；本文的通用安装说明不能替代实验任务单。

当前任务单：
[`2026-08-30_H100_G1_P1_VALIDATION_RUNBOOK.md`](2026-08-30_H100_G1_P1_VALIDATION_RUNBOOK.md)。

当前状态：`PARTIALLY_READY`。H100 stack 和完整 preflight 已在验证提交 `8a090b8` 上通过，但
本轮复用了已存在的 pinned `.venv`，没有验证从空 `.venv` 重装。Unicode-safe P0 与 exact
model/offline、episode cap、增量证据、validation identity、direct-only P1 集成仅在本机通过，
尚未冻结为 validation commit 或在 H100 复验，真实 G1 未启动。历史证据见
[`2026-08-27_H100_G1_AUDIT_RESULTS.md`](2026-08-27_H100_G1_AUDIT_RESULTS.md)。

## 1. 身份与证据边界

| 项目 | 值 |
|---|---|
| 旧 H100 验证日期 | `2026-08-26` |
| 旧 H100 验证基线 | `6ba699acce32feca8ae2c84152200b3153e3fabd` 加未提交兼容补丁 |
| 本次实际移植基线 | `9e4bfe3c2d9d6ef36fbd684efcc78c2b1c600cfd` |
| 本次移植提交 | `bcb28e890c0a6cde48f1d5ea2366798a9ff3590e` |
| H100 实际验证代码 | `8a090b86f667216a28a3846520b717348b4a6764` |
| 实机审计记录提交 | `3faea19e688e56a3a9e15aa4b5d3c839fd0da62b` |
| Agent-R1 gitlink | `b124aa46534cbf2fb8bc8af11405774984c42ac7` |

旧环境已经证明指定软件组合可以在一张 H100 NVL 上运行真实 FlashAttention CUDA kernel。
本轮又在验证提交上证明 stack、真实 kernel 和 preflight 可用，但没有从空 `.venv` 重装，也没有
产生 rollout、backward、optimizer update 或 G1 smoke 通过证据。

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
| Triton | `3.3.0` | torch wheel 依赖；fresh `cuda_utils` compiler probe |
| PyArrow | `20.0.0` | 项目核心依赖 |
| datasets | `4.0.0` | Linux x86_64 `gpu` extra；内置本地 Parquet loader |
| host C compiler | GCC `15.2.0=h7be306e_28` | micromamba base prefix；仅编译 Triton CPU driver shim |

`datasets==1.2.1` 读取本地 Parquet 时仍会联网获取旧 loader script，无法满足本项目的
offline 合同；`4.0.0` 将 Parquet builder 内置于 wheel，且与冻结的 PyArrow 20 兼容。
PyArrow 20 继续作为已验证 H100 ABI 的固定版本，不能随 resolver 漂移。
Transformers 4.53.2 是 vLLM 0.9.2/Agent-R1 已验证组合，不能让 resolver
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

当前使用预编译 FlashAttention wheel，所以不需要 CUDA Toolkit 或 `nvcc`。vLLM 0.9.2 的
Triton runtime 首次加载仍会编译一个很小的 CPU `cuda_utils` driver shim，因此主机必须有 C
编译器；H100 micromamba 环境固定使用 `gcc_linux-64=15.2.0=h7be306e_28`，launcher 会解析绝对路径、设置
`CC` 并在 preflight 中实际编译/加载该 shim。这个要求与 CUDA Toolkit/`nvcc` 无关。

只有以下情况才建立新的环境升级任务：

- 改用 Python 3.11/3.12、非 x86_64 或不同 C++ ABI，且没有匹配 wheel；
- 升级 PyTorch、CUDA ABI 或 flash-attn；
- 修改 CUDA extension 源码；
- 新增必须在主机编译的自定义 CUDA kernel。

不要通过安装任意版本 `nvcc` 来掩盖 wheel、PyArrow 或 Transformers 不兼容。

## 5. clean clone 安装流程

当前 lock 已完整编码 torch 2.7.0+cu128 的 CUDA 12.8 index 和 FlashAttention direct wheel，
因此不再需要先 `uv pip install torch` 再 sync。执行已冻结实验时，先从对应 runbook 取得精确
commit，不能 checkout 或 pull 浮动的 `main`。H100 上从空目录执行：

```bash
git clone --recurse-submodules \
  https://github.com/HaoPan036/finance-credit-agent.git
cd finance-credit-agent
git fetch origin
export FCA_VALIDATION_COMMIT=REPLACE_WITH_ACTIVE_RUNBOOK_EXACT_COMMIT
git checkout --detach "$FCA_VALIDATION_COMMIT"
git submodule update --init --recursive

micromamba install -n finance-credit -c conda-forge \
  "python=3.10.20" "gcc_linux-64=15.2.0=h7be306e_28"
FCA_GPU_PYTHON="$(micromamba run -n finance-credit which python)"
uv venv --python "$FCA_GPU_PYTHON"
uv sync --frozen --extra gpu --extra data
uv pip check --python .venv/bin/python
uv run --no-sync python scripts/verify_h100_stack.py
```

这条 micromamba 路线保证 `.venv` 的 `sys.base_prefix` 与 pinned GCC 位于同一前缀，是当前
H100 的复现路径。若改用 `uv python install` 管理的独立 Python，则必须在宿主机安装完整的
C toolchain（Debian/Ubuntu 通常是 `build-essential`）并把可执行编译器的绝对路径放入
`CC`；不能把 GCC 装到一个与 `.venv` 无关的 micromamba 环境后假定它会被自动发现。

最后一条命令完成版本、CUDA、GPU、imports 和真实 FlashAttention kernel 验收。环境验收后，所有
实验命令使用 `uv run --no-sync ...`，防止运行时隐式改变已验收环境。

准备好 history-only processed 文件和缓存模型后，还必须运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/verify_context_budget.py \
  --config configs/smoke_1p7b.yaml
```

该命令使用 Qwen 缓存 snapshot 的真实 tokenizer 审计 4 个 Train 和 1 个 Valid candidate 的完整
有界轨迹，并同时覆盖 `planned-ascii`、`legal-unicode-max`、多持仓/混合大结果路径和
`invalid-unicode-unknown-tool` 四类场景；
`CONTEXT_BUDGET_PASSED` 只证明提示词预算门禁通过，不等于 G1 passed。

## 6. H100 验收脚本

`scripts/verify_h100_stack.py` fail closed 检查：

- CPython 3.10.20、Linux x86_64；
- torch 2.7.0+cu128 与 `torch.version.cuda == "12.8"`；
- 可执行 host C compiler，且 Triton `cuda_utils` 可实际编译并加载；
- CUDA 可用，设备名包含 `H100`；
- vLLM、verl、Triton、flash-attn、Transformers、PyArrow、datasets 的 import 和精确版本；
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

G1 只允许 direct launcher；先由 runbook 提供 exact commit：

```bash
export FCA_VALIDATION_COMMIT=REPLACE_WITH_EXACT_40_CHAR_COMMIT
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --expected-commit "$FCA_VALIDATION_COMMIT"
```

`scripts/run_queue.sh` 对所有参数都 fail closed，且不得调用 `uv`：

```bash
scripts/run_queue.sh configs/smoke_1p7b.yaml
scripts/run_queue.sh --formal configs/arm_a1.yaml
```

在 queue 的 stage-aware process-group、信号和退出码语义重新设计并验收前，不得恢复。T2 v2 已于
2026-09-01 通过并由 hash-bound artifact 放行；正式 launcher 对 artifact 或其他门禁的任何漂移仍
应 fail closed。

## 9. G1 规模与验收边界

不要修改 hash-bound `configs/smoke_1p7b.yaml` 来增加 `total_training_steps`。G1 规模由 4 行 train
candidate、batch size 2、epoch 1 和 processed-data preflight 共同约束；调度细节可能影响 update
次数，不断言恰好两次。

G1 至少要求：

- `optimizer_step_count >= 1`；
- `observed_update_record_count >= 1`（仅表示探针记录存在，不是独立 trainer step）；
- gradient finite；
- parameter update detected 且 delta finite/non-zero；
- Final OOS 未访问。

环境验收、模型加载或 FlashAttention kernel 成功都不能替代这些证据。

## 10. 升级禁区与维护门禁

不要单独升级 torch、vLLM、verl、flash-attn、Transformers、datasets、PyArrow 或 Python。任何运行栈变更
必须作为一个整体重新 lock，并至少完成：

```bash
uv lock --check
uv sync --frozen --extra gpu --extra data
uv pip check --python .venv/bin/python
uv run --no-sync python scripts/verify_h100_stack.py
uv run --no-sync pytest -q
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/verify_context_budget.py \
  --config configs/smoke_1p7b.yaml
uv run --no-sync ruff check src tests scripts runtime_hooks
uv run --no-sync mypy src tests
git diff --check
```

同时检查非默认 seed 回归、queue fail-closed 和 W&B residue。不得使用旧 micromamba
torch 2.9/vLLM 0.12 环境产出本项目结果。

## 11. 当前证据

| 检查 | 旧基线 + 本地补丁 | 本次移植提交 |
|---|---|---|
| FlashAttention H100 kernel | `PASSED`，2026-08-26 | `PASSED`，验证代码 `8a090b8`，复用 pinned `.venv` |
| 从空 `.venv` clean-clone 安装 | 未记录 | `NOT TESTED` |
| 完整 preflight-only | 未执行 | `PASSED`，未启动训练，Final OOS 未访问 |
| Prompt-budget P0 修复 | 不适用 | `LOCAL PASS / H100 UNVERIFIED`；Train/Valid 四类路径待实机复验 |
| P1 启动与证据门禁 | 不适用 | `LOCAL PASS / H100 UNVERIFIED`；exact snapshot/offline/cap/identity/direct-only 已实现 |
| 端到端 rollout / optimizer update | 未执行 | `NOT RUN`，等待 P0/P1 的 H100 context-budget 与 preflight 复验 |
| G1 smoke | 未执行 | `BLOCKED` |
| 当前 pytest | 旧结果仅供历史参考 | `141 passed`（当前 Mac） |
| Ruff / mypy | 旧结果仅供历史参考 | 通过；H100 审计中的 mypy 检查 44 个 source files |
| lock / local sync / pip check | 旧结果仅供历史参考 | 通过；236 packages resolved，57 packages compatible |

项目状态保持 `PARTIALLY_READY`。下一步是冻结包含 P0/P1 集成的新 validation commit，然后在 H100
重新执行 stack、测试、context-budget 和 preflight 门禁；通过前不得启动 G1。
