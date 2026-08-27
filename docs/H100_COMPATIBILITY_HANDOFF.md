# H100 兼容性修复与上游维护说明

本文用于把 H100 主机上已经定位并验证过的兼容性问题同步给所有合作者，避免后续更新
`pyproject.toml`、`uv.lock`、`SETUP.md` 或 launcher 时重复引入同一批问题。

- 记录日期：2026-08-26
- 目标主机：Linux x86_64、NVIDIA H100 NVL、CPython 3.10
- 当前上游审计基线：`origin/main@79693f61e49d3b0893f4775d9f2a00e130bdf432`
- 本机已验证基线：`6ba699acce32feca8ae2c84152200b3153e3fabd` 加本地兼容补丁
- Agent-R1：`b124aa46534cbf2fb8bc8af11405774984c42ac7`

这批修复与 W&B、数据加密或数字签名无关。新版使用本地文件记录和线下审核的决定可以
保留；这里处理的是 GPU Python 运行栈、配置实际透传和训练队列的 fail-closed 行为。

## 1. 结论

当前主机没有系统级 `nvcc`，但这 **不是当前 pinned H100 环境的 blocker**。PyTorch wheel
自带运行所需的 CUDA 12.8 runtime；`nvcc` 是 CUDA Toolkit 中用于编译 CUDA 源码的编译器，
只有从源码构建 flash-attn 或其他 CUDA extension 时才需要。

当前可行方案是固定 CPython、PyTorch、CUDA ABI 和 flash-attn wheel 的组合，并使用已经发布的
预编译 wheel。该 wheel 已在本机 H100 上执行过真实 FlashAttention CUDA kernel。因此现在不需要
为了本项目单独安装 CUDA Toolkit，也不应让依赖解析器退回 flash-attn 源码包。

在 `origin/main@79693f6` 中，GPU lock 又退回了 flash-attn 的 PyPI source distribution，
同时解析到了未经本项目验证的 PyArrow 23.0.1 和 Transformers 5.10.4。若直接按该版
`SETUP.md` 重建环境，会重新引入已经解决的问题。

## 2. 主机与已验证环境

下列值来自本机实测，不是根据配置推断：

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA H100 NVL，93.09 GiB 可见显存 |
| NVIDIA driver | 580.126.09 |
| `nvidia-smi` 显示的 CUDA 兼容上限 | 13.0（不是当前 runtime 或 Toolkit 版本） |
| 系统 `nvcc` | 未安装 / `PATH` 中不可用 |
| Python | 3.10.20 |
| PyTorch | 2.7.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| vLLM | 0.9.2 |
| verl | 0.7.0 |
| flash-attn | 2.8.0.post2 |
| Transformers | 4.53.2 |
| datasets | 1.2.1（由训练栈传递引入） |
| PyArrow | 20.0.0 |

需要区分三个概念：

1. NVIDIA driver 负责让操作系统和 GPU 通信；`nvidia-smi` 能工作说明驱动路径可用。
2. CUDA runtime 是运行已编译 CUDA 程序所需的库；当前由 `torch==2.7.0+cu128` 提供。
3. `nvcc` 属于 CUDA Toolkit，只在本机编译 CUDA 源码时需要。

因此，“没有 `nvcc`”不等于“没有 CUDA”或“GPU 不能训练”。当前真正的要求是不要把已经
锁定的预编译 wheel 换回源码构建。`nvidia-smi` 中的 CUDA 13.0 只表示 driver 的最高兼容
能力，也不能证明本机安装了 CUDA 13 Toolkit；本项目实际使用的是 PyTorch 报告的 12.8 runtime。

## 3. 环境兼容修复

### 3.1 flash-attn：固定预编译 wheel

上游当前问题：

- `flash-attn==2.8.0.post2` 在 `uv.lock` 中来自 PyPI sdist；
- `SETUP.md` 使用 `--no-build-isolation-package flash-attn` 尝试本机构建；
- 主机没有 `nvcc`，所以该路径不能作为 H100 标准安装方案。

已验证修复：在 `pyproject.toml` 的 `[tool.uv.sources]` 中为 Linux x86_64、CPython 3.10
固定 Dao-AILab 发布的 torch 2.7 / CUDA 12 / C++11 ABI 预编译 wheel：

```toml
flash-attn = [
  { url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl", marker = "python_version == '3.10' and sys_platform == 'linux' and platform_machine == 'x86_64'" },
]
```

lock 中该 wheel 的 SHA-256 应为：

```text
8d7502817aacefb906a06d991b5c5777ecbc541aeefd76fb46466aaa787f0034
```

这个 wheel 只能用于以下组合，不能外推：

- Linux x86_64；
- CPython 3.10；
- PyTorch 2.7；
- CUDA 12；
- C++11 ABI 为 `TRUE`。

若 Python、PyTorch、CUDA、架构或 ABI 改变，必须重新选择匹配 wheel 或建立新的源码构建
方案，不能继续复用此 URL。

### 3.2 PyArrow：固定 20.0.0

上游 lock 当前把 `datasets==1.2.1` 与 `pyarrow==23.0.1` 组合在一起。旧版 datasets 的
`datasets/features.py` 仍定义：

```python
class _ArrayXDExtensionType(pa.PyExtensionType):
```

而 `PyExtensionType` 从 PyArrow 21 起已被移除。依赖解析本身可以成功，但运行时 import 会失败，
所以这属于“lock 合法但程序不可运行”的兼容性问题。

已验证修复：

```toml
"pyarrow==20.0.0"
```

在升级 datasets 之前，不要放宽此 pin。

### 3.3 Transformers：固定 4.53.2

`vllm==0.9.2` 只声明 `transformers>=4.51.1`，没有 major-version 上限。因此重新解析 lock
会自动漂移到 Transformers 5.x；当前上游已经解析到 5.10.4。该 major 版本不属于本项目
Agent-R1/vLLM 0.9.2 的已验证组合。

已验证修复是在 Linux GPU extra 中固定：

```toml
"transformers==4.53.2; sys_platform == 'linux' and platform_machine == 'x86_64'"
```

如需升级 Transformers，应把它视作运行栈升级：同时验证 vLLM、verl、Agent-R1、模型加载、
rollout 和 optimizer update，不能仅因 resolver 能生成 lock 就合并。

### 3.4 推荐安装和运行方式

在上述 pins 已进入 `pyproject.toml` 和 `uv.lock` 后：

```bash
uv sync --frozen
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv sync --frozen --extra gpu --extra data
uv pip check --python .venv/bin/python
```

环境位于 repo 的 `.venv`，不会作为独立环境出现在 `micromamba env list`。可以执行
`source .venv/bin/activate`，也可以不激活并始终通过 `uv run` 调用。

环境验收完成后，实验命令使用：

```bash
uv run --no-sync <command>
```

`--no-sync` 的目的是防止运行命令时根据后来漂移的 lock 隐式改写已验收环境。它不是修复
错误 lock 的替代品；正式合并仍应提交正确的 `pyproject.toml` 和 `uv.lock`。

## 4. Launcher 与队列修复

### 4.1 `run.seed` 必须真实传给上游

当前上游会读取并记录 `run.seed`，但没有把它传给 Agent-R1、verl 和 vLLM。G1 使用 seed 42
时只是碰巧与部分框架默认值一致；以后配置 seed 43 时，元数据可能写 43，而实际 rollout 或
data loader 仍使用默认值。这会产生“看起来可复现，实际不受该配置控制”的结果。

本地已验证的 launcher 修复显式传递：

```text
data.seed
actor_rollout_ref.actor.data_loader_seed
actor_rollout_ref.actor.fsdp_config.seed
+actor_rollout_ref.rollout.engine_kwargs.vllm.seed
```

A2/A3 还需要：

```text
critic.data_loader_seed
critic.model.fsdp_config.seed
```

回归测试必须至少用一个非默认值（例如 123），确认这些值来自 YAML，而不是在 launcher 中
硬编码成 42。

### 4.2 episodes 与 step cap 的新版处理

旧本地补丁曾把 `training.episodes` 透传到 `data.train_max_samples`，并在显式声明时透传
`trainer.total_training_steps`。新版 G1 已通过结构化数据约束替代 smoke 的硬 step cap：

- train candidate 固定为 4 行；
- batch size 为 2；
- epoch 为 1；
- manifest 和 processed-data preflight 会绑定实际行数。

因此 G1 预计产生约 2 次更新，不应为了补 step cap 修改 hash-bound smoke YAML，也不应擅自
加入 `total_training_steps`，否则会破坏已审核配置哈希。通用 launcher 可以保留可选的 episodes /
total-steps 透传，但它不再是新版 G1 的 P0；真实 optimizer/global-step/update-probe 证据才是
smoke 验收依据。

### 4.3 `run_queue.sh` 必须 fail closed

当前上游脚本在无参数时默认排队 A1/A2/A3 正式配置。现在 T2 门禁会阻止它们，但当 T2 完成后，
这个默认行为会重新变成误点火风险。文档警告不能替代脚本保险栓。

推荐并需要测试的行为：

- 无参数：退出非零，不启动任何 `uv`/训练进程；
- 显式传 `configs/smoke_1p7b.yaml`：允许执行 smoke；
- 正式配置：必须显式 `--formal`；
- 正式队列：先对全部配置执行 preflight，全部通过后才启动第一项训练；
- 任一 preflight 或训练失败：保留非零退出码，不继续悄悄运行后续任务。

旧本地 queue 补丁不能原样搬到新版，因为它会连显式 smoke 一起拒绝；应按上述新语义重写。

## 5. 每次环境相关改动的合并要求

凡是修改 GPU 依赖、Python 版本、`pyproject.toml`、`uv.lock` 或 `SETUP.md`，合并前至少检查：

1. `uv lock --check` 和 `uv sync --frozen --extra gpu --extra data` 退出 0；
2. `uv pip check --python .venv/bin/python` 退出 0；
3. 实际版本仍是本文表格中的兼容组合；
4. lock 中 CPython 3.10/Linux x86_64 的 flash-attn 来源是上述 wheel，不是 sdist；
5. `import datasets`、`import pyarrow`、`import transformers`、`import vllm` 全部成功；
6. H100 上执行一次真实 FlashAttention kernel，而不只是 import；
7. 当前提交的完整 pytest、Ruff 和 mypy 通过；
8. launcher 命令测试覆盖非默认 seed；
9. queue 测试证明失败时没有启动训练进程。

FlashAttention 实算可使用：

```bash
uv run --no-sync python - <<'PY'
import torch
from flash_attn import flash_attn_qkvpacked_func

assert torch.__version__ == "2.7.0+cu128"
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
qkv = torch.randn(1, 16, 3, 4, 64, device="cuda", dtype=torch.float16)
output = flash_attn_qkvpacked_func(qkv)
torch.cuda.synchronize()
assert output.shape == (1, 16, 4, 64)
assert torch.isfinite(output).all()
print(torch.cuda.get_device_name(0), output.shape)
PY
```

本机先前的集成工作树结果为 122 passed，且上述 kernel 输出形状为 `(1, 16, 4, 64)`、数值
全部 finite。这个数字只证明当时的本地补丁组合；同步新版 main 后必须重新跑当前测试，不能把
旧测试数当作新版结果。

2026-08-26 的复核证据如下：

| 检查 | 结果 | 证据边界 |
|---|---|---|
| pytest | 122 passed in 21.01s | 仅 `6ba699a` 加本地补丁 |
| Ruff | All checks passed | 同上 |
| mypy | 42 个 source files 无问题 | 同上 |
| `uv lock --check` | 退出 0，237 packages | 本地已修复 lock |
| `uv pip check` | 226 packages compatible | repo `.venv` |
| FlashAttention kernel | `(1,16,4,64)`、FP16、CUDA、全部 finite | H100 实算 |
| 端到端 rollout / optimizer update | 尚未执行 | **不能称为 smoke-passed 或 training-capable** |

在 pins 尚未进入新版 main 前，必须避免：

- 执行上游当前的 flash-attn sdist 源码构建命令；
- 对 `.venv` 运行普通 `uv sync` 或重新解析 `uv lock`；
- 单独升级 torch、vLLM、verl、flash-attn、Transformers 或 PyArrow；
- 把当前 CPython 3.10 wheel 用于 Python 3.11/3.12、非 x86_64 或不同 Torch/ABI；
- 用旧 micromamba `torch 2.9 / vLLM 0.12` compatibility 环境产出正式结果。

## 6. 什么时候才需要安装 `nvcc`

仅在以下情况之一出现时重新评估 CUDA Toolkit / `nvcc`：

- 目标改为 Python 3.11/3.12 或非 x86_64，找不到匹配预编译 wheel；
- 升级 PyTorch、CUDA ABI 或 flash-attn，新的组合没有可验证 wheel；
- 需要修改或调试 flash-attn/CUDA extension 源码；
- 项目新增必须在本机编译的自定义 CUDA kernel。

此时应建立新的环境变更任务，固定 CUDA Toolkit 版本和 `CUDA_HOME`，重新生成 lock，并完成
整套 GPU 验收。单独安装一个任意版本的 `nvcc` 不能解决 PyArrow、Transformers 或 ABI 不匹配，
反而可能制造另一套不可复现环境。

## 7. 当前同步状态

截至本文记录时：

| 修复 | 本机旧基线 | `origin/main@79693f6` | 新版处理建议 |
|---|---|---|---|
| H100 pins / 预编译 flash-attn wheel | 已实现并实算 | 未吸收 | 必须移植后再 sync |
| seed 透传 | 已实现并有回归测试 | 未吸收 | 必须移植 |
| smoke step bound | 旧版显式透传 | 已由 4-row candidate + preflight 结构性替代 | 不改签名 YAML |
| queue 保险栓 | 旧语义已实现 | 未吸收 | 按新版 smoke/formal 语义重写 |

推荐的长期做法是把本文、环境 pins、lock 和对应测试一起合入 `main`。以后合作者更新依赖时，
以本节的合并要求作为门禁，而不是在 H100 上临时修补环境。
