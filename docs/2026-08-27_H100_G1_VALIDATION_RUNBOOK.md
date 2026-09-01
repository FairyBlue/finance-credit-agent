# 2026-08-27 H100 G1 验证执行手册

> **状态更新（2026-08-27）：** 这是绑定 `8a090b8` 的原始执行任务单。实机已完成 H100 stack
> 和完整 preflight，但在点火前发现 prompt-budget P0，因此真实 G1 未运行且继续 `BLOCKED`。
> 当前结论、证据和修复顺序以
> [`2026-08-27_H100_G1_AUDIT_RESULTS.md`](2026-08-27_H100_G1_AUDIT_RESULTS.md) 为准。
> 下文“不要继续修改代码”和“下一步直接运行 G1”的原始指令已经失效；解决 P0、冻结新 commit
> 并重新通过验收前，不得启动 G1。

文档定位：本文件保留本轮验证的原始步骤和验收标准，不是长期环境维护入口。长期版本约束见
[`H100_COMPATIBILITY.md`](H100_COMPATIBILITY.md)。

这轮已经完成了一个关键转折：**H100 兼容性修复从某台机器上的临时补丁，变成了 GitHub
`main` 中可审计、可安装、可复现的正式实现。**

## 原始状态（执行前）

建议拆成两个维度：

| 维度 | 状态 | 解释 |
| --- | --- | --- |
| 仓库交接能力 | `READY` | 合作者已经可以从 GitHub 获取代码、文档、lock 和运行说明 |
| H100 实验执行能力 | `UNVERIFIED`（执行前） | 当时尚未完成 H100 stack、preflight 或真实 optimizer update |
| 项目综合状态 | `PARTIALLY_READY` | 当前判断准确 |

那 10 个新增 pytest 也是合理的。每一个都对应真实实验风险，包括 seed 虚假透传、队列误点火、
formal 未授权、失败后继续训练、依赖漂移。这里没有为了数量增加测试。

## 原始冻结指令（已由 P0 审计结果取代）

下一阶段应冻结：

```text
8a090b86f667216a28a3846520b717348b4a6764
```

先在 H100 上验证这个精确提交。

不要直接执行：

```bash
git checkout main
git pull
```

因为 `main` 可能在验证期间继续前进。实验验证必须绑定确定的代码版本。

建议在 H100 上建立新的 clean clone：

```bash
git clone --recurse-submodules \
  https://github.com/HaoPan036/finance-credit-agent.git \
  finance-credit-agent-g1-8a090b8

cd finance-credit-agent-g1-8a090b8

git fetch origin
git checkout --detach 8a090b86f667216a28a3846520b717348b4a6764
git submodule update --init --recursive
```

先确认运行身份：

```bash
git rev-parse HEAD
git status --short
git submodule status
```

预期：

```text
HEAD = 8a090b86f667216a28a3846520b717348b4a6764
working tree clean
Agent-R1 = b124aa46534cbf2fb8bc8af11405774984c42ac7
```

## H100 上的执行顺序

### 第一关：安装冻结环境

```bash
uv python install 3.10.20
uv venv --python 3.10.20
uv sync --frozen --extra gpu --extra data
uv pip check --python .venv/bin/python
```

安装完成后，先检查解析结果：

```bash
uv run --no-sync python - <<'PY'
import torch
import pyarrow
import transformers

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("pyarrow:", pyarrow.__version__)
print("transformers:", transformers.__version__)
PY
```

预期关键值：

```text
torch = 2.7.0+cu128
torch CUDA = 12.8
CUDA available = True
PyArrow = 20.0.0
Transformers = 4.53.2
```

### 第二关：验证真实 GPU kernel

```bash
uv run --no-sync python scripts/verify_h100_stack.py
```

这一步通过，才能说明：

```text
Python 可以导入训练栈
PyTorch 能识别 H100
预编译 flash-attn wheel 与当前 ABI 匹配
FlashAttention CUDA kernel 真实执行
输出形状正确
输出全部 finite
```

单纯看到 `import flash_attn` 成功还不够。

### 第三关：运行当前提交的测试

在 H100 环境中至少再运行一次完整 pytest：

```bash
uv run --no-sync pytest -q
```

Mac 上的 `136 passed` 保护代码逻辑。H100 上重新运行可以补充验证 Linux、Python 3.10 和 GPU
extra 没有破坏普通功能。

Ruff 和 mypy 可以按照 [`docs/EXPERIMENT_HANDOFF.md`](EXPERIMENT_HANDOFF.md) 中已经确认的命令
执行。

不需要在每次训练前重复完整 136 项测试。它们适合在以下时点运行：

```text
首次建立 H100 环境
代码 commit 改变
依赖或 lock 改变
launcher、数据或 reward 逻辑改变
正式实验开始前
```

### 第四关：运行 G1 preflight

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --preflight-only
```

这里应重点确认：

```text
PREFLIGHT_PASSED
No training was started.
Final OOS not accessed.
模型 revision 正确
数据 manifest 正确
4 个训练 episode 正确
shadow test 来自 Valid
GPU 和输出目录可用
Git revision 与 submodule revision 正确
```

只要 preflight 仍然引用 StockBench Final OOS，就必须停止，不能继续启动 smoke。

## Preflight 通过后才能运行真实 G1

使用 [`docs/EXPERIMENT_HANDOFF.md`](EXPERIMENT_HANDOFF.md) 中记录的唯一 G1 命令。不要临时手写
另一套参数，也不要修改已经绑定哈希的 smoke YAML。

G1 运行期间只观察系统接线证据：

```text
4 个 episode 被正确加载
模型完成 rollout
action 能被解析
交易环境能够执行
reward、loss、gradient 全部 finite
backward_count >= 1
optimizer_step_count >= 1
global_step >= 1
gradient_nonzero = true
parameter_delta_l2 > 0
parameter_update_detected = true
运行退出码为 0
没有残留 GPU 训练进程
Final OOS accessed = false
```

收益、Sharpe 和交易表现都不属于这一步的验收标准。

## G1 结束后的状态判断

### 情况一：环境脚本失败

例如：

```text
版本不匹配
FlashAttention kernel 失败
CUDA 不可用
wheel ABI 不兼容
```

状态应保持：

```text
H100 stack: FAILED
G1: BLOCKED
```

先处理环境，不启动 preflight 和训练。

### 情况二：环境通过，preflight 失败

状态应为：

```text
H100 stack: PASSED
G1: BLOCKED
```

此时通常是数据、manifest、配置、模型路径或 Final OOS 隔离问题。

### 情况三：preflight 通过，训练失败

状态应为：

```text
H100 stack: PASSED
G1 preflight: PASSED
G1 smoke: FAILED
```

保留完整 `runs/<run_id>/summary.json`、`events.log` 和错误信息，然后从第一个失败阶段定位。

### 情况四：真实 optimizer update 成功

状态可以更新为：

```text
H100 stack: PASSED
G1 preflight: PASSED
G1 wiring smoke: PASSED
```

这时才可以说训练链路已经接通。

## G1 通过后需要同步的内容

不要修改运行对应的 commit。实验依旧绑定：

```text
8a090b86f667216a28a3846520b717348b4a6764
```

随后在新的文档提交中更新：

```text
docs/EXPERIMENT_HANDOFF.md
docs/H100_COMPATIBILITY.md
```

记录：

```text
H100 验证日期
被验证的代码 commit
Agent-R1 commit
运行命令
run_id
环境验收结果
preflight 结果
optimizer step 数
global step
参数更新证据
运行产物路径
Final OOS 是否访问
G1 最终状态
下一项任务
```

文档提交会产生新的 `HEAD`，但 G1 结果仍明确归属于 `8a090b8`。

## G1 通过后的项目路线

此时仓库可以标记为：

```text
COLLABORATOR_READY
```

这里表达的是合作者可以稳定接手并运行实验。正式研究训练仍然受到后续门禁约束：

```text
G1 wiring smoke
    ↓
T2 v2 PASSED（2026-09-01）
    ↓
解决实验臂因果混杂
    ↓
实现 M2 或 M3
    ↓
1.7B pilot
    ↓
proposal-level experiment
    ↓
冻结后开放 Final OOS
```

这是执行前记录的原始优先任务：

> **在 H100 的全新目录中 checkout `8a090b8`，运行环境验收脚本，然后运行 G1 preflight。**

上述 stack 和 preflight 已完成。当前唯一优先任务改为修复审计报告中的 prompt-budget P0，冻结
新的 validation commit 并重新验收；在此之前不得启动 G1。
