# H100 G1 冒烟前技术验收结果（2026-08-30）

状态更新：`V0.7 HUMAN REVIEW PASSED / READY FOR FINAL-MAIN PREFLIGHT`

## 技术结论

H100 上 A1/Qwen3-1.7B wiring smoke 的代码、环境、数据、prompt、命令透传和本地证据门禁均已通过。
完整 `--preflight-only` 在精确代码提交
`91c98fc751e8e55d0437a77c45b53322199d96cd` 上退出 0，并明确输出：

```text
PREFLIGHT_PASSED
No training was started.
Final OOS not accessed.
```

本次没有启动训练，也没有访问 StockBench Final OOS。报告生成后，两位合作者已于 2026-08-30
按 [`EVAL_PROTOCOL.md`](../EVAL_PROTOCOL.md) v0.8 在线下完成人工审核，用户确认 v0.7 通过并授权
G1 smoke。最终 `main` exact commit 仍须重新通过完整 preflight；真实 smoke 尚未运行，因此当前
不能写 `G1 PASSED`。

本文件是上述已验收提交的后继纯文档提交。Git 身份门禁绑定完整 commit，所以合并或 cherry-pick
产生新的 SHA 后，必须以新的外部确认 40 位 commit 重跑完整 preflight；不得把本报告提交本身误称为
已经通过 preflight 的代码身份。

## 验收范围与身份

| 项目 | 验收值 |
| --- | --- |
| 仓库 | `HaoPan036/finance-credit-agent` |
| 验收分支 | `fix/h100-pre-smoke-blockers-20260830` |
| 已验收代码 commit | `91c98fc751e8e55d0437a77c45b53322199d96cd` |
| 集成基底 | `origin/main` @ `eab437362dc26e6942cd45a2ed2d3667c0b5907f` |
| Agent-R1 gitlink/checkout | `b124aa46534cbf2fb8bc8af11405774984c42ac7` |
| 模型 | `Qwen/Qwen3-1.7B` |
| 模型 revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 模式 | A1 单卡 wiring smoke 的 `preflight_only` |
| 不在范围内 | 实际训练、A2/A3、4B 正式训练、StockBench Final OOS、T2 v2 解锁 |

这里的 “technically preflight ready” 仅表示从 direct launcher 点火前的机器门禁已经通过；它不证明
训练发生、参数更新有效或实验结果具有科学有效性。

已验收实现由三个有序提交组成：

1. `464bf3d` — 清除 H100 pre-smoke validation blockers；
2. `439761f` — 精确允许惰性 `egg-info` 元数据；
3. `91c98fc` — 对齐根级工具缓存并增加拒绝反例。

## 关键验收证据

| 门禁 | 实测结果 | 结论 |
| --- | --- | --- |
| Git 身份 | exact repo commit；tracked tree clean；Agent-R1 exact pin；无 experiment overlay | 通过 |
| Python/依赖 | Python 3.10.20；226 个已安装包兼容 | 通过 |
| GPU 栈 | H100 NVL；torch 2.7.0+cu128；CUDA 12.8；vLLM 0.9.2；verl 0.7.0 | 通过 |
| FlashAttention | 2.8.0.post2 预编译 wheel；真实 GPU kernel 输出 finite，shape `[1,16,4,64]` | 通过 |
| 静态/单元测试 | pytest `147 passed`；Ruff 通过；mypy 46 个 source files 通过 | 通过 |
| lock | `uv lock --check` 解析 236 个冻结包 | 通过 |
| history 原始源 | 10,820 个文件；20 个 symbol 各 541 sessions | 通过 |
| history manifest | `2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90` | 通过 |
| processed smoke | train=4、valid=1、test=1；逐文件 SHA-256 已记录 | 通过 |
| prompt/context | 620 cases；train=4、valid=1；4 种 scenario；最大 3045/4096 prompt tokens | 通过 |
| exact model | 最终命令只使用 revision 对应的本地 snapshot，不使用浮动 Hub repo ID | 通过 |
| offline | trainer 强制 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` | 通过 |
| 上游训练上限 | `data.train_max_samples=4`、`agent.max_steps=32`、`trainer.total_epochs=1` | 通过 |
| seed 透传 | data、actor dataloader、actor FSDP、vLLM 均为 42 | 通过 |
| Final OOS 隔离 | `final_oos_accessed=false`，smoke test split 来自 history shadow window | 通过 |
| queue 保险栓 | `scripts/run_queue.sh` 对所有参数 fail closed，实测退出 2 且未启动进程 | 通过 |
| 完整 preflight | exit/return code 0，`command_state=FINAL` | 通过 |

### 环境明细

[`scripts/verify_h100_stack.py`](../scripts/verify_h100_stack.py) 实测返回
`H100_STACK_VERIFIED`：

| 组件 | 版本/结果 |
| --- | --- |
| GPU | NVIDIA H100 NVL |
| Python | 3.10.20 |
| torch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| flash-attn | 2.8.0.post2 |
| vLLM | 0.9.2 |
| verl | 0.7.0 |
| Transformers | 4.53.2 |
| PyArrow | 20.0.0 |
| datasets | 1.2.1 |

本机没有系统 `nvcc` 不再是 blocker：当前 flash-attn 来自与冻结 torch/CUDA ABI 匹配的预编译
wheel，且已经通过真实 H100 kernel 调用；本轮没有从源码构建 flash-attn，也不需要安装 CUDA
Toolkit。

## 本分支完成的修复

### 1. validation identity 能正确拒绝覆盖，同时允许惰性本地工件

- Git 输出只移除行尾，不再误删 clean submodule status 的首空格。
- 同时验证 superproject HEAD、tracked tree、gitlink、Agent-R1 checkout HEAD 和 submodule worktree。
- 使用 NUL 分隔的 Git 路径扫描覆盖整个仓库与 Agent-R1，而不是只检查少数源码目录。
- 继承配置的 `smoke_1p7b.yaml`、`base.yaml`、`arm_a1.yaml` 都必须是 tracked source。
- trainer 环境不再继承父进程的任意 `PYTHONPATH`。
- 精确允许 reviewed manifest、data/runs、`.venv`、工具缓存以及 5 种惰性
  `*.egg-info` 元数据；仍拒绝 `sitecustomize.py`、普通 Python overlay、独立 `.pyc`、
  `entry_points.txt`、submodule overlay 等可能改变执行的内容。

`*.egg-info` 误报的原失败路径已不再复现；正常工件和拒绝反例均有真实临时 Git
superproject/submodule 回归测试。该门禁用于可信协作者工作区的可复现性检查，不是用来抵御已经控制
Python 解释器或宿主机的攻击者的安全沙箱。

### 2. preflight 失败与成功证据不再混淆

- non-dry run 从最早期身份检查开始即创建本地 recorder；early failure 也写机器可读 summary。
- 候选命令只写入 `provisional_command.txt`。
- 所有 preflight 门禁通过后才生成 `command.txt`，并把 `command_state` 标为 `FINAL`。
- 后期失败不会抹掉前面已经完成的 commit、模型、数据、manifest、prompt 和命令证据。
- 最终命令在通过 exact-revision 检查后改写为本地 snapshot 路径。

### 3. smoke 控制参数已真实进入上游命令

- `training.episodes=4` 落到 `data.train_max_samples=4`，与 train parquet 的 4 行一致。
- `agent.max_steps=32` 和 `trainer.total_epochs=1` 形成有界 smoke。
- seed 真实传入 data、actor dataloader、actor FSDP 与 vLLM，不再只是 YAML 中声明。
- queue 完全禁用，G1 只能使用带 `--expected-commit` 的 direct launcher。

### 4. 协议与人工签字语义已纠正

- [`configs/reward_calibration.pending.yaml`](../configs/reward_calibration.pending.yaml) 已记录
  `status: signed`、通用 reviewer 角色和带时区时间，不公开个人身份。
- immutable reward payload hash 保持：
  `1b46a81268fe4b45e065c49fde8bf9c0641131676668c8e35e4f884747464430`。
- 自动 agent 只复算 payload/hash 和机器门禁，不再冒充人工 reviewer。
- SETUP、EVAL_PROTOCOL、handoff、compatibility 文档统一为本地文件交接、离线运行与 exact commit
  语义。

## 完整 preflight 记录

权威本地 run：

```text
runs/20260830T063320568686Z-smoke-a1-qwen3-1p7b-91c98fc7/
```

该目录位于 H100 工作区并按协议被 Git 忽略；如合作者需要原始 JSON、日志或最终命令，应通过双方约定
的本地文件拷贝方式交接，而不是把整个 `runs/` 提交进源码仓库。

| 证据 | 值 |
| --- | --- |
| run status | `PASSED` |
| wall clock | 616.543 秒 |
| canonical merged config SHA-256 | `2ec544cb4e539cdbc2d3301b255384d3fdd770f5801f8d0ba3194081ac39d844` |
| resolved YAML bytes SHA-256 | `b826d04cc150c3b9e54728f6e7b669c2a48691a9af36f58ca8efe5b19d103753` |
| final command SHA-256 | `860b00a3acd32bbb10cd05fead10fff8e81e7da8f4eaf81a7cd99191b59d1a05` |
| data manifest record SHA-256 | `8a6b7fec32b47f0566e88a23e88bf76a32966d89909d2fa45fff6f1b5c661770` |
| optimizer/backward count | `0 / 0`，符合 preflight-only |
| peak GPU memory | `0`，没有启动 trainer/model inference |

`summary.json` 与 `artifacts/preflight_evidence.json` 又经过独立程序化断言，确认：

- status/exit/return code 与 `command_state=FINAL` 一致；
- repo、Agent-R1、配置来源、模型 revision/snapshot 和全部哈希一致；
- history 文件数、session 数、processed rows 和 prompt cases 一致；
- 最终命令不存在 `actor_rollout_ref.model.path=Qwen/Qwen3-1.7B` 浮动路径；
- 没有 optimizer step、backward 或 Final OOS 访问。

## 验收方法与稳健性检查

在 clean tracked tree 和 exact commit 上依次执行：

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync mypy src tests
uv lock --check
uv pip check --python .venv/bin/python
uv run --no-sync python scripts/verify_h100_stack.py

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --preflight-only \
  --expected-commit 91c98fc751e8e55d0437a77c45b53322199d96cd
```

除正常通过路径外，测试还覆盖 expected-commit mismatch、tracked dirty、submodule mismatch、
repo/submodule overlay、hostile parent `PYTHONPATH`、early failure summary、部分 preflight 证据保留、
命令 provisional/final 状态以及 train row/episode cap 一致性。

## 限制与剩余门禁

1. **本报告不是一次训练结果。** preflight 没有 rollout、backward、optimizer step 或参数变化，不能
   用来证明“模型学到了东西”。
2. **合并会改变 commit。** 合并、rebase、cherry-pick 或任何文档变更都会产生新 SHA；实际点火前必须
   用新 SHA 重跑全量测试、stack/context 和 preflight。
3. **正式训练仍受 T2 v2 阻断。** G1 wiring smoke 通过不解除 A1/A2/A3 正式实验的 T2 门槛。
4. **身份门禁不是宿主机安全边界。** 它防止可信协作者工作区中的未提交执行覆盖和证据漂移，不防御
   已控制 shell、Python 启动过程或 H100 主机的攻击者。

## 合作者下一步

1. 合并到目标分支后，用新的完整 40 位 commit 重跑本报告的测试、stack 和完整 preflight。
2. 只有再次得到三行成功输出，才使用 direct launcher 启动一次 A1/Qwen3-1.7B
   wiring smoke；不要使用 [`scripts/run_queue.sh`](../scripts/run_queue.sh)。
3. smoke 完成后保留并本地拷贝 `runs/<run_id>/`、控制台日志、峰值显存和总耗时；smoke 通过仍不解除
   T2 v2 正式训练阻断。

实际点火命令必须绑定合并后的新 SHA：

```bash
export FCA_VALIDATION_COMMIT=<EXTERNALLY_REVIEWED_40_CHAR_COMMIT>
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --expected-commit "$FCA_VALIDATION_COMMIT"
```

当前没有剩余的已知人工或机器级 pre-smoke blocker；只需对最终 `main` exact commit 重新验收。
