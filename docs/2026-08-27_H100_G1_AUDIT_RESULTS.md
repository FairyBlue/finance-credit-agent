# 2026-08-27 H100 G1 实机审计结果

更新日期：`2026-08-27`

本地修复补充日期：`2026-08-29`。第 1～7 节保留 2026-08-27 实机审计的原始事实；后续进展见
第 8 节，不能把本地修复结果回写成旧 validation commit 已在 H100 通过。

本文记录 H100 上对 G1 wiring smoke 的实机复核结果。它补充
[`2026-08-27_H100_G1_VALIDATION_RUNBOOK.md`](2026-08-27_H100_G1_VALIDATION_RUNBOOK.md)，
并以实际执行证据修正其中“尚未在 H100 验证”的状态。

## 1. 结论

| 验收项 | 状态 | 证据边界 |
|---|---|---|
| 冻结 H100 软件栈与真实 FlashAttention kernel | `PASSED` | 在 H100 NVL 上完成版本、CUDA、ABI、shape 与 finite 检查 |
| 当前代码测试与静态检查 | `PASSED` | 136 pytest、Ruff、mypy、lock 与 diff 检查通过 |
| history 数据准备与完整 preflight-only | `PASSED` | 10,820 个文件，manifest 匹配；preflight 退出 0，未启动训练 |
| 从空 `.venv` 建立全新 clean-clone 环境 | `NOT TESTED` | 本轮隔离 checkout 复用了已存在且版本匹配的 pinned 环境 |
| 真实 rollout、reward、backward、optimizer update | `NOT RUN` | 发现确定的 prompt-budget P0 后主动停止，未点火 |
| G1 wiring smoke | `BLOCKED` | 首个候选 episode 在第一次生成前已超过 prompt 上限 |
| formal training | `BLOCKED BY T2 v2` | G1 通过也不解除正式训练门禁 |

这里的 `preflight-only status=PASSED` 只表示 preflight 阶段通过，不能写成 G1 已通过。
本轮没有运行真实训练，也没有产生 optimizer-update 证据。

## 2. 验证身份

| 项目 | 值 |
|---|---|
| runbook 所在 `origin/main` | `3f76ee2`（相对冻结验证代码只新增文档） |
| 实际验证代码 | `8a090b86f667216a28a3846520b717348b4a6764` |
| Agent-R1 gitlink | `b124aa46534cbf2fb8bc8af11405774984c42ac7` |
| canonical merged smoke config SHA-256 | `6c58a822d3a6ff2751fbbfcda9451c9ad3eae9ffe81acb24d3417c3de40fe0ed` |
| candidate manifest file SHA-256 | `68c0687f7b917415def083194233b7893a2938cac0364cae0510415412adcd7a` |
| history raw-manifest SHA-256 | `2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90` |
| 本次实际模型 revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

模型 revision 是本次缓存中实际观测到的值；当前 YAML 尚未将它作为不可变输入固定，详见 P1。

本轮使用线下审核后复制的 reviewed manifest。该文件按协议保持 untracked，因此 preflight 记录
`repo_dirty=true`；tracked code 没有 diff。不能将此表述为整个工作树 clean。

## 3. 已通过的实测项目

### 3.1 数据与协议隔离

- 传输归档包含 macOS 产生的 AppleDouble `._*` 条目；过滤 `._*` 和 `__MACOSX*` 后得到
  10,820 个 history 文件，raw manifest 精确匹配上述冻结 SHA。无需仅因这些条目重新传包，
  但不能把归档不加过滤地直接解压进数据目录。
- processed 数据行为 `train=4`、`valid=1`、`test=1`。
- smoke 只使用 history 数据；shadow test 为 Valid 中的 `2024-11-14` 至 `2024-11-26`
  共 9 个交易日。
- 本次隔离验证未配置或访问 StockBench Final OOS；preflight 明确输出
  `Final OOS not accessed.`。

### 3.2 环境与 GPU kernel

`scripts/verify_h100_stack.py` 在 H100 NVL 上退出 0，确认：

| 组件 | 实测值 |
|---|---:|
| Python | `3.10.20` |
| PyTorch | `2.7.0+cu128` |
| PyTorch CUDA runtime | `12.8` |
| vLLM | `0.9.2` |
| verl | `0.7.0` |
| flash-attn | `2.8.0.post2` |
| Transformers | `4.53.2` |
| PyArrow | `20.0.0` |
| datasets | `1.2.1` |

真实 FlashAttention CUDA kernel 输出 shape 为 `[1, 16, 4, 64]`、dtype 为 `float16`，
且全部 finite，最终状态为 `H100_STACK_VERIFIED`。预编译 wheel 路径已经验证，因此本项目当前
冻结组合不需要系统 `nvcc`。

需要保留一个证据边界：本轮在隔离 checkout 中复用了先前建立的 pinned `.venv`，没有从空
`.venv` 完整重装一次。因此“已验证 frozen stack 能在此 H100 工作”成立；“clean-clone install
已完整验证”仍不成立。

### 3.3 代码与 launcher

以下检查均通过：

```text
pytest -q                                      136 passed
ruff check src tests scripts runtime_hooks     passed
mypy src tests                                 44 source files, passed
uv lock --check                                passed, 236 packages
git diff --check                               passed
Agent-R1 Hydra compose                         passed
```

兼容性文档中的 “mypy 检查 52 个 source files” 应更正为本次命令实际报告的 44。

dry-run 还确认 `run.seed=42` 已传到 data、actor data loader、actor FSDP 和 vLLM；A1 的
`rollout_n=2` 与 `max_agent_steps=32` 也进入最终 upstream command。该检查证明参数进入 Hydra
命令，不承诺整个分布式训练 bitwise deterministic。

### 3.4 完整 preflight-only

在真实 H100、完整 history 数据和已缓存模型上执行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --preflight-only
```

约 150 秒后退出 0，并输出：

```text
PREFLIGHT_PASSED
No training was started.
Final OOS not accessed.
```

这证明环境、数据、配置解析、模型预取、manifest 和协议门禁能够走完整个 preflight；它不执行
rollout、backward 或 optimizer step。

## 4. P0：当前 hash-bound G1 无法产生首个 rollout

对首个 candidate episode `smoke-train-001`，使用冻结 history、实际六个 tool schema、当前
`TradingEnv.reset()` 输出和 Qwen3-1.7B tokenizer revision `70d244...` 复现了真实 chat template：

```text
configured max_prompt_length = 4,096
first generation prompt       = 5,141 tokens
tool count                    = 6
```

锁定的 Agent-R1 loop 在调用模型前检查 prompt 长度；超过 `prompt_length` 后直接 `break`。
因此这个确定失败样本会返回空 steps，无法产生 rollout、reward、backward 或 optimizer-update
证据。一个首轮必失败样本已经足以阻断 G1，无需冒险启动正式 GPU 训练来再次证明。

仅提高 `max_prompt_length` 不是充分修复。最短的连续 `hold` 路径在八次生成前的 prompt 长度为：

```text
5,141 / 10,190 / 15,205 / 20,169 / 25,189 / 30,246 / 35,276 / 40,285
```

模型 context 为 40,960 tokens，还必须为 `max_response_length=2,048` 预留空间。当前状态文本会
携带 20 个完整 observation record，而对话历史又重复累计先前完整状态；`retrieve` 结果还可能
进一步放大 prompt。

### 必需修复

1. 将 observation 改为面向决策的紧凑表示，不重复完整嵌套 metadata。
2. 使用有界/滚动消息历史，只保留当前状态、必要动作摘要和有限的近期工具结果。
3. 对 `retrieve` 的条目数、字段和总 token 数设置明确上限。
4. 增加 tokenizer-level context-budget 门禁，至少覆盖完整 8-advance 最短路径和最坏允许工具输出，
   并断言 `prompt_tokens + allowed_response_tokens <= model_context_tokens`。
5. 修复后冻结新的代码 commit，重复 stack、tests、数据准备和 preflight，再启动 G1。

如果修改 smoke YAML 中的长度字段，必须同步更新 merged-config SHA 常量、测试和文档；如果只做
代码级压缩，配置 SHA 可以不变，但验证 commit 必须更新。runbook 中“现在不要继续修改代码”的
指示应撤销：在解决此 P0 前不得点火。

## 5. P1：建议在再次点火前修复

### 5.1 `SETUP.md` 的 exact-sync 命令会拆掉 GPU 环境

`SETUP.md` 先执行：

```bash
uv sync --frozen --extra gpu --extra data
```

但数据准备章节随后执行：

```bash
uv sync --frozen --extra data
```

`uv sync` 默认是 exact sync；第二条命令不会“只补 data”，而会把未选中的 GPU extra 从环境
移除。本轮 dry-run 显示将移除超过 160 个包，包括 flash-attn、PyTorch CUDA、vLLM、verl 和 CUDA
runtime。应将后者改为仍包含 `--extra gpu --extra data`，或明确使用独立 CPU 数据环境。

环境验收后的所有运行命令还应统一使用 `uv run --no-sync ...`；当前 `SETUP.md` 和 `README.md`
仍有多处裸 `uv run`，可能隐式改变已验收环境。

### 5.2 模型 revision 与 offline 策略未完全 fail closed

- 当前下载仍使用可变的 `main`，配置没有固定 exact model revision。
- preflight 会记录实际 revision，但不校验它是否等于预期值。
- launcher 使用 `setdefault("HF_HUB_OFFLINE", "1")`，外部 `HF_HUB_OFFLINE=0` 可以覆盖政策。
- upstream 仍接收模型 repo ID，而不是已解析的 exact snapshot/revision 路径。

建议固定 revision 或 exact local snapshot，强制 offline，并在 preflight 中校验实际加载身份。
若因此修改 hash-bound smoke 配置，需同步更新其配置哈希。

### 5.3 已验证的数据未被完整固化到证据包

raw manifest 本轮已实际校验通过；问题不是“数据未验证”，而是 observed audit 结果没有完整持久化：

- raw manifest verifier 的观测结果未写入 summary；
- reviewed manifest 与三个 processed Parquet 的实际 SHA-256 未记录；
- processed 文件的 row count 未全部写入可交接证据；
- `training.episodes=4` 与 processed train row count 之间缺少直接 fail-closed 绑定。

建议在 preflight artifacts 中保存 observed history audit、candidate/reviewed SHA、processed SHA 与
row counts、benchmark identity 和模型 snapshot/revision，并恢复 episode-count 绑定及回归测试。

### 5.4 update 证据字段语义需要收紧

当前 `summary.global_step` 是 optimizer record 数量的别名，并非从 upstream trainer 独立采集的
global step。验收时不能把它和 `optimizer_step_count` 当成两份独立证据。应采集真实 trainer
step，或更名并在 runbook 中说明其语义。

### 5.5 queue watchdog 与失败清理尚未闭环

- A1 是 terminal-only reward；在终止前连续出现零 reward 是正常现象。固定“8 次零 reward”
  watchdog 可能误杀合法轨迹。
- watchdog 事件写入 queue 全局日志，而不在对应 run 的 `events.log`，与 runbook 验收路径不一致。
- direct G1 命令没有 watchdog，当前 wrapper 不能称为与 direct command 等价的安全入口。
- trap 只处理 launcher/watchdog PID，不能保证 Ray/vLLM 子孙进程已退出。

建议使用 episode-aware/stage-aware watchdog，将事件写入 per-run artifacts；用独立进程组启动并
清理训练，在保留错误证据后再核查 Ray、GPU 和残留进程。修复前以 runbook 的 direct command
作为唯一 G1 入口，并人工监控。

### 5.6 validation commit 仍主要靠人工核对

`require_clean_repo=false` 是 reviewed manifest 本地交接方式带来的合理例外，但也意味着 preflight
不会强制 HEAD 等于 runbook 的 validation commit。建议给 G1 增加 expected-commit 机器门禁，
或由验收脚本 fail closed 核对 `git_state`，同时允许明确列出的 reviewed artifact。

## 6. P2：文档与防误用修订

- runbook 应把 data copy → AppleDouble 过滤 → offline review → prepare → model cache 明确列为
  preflight 前置，而不是只跳转到另一份文档。
- runbook 列出的 `gradient_nonzero` 和 `parameter_update_detected` 不是当前 summary 的实际顶层字段；
  应改成真实嵌套路径，并说明 sampled-gradient 的证据边界。
- 应将 `verify_h100_stack.py` 的 JSON 输出保存到每次 run artifacts。
- history-only G1 验证环境不得存在可访问的 StockBench Final OOS 副本；点火前应检查并将任何
  此类副本移出 G1 主机。只移动已有副本，不删除唯一副本。
- `SETUP.md` 中遗留的 “Codex signed” 和 formal StockBench 流程应明确标成历史/正式阶段，避免与
  当前“线下人员确认、G1 history-only”流程混淆。
- preflight-only summary 应使用阶段特定状态，避免顶层 `PASSED` 被误读成训练成功。

## 7. 建议修复与复验顺序

1. 先修 P0：状态压缩、有界历史、retrieve 上限、tokenizer 预算测试。
2. 同一批修复 `SETUP.md` exact-sync、模型 identity/offline 和 episode-count 绑定。
3. 扩充持久化证据，再修 queue watchdog、per-run 事件和进程组清理。
4. 冻结新 validation commit；重新核对 Agent-R1 gitlink、config SHA 与数据 manifest。
5. 在 H100 重跑 `verify_h100_stack.py`、136 tests、完整 preflight-only 和 prompt-budget 门禁。
6. 全部通过后才运行真实 G1，并以至少一次 finite backward 和真实 optimizer update 作为通过条件。

本轮控制台输出和本地 run artifacts 已保留，可通过线下方式交接；其中不包含 W&B 或云端数据依赖。

## 8. 2026-08-29 本地 P0 修复进展

本节记录的是第一轮本地修复；其“完成 P0”结论已被第 9 节的新反例推翻。第一轮工作树实现了：

- 当前 session 状态只保留决策所需 OHLCV、组合、有限近期事件和最后一次工具结果；
- 对话改为初始任务加当前状态的滚动窗口，不再累计全部 assistant/tool 历史；
- retrieve 最多返回 3 条、最多 8 个字段、文本字段最多 160 字符，完整 JSON 不超过 1 KiB；
- smoke preflight 使用缓存 Qwen snapshot 的真实 tokenizer 和 chat template，覆盖 4 个 train
  candidate、每个 episode 的完整 8-advance 路径，并同时检查 4,096 prompt 上限以及
  `prompt + 2,048 response <= model context`；
- Agent-R1 外部 flow 在运行期如果再次超过 prompt 上限会明确抛错，不再静默返回空 steps；
- 新增独立 H100 命令 `scripts/verify_context_budget.py`，不会启动推理或训练。

当前本机结果为 `138 passed`，相关 Ruff 与 mypy 检查通过。由于本机没有 H100 GPU 栈、缓存模型
和 processed smoke Parquet，真实 Qwen token 数仍为 `UNVERIFIED ON THIS MACHINE`。因此第 1 节的
G1 状态仍是 `BLOCKED`；下一步必须先冻结新的 validation commit，再在 H100 依次运行：

```bash
uv run --no-sync python scripts/verify_h100_stack.py
uv run --no-sync pytest -q
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/verify_context_budget.py \
  --config configs/smoke_1p7b.yaml
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml --preflight-only
```

只有四项全部通过后，才重新评估是否允许运行真实 G1。

## 9. 2026-08-29 Unicode 极限路径纠正

`9b72dca` 虽然消除了旧的 5,141-token 首轮膨胀和累计历史，但其 tokenizer gate 每个 session 只
模拟两次 ASCII `market` 查询。当前 `max_agent_steps=32` 对 8 次推进仍允许每个 session 三次
retrieve；使用最大长度 Unicode 查询的合法路径在真实 Qwen tokenizer 上达到 4,548 tokens，超过
4,096 上限。因此第 8 节不能作为 P0 closed 证据。

第二轮本地修复执行两项收口：

- 用户控制的 query/reason/error 等近期事件文本先转换为固定长度 ASCII 摘要，避免 Unicode 经
  JSON escape 后放大；
- tokenizer gate 同时跑 `planned-ascii` 和 `legal-unicode-max`，后者为每个 session 三次、每次
  200 个 Unicode 字符，完整使用 32-step 合法预算。

本地回归测试已经覆盖该边界，但真实修复后 token 数仍是 `UNVERIFIED ON THIS MACHINE`。只有新
validation commit 在 H100 输出同时包含两个场景的 `CONTEXT_BUDGET_PASSED`，并再次通过完整
preflight，P0 才能关闭。
