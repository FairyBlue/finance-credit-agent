# 2026-08-30 H100 G1 P1 集成交接

> **归档说明：** 本文描述集成要求，不再是可直接复制的当前执行入口。实际 H100 命令与外部
> exact commit 以 `2026-08-30_H100_G1_P1_VALIDATION_RUNBOOK.md` 为准。
>
> **后续状态：** G1 已于 2026-08-31 通过，T2 v2 已于 2026-09-01 通过。本文中的 `BLOCKED`
> 和 T2 阻断描述仅代表集成当时状态。

## 1. 任务与状态边界

请以最新 prompt 修复为基底，把 H100 本地审计分支中尚未进入 `main` 的 P1 行为重新实现到新的
integration branch。本任务同时包含 prompt gate 补强，以及模型身份、训练上限、证据、queue 安全和
协议同步等非 prompt 问题。

本文不是运行授权。所有门禁通过前保持 `G1: BLOCKED`；代码合并不等于
`READY_TO_IGNITE`，preflight 通过也不等于 `G1 PASSED`。

数据、运行目录、审核和签字均在线下由人交接。不要恢复 W&B、云端数据同步、加密交接或
“Codex signed”等旧流程。

## 2. 精确基线

| 角色 | Exact ref | 规则 |
|---|---|---|
| 集成基底 | `origin/main@744369917dc89c0de59390d414d8e6079e1b6c31` | 保留最新 Unicode-safe prompt/context 实现 |
| P1 参考 | `origin/fix/h100-g1-audit-20260827@794cfe0c196122b8ec3fa41d3b50f0d40461fb51` | 只参考行为和测试，不整分支合并 |
| Agent-R1 | `b124aa46534cbf2fb8bc8af11405774984c42ac7` | submodule 必须保持精确 pin |
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 配置、缓存和最终训练命令必须一致 |

如果 `origin/main` 已更新，先审新增 diff，再重新指定基底。实验必须绑定 exact validation commit，
不能使用浮动 `main`。

建议从本交接分支开始：

```bash
git fetch origin
git switch -c fix/h100-g1-p1-integration \
  origin/docs/h100-p1-integration-handoff-20260830
git submodule update --init --recursive
```

查看两端差异时比较 exact endpoint，避免 triple-dot 隐藏 `main` 侧的 Unicode 修复：

```bash
git diff 744369917dc89c0de59390d414d8e6079e1b6c31 \
  794cfe0c196122b8ec3fa41d3b50f0d40461fb51 -- \
  configs/smoke_1p7b.yaml src/agents/launcher.py src/agents/local_runs.py \
  src/env/trading.py scripts/run_queue.sh tests/test_run_metadata.py \
  tests/test_trading_env.py
```

## 3. 必须保留的 prompt 基线

保留 `7443699` 的以下实现，不要用旧分支对应文件覆盖：

- `src/agents/context_budget.py` 的真实 tokenizer gate；
- `planned-ascii` 和 `legal-unicode-max` 场景；
- `_bounded_ascii_text()`、最多 3 个 recent events、最多 3 条/1 KiB retrieve；
- 两消息滚动窗口，不累计完整 assistant/tool 历史；
- `src/agents/agent_flow.py` 的运行期 prompt 超限硬失败；
- Unicode regression tests。

本次 H100 审查对 `7443699` 的真实 Qwen tokenizer 结果为：

```text
status                  CONTEXT_BUDGET_PASSED
cases_checked           232
episodes_checked        4
scenarios_checked       legal-unicode-max, planned-ascii
max_prompt_tokens       2675
prompt_limit            4096
response_limit          2048
model_context_tokens    40960
```

额外手工压力路径也均低于 4,096：三次最大 Unicode retrieve 为 `2390 / 2488 / 2586`，
三次 Unicode unknown-tool 最高 `2711`，7 个持仓叠加混合检索最高 `3024`。最终 integration commit
仍必须重新执行这些检查。

## 4. 必须一并修复的 P1

| P1 | `main@7443699` 当前问题 | 集成完成条件 |
|---|---|---|
| Exact model/offline | revision 未固定；训练仍可能收到 repo ID；offline 可被外部值绕过 | exact revision、本地 snapshot、双 offline fail-closed |
| Episode cap | YAML/context 有 4，但上游无 `train_max_samples` | 4 rows 与 `data.train_max_samples=4` 双绑定 |
| Evidence | 数据、模型和最终命令证据未完整闭环 | 成功、失败和 preflight 都产生机器可读证据 |
| Retrieve 可见命中 | 搜索 full record 后返回 compact record | 只能搜索模型最终看到的 compact evidence |
| Prompt coverage | 只审 train 的 retrieve+hold | 增加 valid、invalid、混合结果和多持仓 |
| Trainer step | `global_step` 只是 optimizer record 数的别名 | 捕获真实 trainer step，或删除/准确更名 |
| Queue/watchdog | stall 判据、per-run 事件和子进程清理不足 | stage-aware、可审计、独立进程组清理，或禁用 queue |
| Validation identity | smoke 允许 dirty tree，缺少 expected-commit 硬门禁 | 外部冻结 exact commit、tracked tree 和实验相关路径 fail-closed |
| Docs/protocol | SETUP/README 命令和 EVAL 协议未同步 | 按最终语义统一并重新冻结配置 hash |

### 4.1 Exact model identity 与强制离线

参考旧分支的 `configs/smoke_1p7b.yaml`、`src/agents/launcher.py` 和
`tests/test_run_metadata.py`，在最新 `main` 上重新实现：

1. `model.revision` 固定为完整 `70d244cc...ad5e`；
2. T2 smoke exemption 同时绑定 model name 与 exact revision；
3. preflight 只接受该 revision 对应的本地 snapshot，缺失或不一致立即失败；
4. preflight 后把 actor 和 critic Hydra path 改写为 exact local snapshot，不把 Hub repo ID 交给
   训练子进程；
5. 无条件覆盖 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，即使父进程设为 `0`；
6. 将最终解析命令及 SHA-256 写入 run evidence；
7. `environment.json` 只记录 allowlisted offline flags，不能转储完整环境或秘密。

测试必须覆盖 hostile parent env、actor/critic snapshot 改写、revision mismatch fail-closed，以及证据中
两个 offline flags 最终均为 `1`。

### 4.2 Episode cap 真正传入上游

- 拒绝 bool、0、负数、浮点和字符串 episode cap；
- dry-run 与最终 command 都必须包含 `data.train_max_samples=4`；
- processed train Parquet 必须恰好 4 行，并与 reviewed manifest 的 train 数量一致；
- 保留 `max_steps=32`、`total_epochs=1` 及冻结的 seed、batch、rollout 参数。

测试同时断言 upstream command 和 processed row binding，防止“YAML 看似生效但上游未消费”。

### 4.3 Preflight 与 run evidence

至少记录：

- repo HEAD、submodule HEAD、tracked-tree 状态；
- canonical merged-config JSON SHA-256，用于 T2 exemption；
- `resolved_config.yaml` 文件字节 SHA-256，使用不同字段名，不能与上一项混称；
- active source manifest SHA-256、文件数、sessions/symbol；
- reviewed manifest path/SHA-256、benchmark-series SHA-256；
- train、valid、smoke-shadow test Parquet path/SHA-256/row count；
- exact model revision/snapshot、tokenizer class、逐 split/scenario prompt maxima；
- Final OOS 未访问；
- 最终 upstream command 和 command SHA-256。

证据应增量写入 recorder。若后续 gate 抛错，失败 summary 仍须保留此前已通过阶段的证据；不能等
`_verify_all_preflight_inputs()` 全部返回后才一次性写入。增加故障注入测试：source audit 通过后故意让
后续 gate 失败，并断言 source evidence 仍存在。

processed Parquet 内含路径绑定；换 repo 路径时重新生成 derived Parquet，不得手改 embedded path、
manifest SHA 或配置 SHA。

### 4.4 Retrieve 只能命中可见证据

保留 `main` 的 3-result、1 KiB、Unicode-safe 和字段/文本上限；但应先生成最终 compact record，再
只搜索其序列化文本。查询命中被 compact 丢弃的字段时必须返回空结果。恢复等价于旧分支
`test_retrieve_only_matches_evidence_present_in_compact_result` 的测试。

### 4.5 Prompt gate 覆盖

继续扩展 `src/agents/context_budget.py`，不要复制旧 `src/agents/prompt_budget.py`。至少覆盖：

- 4 个 train 和 1 个 valid episode；
- `planned-ascii`；
- 每 session 三次、每次 200 字符的 `legal-unicode-max`；
- 最大合法多持仓状态；
- 有界大 retrieve result 加最大 recent-event summaries 的混合路径；
- 最大 Unicode invalid/unknown-tool error steady state。

输出应包含 `splits_checked`、`scenarios_checked` 和逐场景 max case/tokens。运行期 hard guard 只是
fail-closed 兜底，不能替代训练前门禁。

### 4.6 Trainer-step 与 optimizer evidence

不要把当前 `summary.global_step` 当独立训练证据：旧实现只是把 optimizer probe record 数量写成
`global_step`，它与 `optimizer_step_count` 不是两份独立证明。

二选一：

1. 从 trainer 本身采集并记录真实 global step，同时保留 optimizer probe；或
2. 删除 `global_step`，或准确更名为 `observed_update_record_count`，并同步 runbook 和验收脚本。

真实 G1 的最低证据为：`backward_count>=1`、采样梯度 finite 且 non-zero、
`optimizer_step_count>=1`、parameter delta non-zero；如果保留 trainer step 字段，它必须独立采集。

### 4.7 Queue/watchdog 与进程清理

A1 是 terminal-only reward；在 terminal 前出现连续零 reward 是合法路径，watchdog 不能只凭固定次数
零 reward 判定 stall。请选择并实现：

- episode/stage-aware watchdog，同时覆盖 queue 与 direct entry；事件写入对应 `runs/<id>/`；使用独立
  进程组，在 INT/TERM/EXIT 时清理本次 Ray/vLLM 子孙进程；或
- 在上述能力完成前 fail-closed 禁用 queue，文档明确 direct entry 是唯一批准入口。

不得依赖不会清理子孙进程的简单 shell trap，也不要用可能影响其他任务的无范围全局清理充当正常流程。

### 4.8 Exact validation commit 与工作树

`training.require_clean_repo=false` 不能替代实验身份门禁。最终实现必须机器校验：

- `HEAD == expected_validation_commit`，其中 expected SHA 来自被验证 commit 之外的冻结输入，例如
  runbook 参数 `--expected-commit <SHA>`、线下审核文件或外部 exact ref；不得把自身 SHA 写入同一
  tracked commit，造成无法收敛的 SHA 自引用；
- tracked tree 无修改，submodule 与 pinned commit 一致；
- 对可能影响执行的 untracked import、配置和 runtime-hook 路径 fail-closed；
- `.venv`、`.pytest_cache`、`.mypy_cache`、`.ruff_cache`、`__pycache__` 等工具环境/cache 可以存在，
  不做逐文件 SHA；环境身份由 lock、H100 stack 和 `environment.json` 证明；
- configured reviewed/data/processed inputs 与 run outputs 才做明确的路径、SHA 和行数绑定。

为 expected-commit mismatch、tracked dirty、submodule mismatch 和实验相关未跟踪代码/config 增加
fail-closed 测试。外部 expected SHA 只能在代码、配置和人工审核全部冻结后产生，不能为通过门禁
修改 expected value。

### 4.9 SETUP、README 与 EVAL_PROTOCOL

- exact sync 使用 `uv sync --frozen --extra gpu --extra data`；
- SETUP、README 和 runbook 的执行命令统一为 `uv run --no-sync`；
- 签字由合作者线下人工完成，不使用 “Codex signed”；
- evidence 通过本地目录拷贝，不依赖 W&B；
- 正式协议应记录 `retrieve<=3`、retrieve JSON `<=1 KiB`、`recent_events<=3`、Unicode-safe
  summaries、两消息窗口和真实 tokenizer gate。

不要原样恢复旧分支 `retrieve=10`/signals 版 v0.7。应按最终实现形成新的协议提案并线下审核。

不要复用旧 canonical merged-config hash
`dd06b1b52f1541ae89d00ba49549562d6819f2dc197a4fd4399e7c48d944b4f8`；语义已经变化，必须重新计算。

## 5. 禁止整分支合并

不要执行：

```bash
git merge origin/fix/h100-g1-audit-20260827
git cherry-pick e42fdda
git cherry-pick 2ff761c
git cherry-pick 794cfe0
```

旧分支包含另一套 observation、retrieve 和 prompt-budget 语义，整段合并可能覆盖 `7443699` 并重新
引入 Unicode P0。请按 4.1 至 4.9 分成可审查的代码和测试提交。

## 6. CPU 与 H100 验收

CPU 检查：

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts runtime_hooks
uv run --no-sync mypy src tests
uv lock --check
git diff --check
git submodule status
```

不得少于当时的 `138 passed` 历史基线；新增测试后总数应增加，不能通过删除测试维持绿色。

H100 顺序：

```bash
export FCA_VALIDATION_COMMIT=REPLACE_WITH_EXACT_40_CHAR_COMMIT
uv run --no-sync python scripts/verify_h100_stack.py
uv run --no-sync pytest -q
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/verify_context_budget.py \
  --config configs/smoke_1p7b.yaml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml --preflight-only \
  --expected-commit "$FCA_VALIDATION_COMMIT"
```

下表全部 PASS 后才是 `READY_TO_IGNITE`：

| Gate | PASS 条件 |
|---|---|
| Exact identity | 外部 expected commit、clean tracked tree、submodule pin 和实验相关未跟踪路径全部机器校验 |
| Frozen stack | 输出 `H100_STACK_VERIFIED`，版本与 lock 一致，真实 FlashAttention kernel 通过 |
| CPU quality | pytest、Ruff、mypy、lock、diff 全部退出 0 |
| Model/offline | exact revision/snapshot；最终 command 无 repo ID；双 offline 强制为 1 并留证 |
| Episode cap | train=4 rows；command 含 `train_max_samples=4`、`max_steps=32`、`total_epochs=1` |
| Prompt paths | 4 train + 1 valid 及 4.5 全部场景；prompt `<=4096` 且加 response 后 `<=40960` |
| Data/evidence | source、reviewed、processed hashes/rows 全匹配，partial failure evidence 可验证 |
| Queue/direct | watchdog 与清理验收通过；若未实现则 queue 明确禁用且 direct-only fail-closed |
| Full preflight | 输出 `PREFLIGHT_PASSED`、`No training was started.`、`Final OOS not accessed.` |

只有随后真实 smoke 产生 rollout、finite reward、至少一次满足 4.6 的 backward/optimizer/parameter
update 证据并干净退出，才能写 `G1 PASSED`。这是集成当时的门禁；G1 与 T2 v2 后续均已通过。

## 7. 合作者交付物

请返回：

1. integration branch 与 exact commit；
2. 相对 `7443699` 的逐文件 diff summary；
3. 新 canonical merged-config JSON SHA 和 resolved YAML file SHA，以及线下人工审核状态；
4. CPU、H100 stack、真实 tokenizer 和完整 preflight 日志；
5. `runs/<preflight-id>/` 本地拷贝及 SHA-256 清单；
6. queue enabled 或 direct-only 的明确状态；
7. `BLOCKED`、`READY_TO_IGNITE` 或 `G1 PASSED` 中唯一一个状态。
