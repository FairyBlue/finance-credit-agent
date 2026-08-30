# Finance Credit Agent：实验接手与运行手册

更新日期：`2026-08-30`

这是一份面向新合作者的单一入口文档。即使没有读过历史对话，只要拥有本仓库、冻结历史数据和
一台 Linux x86_64 H100，就应当能够通过本文理解研究动机、判断当前进度、知道 G1 为什么尚未
获准点火，并知道哪些验证和正式实验现在允许或禁止运行。

仓库代码、配置、Git 状态和实际运行结果高于本文。如果出现冲突，立即停止实验，按第 16 节处理，
不要自行放宽门禁。

## 0. 接手后先看这里

当前结论只有五条：

1. 本地实验脚手架和协议门禁已经完成，状态为 `G0 PASSED`；当前本机为 141 tests passed。
2. 验证提交 `8a090b8` 已在 H100 NVL 上通过 frozen stack、真实 FlashAttention kernel、136 tests
   和完整 preflight；本轮复用了 pinned `.venv`，没有证明从空 `.venv` clean-clone 安装通过。
3. `7443699` 的 Unicode-safe prompt 修复已保留；本轮又补齐 exact model/offline、4-row cap、
   Train/Valid 多场景 tokenizer gate、增量证据和 exact validation identity。当前只在本机验证，
   尚未在 H100 复验，不能宣称 P0/P1 closed。
4. 下一项任务是把本轮变更冻结为新的 validation commit，在 H100 依次重跑 stack、tests、独立
   context-budget 检查和完整 preflight；这些门禁通过前仍不得点火 G1。
5. `scripts/run_queue.sh` 已全面 fail closed；G1 只能使用带 `--expected-commit` 的 direct launcher。
   正式训练继续由 T2 v2 阻断，不得读取 StockBench Final OOS 来调试 smoke。

本次 H100 实测、P0/P1/P2 和证据边界见
[`docs/2026-08-27_H100_G1_AUDIT_RESULTS.md`](2026-08-27_H100_G1_AUDIT_RESULTS.md)。

接手后的推荐顺序：

```text
读取 H100 实机审计
  → checkout 新的 P0 validation commit
  → 在 H100 重新核对 stack、tests、数据与模型身份
  → 运行真实 tokenizer context-budget 验收
  → 重新运行 --preflight-only
  → 全部通过后才运行唯一一次批准的 G1 smoke
  → 检查真实 backward、trainer step 和 optimizer-update 证据
```

## 1. 这个项目为什么存在

交易 Agent 通常要经过很多次检索、计算和下单，最后才得到一个组合收益。只奖励最终结果时，模型
很难判断究竟是哪一步做对了；但如果随意给中间步骤奖励，又可能让模型优化错误的代理指标。

本项目研究的问题是：**金融领域原生的逐步 credit assignment，能否比只看最终收益或通用逐步
奖励更可靠地训练工具型交易 Agent？**

三条实验臂是：

| Arm | 方法 | 回答的问题 |
|---|---|---|
| A1 | Outcome-only GRPO | 只给最终组合结果时，基准表现如何？ |
| A2 | 通用 StepPO | 通用逐步奖励是否比最终奖励更好？ |
| A3 | Finance-native credit | 按金融轨迹结构分配 credit 是否进一步改善训练？ |

这里最重要的不是先跑出一个高收益数字，而是保证结论没有被未来数据、错误成交边界、不同账务
引擎或不可复现的数据版本污染。金融回测中的这些错误往往不会让程序报错，反而会让结果看起来
更好，所以项目采用“先证明实验有效，再扩大训练”的顺序。

## 2. 为什么现在只跑 G1 smoke

G1 smoke 只证明训练链路接通：数据能加载、环境能推进、模型能 rollout、reward 能产生、反向传播
和 optimizer 能真实更新参数、运行证据能完整保存。

它不属于正式训练，不能用于：

- 收益、Sharpe、显著性或论文结论；
- A1/A2/A3 优劣比较；
- 超参数、模型或 checkpoint 选择；
- 调整 reward、Regime 或 T2 规则。

正式训练暂时不能开始，原因有两个：

- T2 v2 泄漏探针已经预注册，但尚未实现和通过；正式 launcher 会 fail closed。
- A3 当前只有 M1 uniform broadcast 作为 smoke/ablation，支持科学主结论的 M2 或 M3 尚未实现。

因此，完成 G1 后的正确动作是记录结果并返回开发，不是顺手启动正式三臂实验。

## 3. 当前进度与真实阻塞

状态截至 `2026-08-29`：

| 项目 | 状态 | 证据或说明 |
|---|---|---|
| G0 协议与本地脚手架 | `PASSED` | 141 tests；Ruff、mypy、lock 与 whitespace checks 通过 |
| Prompt-budget P0 修复 | `LOCAL PASS / H100 UNVERIFIED` | 保留 Unicode 修复，并覆盖 Train/Valid、混合持仓/检索及 Unicode invalid 路径 |
| G1 P1 门禁 | `LOCAL PASS / H100 UNVERIFIED` | exact model/offline、4-row cap、增量证据、direct-only 与 expected-commit 已实现 |
| G1 A1/Qwen3-1.7B smoke | `BLOCKED` | 新 integration commit 尚未冻结并在 H100 通过 context-budget/preflight，真实 smoke 未运行 |
| T1 未来价格扰动 | 已实现 | 历史读取与数值特征保持不变 |
| T2 v2 train/valid 探针 | `BLOCKED` | 已预注册，未实现；禁止回退旧 StockBench 探针 |
| T3 train-only 标准化 | 已实现 | 全样本 fit 会失败 |
| T4 未来文档删除不变性 | 已实现 | 历史检索结果保持不变 |
| Reward 标定 | 已冻结 | 只使用 Train；表和哈希已绑定 |
| Smoke Final OOS 隔离 | 已实现 | 配置、manifest、processed provenance 三层 fail closed |
| 本地结构化运行记录 | 已实现 | 非 dry-run 均写入 `runs/<run_id>/` |
| H100 GPU/model 实测 | `PARTIAL PASS` | `8a090b8` 上 stack、真实 kernel 和模型缓存通过；复用 pinned `.venv`，未做空环境重装 |
| Smoke manifest candidate | 已生成 | 状态为 `pending_human_review`，文件和 benchmark 哈希已冻结 |
| Reviewed smoke manifest | H100 已使用 | 经线下审核后复制，按协议保持 untracked；GitHub 不包含该文件 |
| Smoke processed Parquet | H100 已验证 | 实测 `train=4`、`valid=1`、`test=1`；文件不进入 GitHub |

本次 H100 移植以 `9e4bfe3c2d9d6ef36fbd684efcc78c2b1c600cfd` 为实际起点；精确移植
commit 记录在 `docs/H100_COMPATIBILITY.md`。开发时先同步最新 `origin/main`；执行实验时必须
使用当前 runbook 指定的精确 validation commit，不能在浮动的 `main` 上运行。

## 4. 冻结的实验边界

正式边界保持不变：

| Split | 日期 | 用途 |
|---|---|---|
| Train | `2023-01-03`～`2024-10-31` | 训练和 train-only 标定 |
| Valid | `2024-11-01`～`2025-02-28` | 验证与正式选择流程 |
| Final OOS | `2025-03-03`～`2025-06-30` | 最终科学评测，83 个 XNYS session |

G1 只使用 history 中的短切片：

- Smoke Train：Train 最前 36 个交易日，拆成 4 个连续、无重叠的 9-session episode；
- Smoke Valid：`valid_dates[0:9]`，即 `2024-11-01`～`2024-11-13`；
- Smoke Shadow Test：`valid_dates[9:18]`，即 `2024-11-14`～`2024-11-26`。

Shadow Test 必须包含以下 9 个 session，日历结果不同就停止：

```text
2024-11-14  2024-11-15  2024-11-18
2024-11-19  2024-11-20  2024-11-21
2024-11-22  2024-11-25  2024-11-26
```

其 provenance 必须始终是：

```yaml
split_role: smoke_shadow
source_split: valid
purpose: wiring_only
scientific_use_allowed: false
checkpoint_selection_allowed: false
final_oos: false
```

Smoke 配置会显式禁用 StockBench。路径、manifest、processed row 或日期只要指向 StockBench/
Final OOS，preflight 与 launcher 都会失败。把 OOS 文件改名成 `smoke_test.parquet` 也不会通过。

## 5. 复现身份与不可变规则

必须保持：

- Agent-R1 submodule：`b124aa46534cbf2fb8bc8af11405774984c42ac7`；
- Smoke merged-config SHA-256：
  `2ec544cb4e539cdbc2d3301b255384d3fdd770f5801f8d0ba3194081ac39d844`；
- History raw-manifest SHA-256：
  `2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90`；
- 模型名称：`Qwen/Qwen3-1.7B`；固定 revision：
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；
- seed：`42`；训练 episode：`4`；rollout n：`2`；epoch：`1`；
- W&B 禁用，运行证据只保存在本地文件中。

禁止：

- 编辑 `third_party/agent-r1`；
- 为了通过 smoke 修改配置哈希、日期、provenance 或门禁；
- 从 Yahoo 重新抓一份“看起来相同”的数据来替代冻结快照；
- 在 smoke 主机上放入或读取 StockBench Final OOS；
- 重跑旧 `scripts/run_t2_probe.py`；
- 把 preflight、模型加载或只有 rollout 的结果写成 G1 passed。

## 6. 从 GitHub 干净接手

在 H100 主机上执行：

```bash
git clone --recurse-submodules \
  https://github.com/HaoPan036/finance-credit-agent.git
cd finance-credit-agent
git fetch origin
export FCA_VALIDATION_COMMIT=REPLACE_WITH_CURRENT_RUNBOOK_EXACT_COMMIT
git checkout --detach "$FCA_VALIDATION_COMMIT"
git submodule update --init --recursive

git status --short
git rev-parse HEAD
git rev-parse origin/main
git -C third_party/agent-r1 rev-parse HEAD
```

验收：

- `git status --short` 没有输出；
- 本地 HEAD 与外部冻结的 `$FCA_VALIDATION_COMMIT` 完全一致；`origin/main` 只作信息记录，不能
  替代 exact validation commit；
- submodule 输出
  `b124aa46534cbf2fb8bc8af11405774984c42ac7`。

安装 uv 和 CPU 依赖：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10.20
uv venv --python 3.10.20
uv sync --frozen
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts runtime_hooks
uv run --no-sync mypy src tests
uv lock --check
git diff --check
```

所有已收集 pytest 用例和其他检查必须成功；测试数量随冻结提交增加时，不要为恢复旧数字而删除
测试。

## 7. 获取冻结 history 数据

GitHub 不包含原始数据。G1 不能仅靠 clone 启动，必须从项目维护者持有的冻结快照同步：

```bash
export DATA_SOURCE_HOST=<保存冻结数据的-ssh-host>
export DATA_SOURCE_PROJECT_DIR=<该主机上的绝对项目目录>

mkdir -p data/history/storage
rsync -a --partial --info=progress2 \
  "$DATA_SOURCE_HOST:$DATA_SOURCE_PROJECT_DIR/data/history/storage/" \
  data/history/storage/
```

如果由数据持有者从工作站主动推到 H100，则使用：

```bash
export H100_HOST=<h100-ssh-host>
export H100_PROJECT_DIR=<h100上的绝对项目目录>

rsync -a --partial --info=progress2 data/history/storage/ \
  "$H100_HOST:$H100_PROJECT_DIR/data/history/storage/"
```

不要为 G1 同步 `data/stockbench/`。同步后先做 history-only 原始字节核验：

```bash
uv run --no-sync python - <<'PY'
from src.agents.config import load_experiment_config
from src.agents.launcher import verify_configured_source_manifests

config = load_experiment_config("configs/smoke_1p7b.yaml")
audits = verify_configured_source_manifests(config, repo_root=".")
assert set(audits) == {"history"}
audit = audits["history"]
print("manifest_sha256", audit.manifest_sha256)
print("file_count", audit.file_count)
print("sessions_per_symbol", sorted(set(audit.sessions_per_symbol.values())))
PY
```

若 SHA-256、文件数 `10,820` 或每票 session 数 `541` 不一致，停止并重新同步原冻结快照。不要
联网重抓后改配置来迁就新数据。

## 8. 验证并人工审核 smoke 输入

以下命令全部在仓库根目录执行。

### 8.1 先验证仓库内已冻结的 candidate

仓库已提供 `configs/episode_manifest.smoke.candidate.yaml`，不需要每次重新生成。它必须满足：

```text
file SHA-256:             68c0687f7b917415def083194233b7893a2938cac0364cae0510415412adcd7a
benchmark series SHA-256: eb3a6583e2f35b7b26e5f0a0d862134cddf2b5fc71febf8e9742eee6a7c14988
status:                   pending_human_review
episodes:                 train=4, valid=1, test=1
```

验证文件哈希：

```bash
uv run --no-sync python - <<'PY'
from hashlib import sha256
from pathlib import Path

path = Path("configs/episode_manifest.smoke.candidate.yaml")
expected = "68c0687f7b917415def083194233b7893a2938cac0364cae0510415412adcd7a"
actual = sha256(path.read_bytes()).hexdigest()
assert actual == expected, (actual, expected)
print(actual)
PY
```

如果哈希不符，先确认 checkout 的是否为 runbook 指定的 exact validation commit；不要直接修改
candidate，也不要用浮动 `origin/main` 替换冻结身份。

### 8.2 可选：独立重现 candidate

这一步只用于审计或仓库文件缺失时重现。生成器只读配置、XNYS 日历和冻结 history，
不读取 StockBench 或 Final OOS。输出写入临时目录，不覆盖仓库候选文件。

```bash
audit_dir="$(mktemp -d)"
uv run --no-sync python scripts/build_smoke_manifest.py \
  --config configs/smoke_1p7b.yaml \
  --output "$audit_dir/draft.yaml"
uv run --no-sync python scripts/freeze_benchmarks.py \
  --config configs/smoke_1p7b.yaml \
  --draft "$audit_dir/draft.yaml" \
  --output "$audit_dir/candidate.yaml"
cmp "$audit_dir/candidate.yaml" \
  configs/episode_manifest.smoke.candidate.yaml
```

`cmp` 必须没有输出且退出码为 0。比对完成后可删除该临时目录。

### 8.3 完成人工审核

不需要数字签名或额外加密签名。由合作者线下确认以下内容，并在团队记录中留痕即可：

- data snapshot 只有 `history`，没有 `stockbench`；
- Train 为 4×9 session，Valid 和 Shadow Test 各 9 session；
- Shadow Test 日期和第 4 节列出的 9 日完全一致；
- 所有 open/close 时间符合 `America/New_York` 的 XNYS session；
- Shadow Test 的六个 provenance 字段完全一致；
- benchmark NAV 来自共享 next-open ledger；
- `src/data/history.py` 的后续拆股反向还原与 volume 最近整数还原符合冻结规则；
- 没有启用 news 或未经逐字段 PIT 审计的 current fundamentals。

审核完成后，复制 candidate；不要手工改写其内容：

```bash
cp configs/episode_manifest.smoke.candidate.yaml \
  configs/episode_manifest.smoke.reviewed.yaml
cmp configs/episode_manifest.smoke.candidate.yaml \
  configs/episode_manifest.smoke.reviewed.yaml
```

`cmp` 必须没有输出且退出码为 0。

### 8.4 生成三个 processed 文件

```bash
uv run --no-sync python scripts/prepare_episodes.py \
  --config configs/smoke_1p7b.yaml \
  --manifest configs/episode_manifest.smoke.reviewed.yaml

test -f data/processed/smoke_train.parquet
test -f data/processed/smoke_valid.parquet
test -f data/processed/smoke_test.parquet
```

这些文件都来自 history；`smoke_test.parquet` 只是保留的文件名，其内部 provenance 明确是 Valid
shadow，不是 Final OOS。

## 9. 安装 H100 GPU 环境并缓存模型

当前 lock 已移植旧 H100 验证过的 Python 3.10.20 / torch 2.7.0+cu128 / flash-attn
2.8.0.post2 / vLLM 0.9.2 / verl 0.7.0 / Transformers 4.53.2 / PyArrow 20.0.0 组合。
该 stack 和真实 kernel 已在验证提交 `8a090b8` 上通过；证据来自复用的 pinned `.venv`，仍未证明
从空 `.venv` clean-clone 安装。完整 wheel 哈希、证据边界、禁止升级事项和 nvcc 政策见
[`docs/H100_COMPATIBILITY.md`](H100_COMPATIBILITY.md)。

```bash
nvidia-smi
uv sync --frozen --extra gpu --extra data
uv pip check --python .venv/bin/python
uv run --no-sync python scripts/verify_h100_stack.py
```

使用预编译 FlashAttention wheel，不安装 CUDA Toolkit，不要求系统 `nvcc`。
缓存模型并打印实际 revision 路径：

```bash
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
)
print(path)
PY
```

preflight 不联网，只接受上述 revision 的本地 exact snapshot；最终 trainer 命令使用该目录而非
repo ID，并强制 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。`run.seed` 控制主要数据、
actor、critic 和 vLLM 随机源，但分布式 GPU 训练不承诺 bitwise deterministic。

## 10. 先 dry-run，再运行 H100 preflight

`8a090b8` 已在真实 H100 上完成一次完整 preflight，并输出本节三行结果；它没有启动训练，也不
等于 G1 passed。prompt-budget P0 修复后，必须在新的 validation commit 上先运行独立预算检查：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/verify_context_budget.py \
  --config configs/smoke_1p7b.yaml
```

只有输出 `"status": "CONTEXT_BUDGET_PASSED"`，并记录 `max_prompt_tokens`、
`model_context_tokens`、`prompt_limit`、`response_limit` 和实际 tokenizer class，才继续完整
preflight。该脚本使用缓存的 exact Qwen snapshot、真实 chat template、六个 tool schema、4 个
Train candidate 和 1 个 Valid candidate 的完整 8-advance 路径，不启动模型推理或训练。门禁
必须同时覆盖：

- 每个 session 两次 ASCII retrieve 的计划路径；
- 每个 session 三次、每次 200 个 Unicode 字符的合法 32-step 极限路径。
- 多持仓、最大 recent events 与有界 retrieve result 的混合路径；
- 最大 Unicode unknown-tool/invalid error 的稳定路径。

输出必须列出 Train/Valid，并按 split/scenario 给出最大 token 数。只测 Train 或只测
`planned-ascii` 不能关闭 P0。

CPU-safe dry-run 只检查配置合并、provisional 命令构造和基础 Git/submodule/data-version 值；它
不执行 external expected-commit、overlay、GPU/model、raw/processed data 或 tokenizer 门禁：

```bash
uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml --dry-run
```

确认输出标为 `command_state=PROVISIONAL`，只有 console logger、没有 W&B，并包含：

```text
model.path=Qwen/Qwen3-1.7B
trainer.n_gpus_per_node=1
reproducibility.agent_r1_commit=b124aa46534cbf2fb8bc8af11405774984c42ac7
```

然后执行不会启动训练的完整 H100 preflight：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml --preflight-only \
  --expected-commit "$FCA_VALIDATION_COMMIT"
```

它会检查配置与唯一 smoke 豁免、Git/submodule、reward freeze、GPU/CUDA、模型缓存、history 原始
manifest、reviewed manifest、benchmark 重算、三个 processed 文件、Final OOS 隔离，并再次执行
上述真实 tokenizer prompt-budget 门禁。

只有同时看到以下三行才算通过：

```text
PREFLIGHT_PASSED
No training was started.
Final OOS not accessed.
```

通过后可把 G1 从 `BLOCKED` 记为 `IN_PROGRESS`。preflight 本身绝不能记为 `PASSED`。

## 11. 运行唯一批准的 G1 smoke

**当前禁止执行本节命令。** 第二轮 P0 代码修复只在当前本机通过；必须先冻结新 commit，并在 H100
重新通过 stack、tests、数据身份、模型身份、context-budget 和 preflight 门禁。
以下命令只保留为门禁全部恢复后的执行入口。

推荐直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --expected-commit "$FCA_VALIDATION_COMMIT"
```

`FCA_VALIDATION_COMMIT` 必须由 runbook 或线下 exact ref 提供，不能使用浮动 `main`。launcher 会
核对 HEAD、tracked tree、Agent-R1 submodule 与实验相关代码/config overlay。

queue wrapper 当前全面禁用：watchdog 对 A1 terminal-only reward 可能误杀，事件位置和子进程清理
也尚未闭环。任何参数都会在调用 `uv` 前退出非零：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_queue.sh configs/smoke_1p7b.yaml
```

不得通过 `--formal` 绕过。正式训练由 T2 v2 门禁单独阻断。

每次非 dry-run 都会创建：

```text
runs/<run_id>/
  provisional_command.txt
  command.txt
  resolved_config.yaml
  git_state.json
  environment.json
  data_manifest.json
  events.log
  metrics.jsonl
  summary.json
  checkpoints/
  artifacts/
```

失败时 `provisional_command.txt` 保留尚含 Hub ID 的候选命令，`command.txt` 明确标记没有 final
命令；只有完整 preflight 通过后，`command.txt` 才是绑定本地 exact snapshot 的权威命令。即使
preflight 或训练失败，也必须保留该目录，不要删除失败证据。

## 12. 如何判断 G1 是否真正通过

找到最近一次 smoke 记录：

```bash
RUN_DIR=$(find runs -mindepth 1 -maxdepth 1 -type d \
  -name '*smoke-a1-qwen3-1p7b*' | sort | tail -1)
test -n "$RUN_DIR"
echo "$RUN_DIR"
uv run --no-sync python -m json.tool "$RUN_DIR/summary.json"
```

只有以下条件全部满足，G1 才能标为 `PASSED`：

- `summary.json.status == "PASSED"`；
- `exit_code == 0`；
- `final_oos_accessed == false`；
- `backward_count > 0`；
- `optimizer_step_count > 0`；
- `observed_update_record_count > 0`；该字段只是探针记录数，不作为独立 trainer-step 证明；
- `optimizer_update_evidence.detected == true`；
- representative update 的 `grad_norm` 有限；
- representative parameter 的 `parameter_delta_l2` 和 `parameter_delta_max` 有限且非零；
- Git、Agent-R1、config、data 和 model revision 已记录；
- `events.log` 中没有 reward collapse、重复输出或 watchdog failure；
- rollout 与 reward 确实出现，而不只是模型加载成功。

可以运行以下最低限度的机器检查：

```bash
RUN_DIR="$RUN_DIR" uv run --no-sync python - <<'PY'
import json
import math
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
summary = json.loads((run_dir / "summary.json").read_text())
evidence = summary["optimizer_update_evidence"]
update = evidence["representative_update"]

assert summary["status"] == "PASSED"
assert summary["exit_code"] == 0
assert summary["final_oos_accessed"] is False
assert summary["backward_count"] > 0
assert summary["optimizer_step_count"] > 0
assert summary["observed_update_record_count"] > 0
assert evidence["detected"] is True
assert math.isfinite(update["grad_norm"])
assert math.isfinite(update["parameter_delta_l2"])
assert math.isfinite(update["parameter_delta_max"])
assert update["parameter_delta_l2"] > 0
assert update["parameter_delta_max"] > 0
print("G1 machine acceptance checks passed; inspect rollout/reward events manually.")
PY
```

完整 1.7B checkpoint 对 G1 不是必需；真实参数更新证据是必需。

## 13. 常见阻塞与处理办法

| 报错或现象 | 含义 | 正确处理 |
|---|---|---|
| 缺少 `torch/verl/vllm/flash_attn` | GPU extra 未正确安装 | 按第 9 节顺序重装，不要随意升级 verl |
| `CUDA GPU is unavailable` | 驱动、CUDA visibility 或资源分配错误 | 检查 `nvidia-smi`、调度器和 `CUDA_VISIBLE_DEVICES` |
| model revision not cached | 本地没有 Qwen3-1.7B snapshot/ref | 按第 9 节预下载，训练中不要联网补下载 |
| missing reviewed manifest | 尚未完成 candidate 审核 | 回到第 8 节，不要把 draft 直接冒充 reviewed |
| missing processed dataset | 三个 Parquet 尚未生成 | 从 reviewed manifest 重新运行 `prepare_episodes.py` |
| raw manifest SHA mismatch | 数据不是冻结快照或同步损坏 | 停止，重新 rsync 原快照；不要修改冻结 SHA |
| invalid T2 smoke exemption | smoke 配置漂移或启动了非批准配置 | 重新 checkout runbook 的 exact commit，只使用 `configs/smoke_1p7b.yaml` |
| Final OOS/StockBench rejection | smoke 输入污染了最终评测边界 | 删除错误输入并重新生成 history-only smoke 数据 |
| 进程退出 0 但 update 未检出 | 没有证明发生真实训练 | G1 失败；保留 artifacts，检查梯度/optimizer/Ray hook |
| reward collapse 或重复输出 | rollout 已失效 | G1 失败；保留 run 记录，不得挑选性删除失败 run |

任何失败都不得通过降低门禁、改 provenance 或删除失败记录来“修复”。

## 14. 同步运行证据

H100 完成后，把完整 `runs/` 同步回协作目录：

```bash
export H100_HOST=<h100-ssh-host>
export H100_PROJECT_DIR=<h100上的绝对项目目录>

rsync -a --partial --info=progress2 \
  "$H100_HOST:$H100_PROJECT_DIR/runs/" runs/
```

W&B、API key、云端 logger、数字签名或加密签字流程都不需要。Git/data/config/model 哈希用于实验
身份核验，不是人员签名系统。

## 15. G1 之后该做什么

当前尚未进入“G1 之后”。Unicode-safe P0 与本轮 P1 集成已在本机通过定向检查；当前唯一下一步
是冻结新 validation commit，并在 H100 重跑 stack、tests、Train/Valid 四类 context-budget 场景
和 preflight。H100 结果回来前，只能写成 `LOCAL PASS / H100 UNVERIFIED`。

G1 通过后：

1. 保存并共享 `runs/<run_id>/`；
2. 在团队进度记录中把 G1 改为 `PASSED`；
3. 不打开 Final OOS，不启动正式三臂训练；
4. 实现 T2 v2：只允许 history Train/Valid，pooled per-stock，强弱两档正控制均只在内存；
5. T2 v2 必须同时满足强档检出、40% flip 弱档检出、真实管道未显著检出；
6. 弱档未检出必须如实保留为功效边界；
7. 正式 A3 主结论前，实现并预注册 M2 或 M3，不能用 M1 冒充。

只有 T2 v2 通过、正式 manifests 完成审核、代码和数据身份冻结后，才进入正式实验设计与执行。
Final OOS 只在最终科学评测时开启。

## 16. 信息冲突时如何处理

优先级从高到低：

1. 当前 Git checkout、配置、代码和实际测试/运行结果；
2. `EVAL_PROTOCOL.md` 最新追加修订；
3. 本文；
4. `README.md` 与 `SETUP.md`；
5. 历史 handoff、聊天记录或口头说明。

若本文的测试数、commit SHA、文件状态或命令与当前 `main` 不一致：

- 不要猜；
- 记录 `git status --short --branch`、`git rev-parse HEAD`、失败命令和 run 目录；
- 在不触碰 Final OOS 的前提下复核代码和配置；
- 让仓库维护者更新本文后再继续。

## 17. 文件导航

只跑 G1 时，最重要的文件是：

| 文件 | 作用 |
|---|---|
| `docs/EXPERIMENT_HANDOFF.md` | 本接手与运行手册 |
| `docs/2026-08-27_H100_G1_AUDIT_RESULTS.md` | 当前 H100 实测结果、P0/P1/P2 与复验顺序 |
| `docs/2026-08-27_H100_G1_VALIDATION_RUNBOOK.md` | `8a090b8` 原始执行任务单；已由实测结果更新状态 |
| `docs/2026-08-30_H100_G1_P1_VALIDATION_RUNBOOK.md` | 本轮 P0/P1 集成的唯一当前 H100 验收命令 |
| `docs/H100_COMPATIBILITY.md` | H100 frozen stack、wheel 哈希、clean-clone 和证据边界 |
| `EVAL_PROTOCOL.md` | 冻结协议与历次追加修订 |
| `configs/smoke_1p7b.yaml` | 唯一批准的 G1 配置 |
| `configs/base.yaml` | 数据、reward、交易、Regime 与门禁基础配置 |
| `configs/episode_manifest.smoke.candidate.yaml` | 已冻结、待线下人工审核的 G1 候选清单 |
| `scripts/build_smoke_manifest.py` | 生成确定性 history-only smoke 草稿 |
| `scripts/freeze_benchmarks.py` | 用共享账务引擎生成 benchmark NAV |
| `scripts/prepare_episodes.py` | 生成 Agent-R1 processed Parquet |
| `scripts/verify_h100_stack.py` | 验证版本、CUDA 与真实 FlashAttention kernel |
| `scripts/verify_context_budget.py` | 用缓存模型的真实 tokenizer 验证完整 G1 提示词预算，不启动训练 |
| `src/data/episodes.py` | sampling、manifest 和 Final OOS 隔离 |
| `src/agents/launcher.py` | preflight、门禁与真实启动 |
| `src/agents/local_runs.py` | `runs/<run_id>/` 结构化记录 |
| `src/agents/update_probe.py` | 真实参数更新证据 |
| `SETUP.md` | 更完整的环境与数据背景说明，本文未覆盖时再读 |

## 18. 最终交接定义

对新合作者而言，“可以直接接手”不等于 GitHub clone 后立刻能训练，因为冻结数据和模型因
体量和复现边界不放在 Git 仓库中。它的准确含义是：

- GitHub 提供完整代码、冻结规则、命令和验收标准；
- 维护者只需另行提供冻结 history 快照和 H100 访问；
- 合作者无需依赖历史聊天、W&B、个人电脑路径或数字签名流程；
- 合作者能够独立判断当前为何阻塞、下一步做什么、何时必须停止，以及什么证据才算实验通过。

当前 readiness：`PARTIALLY_READY`。代码、G0、Unicode-safe P0 和 P1 集成已在本机就绪，但本轮
变更尚未冻结为 validation commit，也未在 H100 复验；clean-clone 重装与真实 G1 均未执行。
