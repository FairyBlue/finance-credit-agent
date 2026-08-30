# H100 G1 P1 验收任务单（2026-08-30）

状态：`WAITING_FOR_VALIDATION_COMMIT`。本文只验收 P0/P1 和 G1 接线，不授权正式训练，不访问
StockBench Final OOS。本轮改动提交后，由维护者在线下把该提交的完整 40 位 SHA 作为命令参数
发给执行者；执行者只在 shell 中替换下面占位值，不修改本任务单，验证期间不得改用浮动 `main`。

```bash
export FCA_VALIDATION_COMMIT=REPLACE_WITH_EXACT_40_CHAR_COMMIT
git fetch origin
git checkout --detach "$FCA_VALIDATION_COMMIT"
git submodule update --init --recursive
git rev-parse HEAD
git status --short
git submodule status
```

预期：HEAD 等于外部给定 SHA、tracked tree clean，Agent-R1 为
`b124aa46534cbf2fb8bc8af11405774984c42ac7`。reviewed manifest 和 processed data 是允许的线下
输入；`src/`、`scripts/`、`runtime_hooks/` 或其他 `configs/` overlay 会被 launcher 拒绝。

## 1. 冻结环境与 exact 模型

```bash
uv python install 3.10.20
uv venv --python 3.10.20
uv sync --frozen --extra gpu --extra data
uv pip check --python .venv/bin/python
uv run --no-sync python scripts/verify_h100_stack.py

uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
))
PY
```

不得用 `revision="main"`，不得安装 CUDA Toolkit，不要求系统 `nvcc`，不得单独升级 GPU 栈。

## 1.5 冻结的 G1 本地输入

G1 只复制 history，不复制或访问 StockBench。把冻结快照放到 `data/history/storage/` 后先核对：

```bash
uv run --no-sync python - <<'PY'
from src.agents.config import load_experiment_config
from src.agents.launcher import verify_configured_source_manifests

config = load_experiment_config("configs/smoke_1p7b.yaml")
audit = verify_configured_source_manifests(config, repo_root=".")["history"]
assert audit.manifest_sha256 == "2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90"
assert audit.file_count == 10820
assert set(audit.sessions_per_symbol.values()) == {541}
print("HISTORY_MANIFEST_VERIFIED", audit.manifest_sha256, audit.file_count)
PY
```

然后由两位合作者线下审核 candidate；审核记录不由自动 agent 代签。通过后逐字节复制并生成
derived smoke 数据：

```bash
cp configs/episode_manifest.smoke.candidate.yaml \
  configs/episode_manifest.smoke.reviewed.yaml
cmp configs/episode_manifest.smoke.candidate.yaml \
  configs/episode_manifest.smoke.reviewed.yaml
uv run --no-sync python scripts/prepare_episodes.py \
  --config configs/smoke_1p7b.yaml \
  --manifest configs/episode_manifest.smoke.reviewed.yaml
test -f data/processed/smoke_train.parquet
test -f data/processed/smoke_valid.parquet
test -f data/processed/smoke_test.parquet
```

任一 hash、文件数、session 数或 `cmp` 不匹配都立即停止；不得改配置 SHA 迁就本地数据。

## 2. 代码与 prompt 门禁

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts runtime_hooks
uv run --no-sync mypy src tests
uv lock --check
git diff --check

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/verify_context_budget.py \
  --config configs/smoke_1p7b.yaml
```

context 输出必须包含 Train 与 Valid，并覆盖 `planned-ascii`、`legal-unicode-max`、
`max-positions-mixed-bounded`、`invalid-unicode-unknown-tool`；所有 prompt 必须同时低于 4096 的
prompt limit，且 prompt 加 2048 response 不超过模型 context。

## 3. 完整 preflight

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --preflight-only \
  --expected-commit "$FCA_VALIDATION_COMMIT"
```

只有同时出现 `PREFLIGHT_PASSED`、`No training was started.`、`Final OOS not accessed.` 才通过。
检查最新 `runs/<run_id>/artifacts/preflight_evidence.json`：应包含 validation identity、exact model
revision/snapshot、source/reviewed/processed hashes 与行数、逐 split/scenario token maxima、最终命令
hash，以及两个 offline flag。`command.txt` 不得含 `Qwen/Qwen3-1.7B` repo ID，必须含本地 exact
snapshot，并包含 `data.train_max_samples=4`。`summary.json.command_state` 必须是 `FINAL`；失败 run
的候选命令只在 `provisional_command.txt`，不能冒充 final command。

配置身份应分别记录：canonical merged JSON SHA-256
`2ec544cb4e539cdbc2d3301b255384d3fdd770f5801f8d0ba3194081ac39d844`，以及当前
`resolved_config.yaml` 字节 SHA-256
`b826d04cc150c3b9e54728f6e7b669c2a48691a9af36f58ca8efe5b19d103753`。后者必须以实际 run 文件
重新计算核对；两者不是同一个 hash。

## 4. G1 点火（前三关通过后才允许）

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml \
  --expected-commit "$FCA_VALIDATION_COMMIT"
```

不得使用 `scripts/run_queue.sh`；它被设计为对所有参数 fail closed。G1 通过要求退出码 0、
Final OOS 未访问、rollout/reward 出现、`backward_count >= 1`、`optimizer_step_count >= 1`、gradient
finite、parameter delta finite 且非零。`observed_update_record_count` 只是 probe 记录数，不是独立
trainer global step 证据。收益、Sharpe 和显著性不属于 G1 验收。

任一步失败：停止后续步骤，保留完整 run 目录和首个错误。不要降低门禁、改配置、启动正式训练、
实现 T2 v2，或读取 Final OOS。
