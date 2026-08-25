# Finance Credit Agent experiment handoff

Updated: `2026-08-25`

This is the operational source of truth for handing the repository to a new operator. It explains
what may run now, what is blocked, what files must be supplied, and what evidence makes the G1 smoke
valid. Repository configuration, code, Git state, and test output take precedence if this document
ever drifts.

## 1. Purpose

The project asks whether assigning credit to individual decisions improves a tool-using trading
agent over rewarding only the final portfolio outcome. It compares three arms:

- A1: outcome-only GRPO;
- A2: generic step-level credit;
- A3: finance-specific step-level credit.

The immediate task is narrower: prove that the A1/Qwen3-1.7B training chain is wired correctly on
one H100. This G1 wiring smoke is not formal training and cannot support a return, Sharpe, model
selection, or arm-comparison claim.

## 2. Authoritative frozen decisions

- Official Train: `2023-01-03` through `2024-10-31`.
- Official Valid: `2024-11-01` through `2025-02-28`.
- Final OOS: StockBench `2025-03-03` through `2025-06-30`, 83 XNYS sessions, reserved for final
  scientific evaluation.
- G1 Smoke Train: first 36 Train sessions, four contiguous non-overlapping episodes of nine
  sessions each.
- G1 Smoke Valid: `valid_dates[0:9]`, `2024-11-01` through `2024-11-13`.
- G1 Smoke Shadow Test: `valid_dates[9:18]`, exactly nine sessions from `2024-11-14` through
  `2024-11-26`.
- The shadow test is history-backed and wiring-only. Its frozen provenance is
  `split_role=smoke_shadow`, `source_split=valid`, `purpose=wiring_only`,
  `scientific_use_allowed=false`, `checkpoint_selection_allowed=false`, and `final_oos=false`.
- W&B is disabled. Every run writes structured evidence under `runs/<run_id>/` and is synchronized
  with rsync.
- T2 v2 remains preregistered but unimplemented. It blocks every formal run. Only the exact
  hash-bound A1/Qwen3-1.7B wiring smoke is exempt.

The approved merged smoke configuration SHA-256 is
`6c58a822d3a6ff2751fbbfcda9451c9ad3eae9ffe81acb24d3417c3de40fe0ed`.

## 3. Gate status

| Gate | Status | Meaning |
|---|---|---|
| G0 protocol and local scaffold | `PASSED` | Protocol, shadow isolation, local recorder, preflight, optimizer probe, 126 tests, Ruff, mypy, lock, and whitespace validation pass locally. |
| G1 A1/Qwen3-1.7B wiring smoke | `BLOCKED` | No real H100 smoke has been run from this worktree. Required reviewed manifest, processed files, GPU/model availability, and clean committed Git state are not all verified here. |
| T2 v2 formal-training gate | `BLOCKED` | Implementation and passing train/valid result are intentionally outside this task. |

G1 becomes `IN_PROGRESS` only after all inputs exist and `--preflight-only` passes on the H100.
G1 becomes `PASSED` only after a real smoke records a successful rollout, reward, at least one
backward pass, at least one optimizer step, and a finite non-zero parameter delta. Dry-run,
preflight, model loading, or rollout without an update is not enough.

## 4. Final OOS fail-closed boundary

Smoke startup fails if any resolved source, path, manifest, processed row, provenance field, or date
points to StockBench/Final OOS. The launcher does not verify the StockBench source or run the
history/StockBench overlap audit during smoke, because touching that source is unnecessary. The
guard checks content provenance as well as names, so renaming an OOS file to `smoke_test.parquet`
does not make it acceptable.

The Final OOS remains available to the formal evaluation pipeline only. Do not use it for the smoke,
T2 v2, feature selection, hyperparameters, checkpoints, or power analysis.

## 5. Inputs that must exist on H100

The repository expects the following external or reviewed inputs:

- `data/history/storage/`: the frozen 20-stock history snapshot whose raw manifest SHA-256 is
  `2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90`;
- `configs/episode_manifest.smoke.reviewed.yaml`: a human-reviewed history-only smoke manifest;
- `data/processed/smoke_train.parquet`;
- `data/processed/smoke_valid.parquet`;
- `data/processed/smoke_test.parquet`, whose embedded provenance says Valid shadow/wiring only;
- locally cached `Qwen/Qwen3-1.7B` model files;
- the Linux/x86_64 GPU extra, CUDA, and one visible H100-class GPU.

Raw data, processed data, models, runs, checkpoints, and artifacts remain ignored and must not be
committed.

## 6. Build and review the smoke manifest

Generate a deterministic draft from the frozen exchange calendar. This does not read StockBench or
Final OOS:

```bash
uv run python scripts/build_smoke_manifest.py \
  --config configs/smoke_1p7b.yaml \
  --output configs/episode_manifest.smoke.draft.yaml
```

Review every episode date/open time and the history snapshot identity. The review must explicitly
cover later-split reversal and restored volume logic. Then freeze benchmark NAV using the same
next-open ledger as the agent:

```bash
uv run python scripts/freeze_benchmarks.py \
  --config configs/smoke_1p7b.yaml \
  --draft configs/episode_manifest.smoke.draft.yaml \
  --output configs/episode_manifest.smoke.reviewed.yaml

uv run python scripts/prepare_episodes.py \
  --config configs/smoke_1p7b.yaml \
  --manifest configs/episode_manifest.smoke.reviewed.yaml
```

The reviewed manifest is a normal repository artifact only after human review. Do not relabel a
generated draft as reviewed automatically.

## 7. Sync without cloud experiment tracking

Use placeholders rather than machine-specific usernames or directories:

```bash
export H100_HOST=<ssh-host-alias>
export H100_PROJECT_DIR=<absolute-project-directory-on-h100>

rsync -a --partial --info=progress2 data/history/storage/ \
  "$H100_HOST:$H100_PROJECT_DIR/data/history/storage/"

rsync -a --partial --info=progress2 \
  "$H100_HOST:$H100_PROJECT_DIR/runs/" runs/
```

No W&B login, API key, cloud logger, digital signature, or extra cryptographic signing ceremony is
required. Existing snapshot and Git hashes are experiment identity checks, not a user-signature
scheme.

## 8. Commands to run

CPU-safe command construction:

```bash
uv run python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml --dry-run
```

H100 preflight, which performs no training:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml --preflight-only
```

Successful preflight prints all three lines:

```text
PREFLIGHT_PASSED
No training was started.
Final OOS not accessed.
```

Real approved smoke:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.agents.arm_a1 \
  --config configs/smoke_1p7b.yaml
```

The queue wrapper may be used for the same one config:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_queue.sh configs/smoke_1p7b.yaml
```

Do not run the no-argument A1/A2/A3 formal queue while T2 v2 is blocked.

## 9. Run evidence and acceptance

Every attempted non-dry run creates:

```text
runs/<run_id>/
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

Failures retain `summary.json` with the error. Real training installs an opt-in runtime probe in the
launcher and Ray workers. It samples a trainable parameter with finite non-zero gradient before the
optimizer step and records before/after hashes, gradient norm, parameter delta L2/max, backward
count, optimizer-step count, and global-update count. A zero or missing finite delta fails G1.

For acceptance, inspect `summary.json` and require:

- `status` is `PASSED` and return code is zero;
- `final_oos_accessed` is false;
- optimizer evidence has `detected=true`;
- backward, optimizer-step, and global-update counts are positive;
- the recorded Git/submodule/config/data identities match the frozen values;
- rollout and reward events contain no collapse/watchdog failure.

Full checkpoints are optional for G1; direct update evidence is mandatory.

## 10. Validation before handoff

Run from repository root:

```bash
uv run pytest -q
uv run ruff check src tests scripts runtime_hooks
uv run mypy src tests
uv lock --check
git diff --check
uv run python -m src.agents.arm_a1 --help
uv run python -m src.agents.arm_a1 --config configs/smoke_1p7b.yaml --dry-run
```

On a CPU-only Mac, `--preflight-only` is expected to stop at the GPU-extra/CUDA check and write a
failed local summary. That is a valid CPU-path exercise, not G1 progress. The complete preflight and
real optimizer probe remain H100-only checks.

## 11. Remaining work after this scaffold

1. Finish human review of the smoke manifest and split-restoration logic.
2. Sync the history snapshot and repository to the H100 host.
3. Install the locked GPU extra and cache Qwen3-1.7B locally.
4. Commit the repository only when explicitly authorized; preflight may intentionally require a
   committed clean state according to the final run policy.
5. Run H100 preflight. If it passes, mark G1 `IN_PROGRESS`.
6. Run the one approved smoke and inspect its structured evidence. Mark G1 `PASSED` only if every
   acceptance condition above holds.
7. Separately implement and pass T2 v2 before any formal A1/A2/A3 training.

Current overall readiness is `PARTIALLY_READY`: the local scaffold is implemented, but the worktree
is uncommitted and H100-only evidence does not exist.
