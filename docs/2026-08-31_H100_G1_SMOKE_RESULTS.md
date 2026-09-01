# H100 G1 Wiring Smoke 结果（2026-08-31）

状态：`G1 PASSED`

## 技术结论

A1 / Qwen3-1.7B 的单卡 H100 wiring smoke 在代码提交
`2d48bf6db8961998b1450644ef542a293ea8f95d` 上完整结束：rollout、reward、backward 和
optimizer update 均留下了结构化证据，launcher/trainer 最终返回 `0`，且未访问 Final OOS。

本结果只证明训练链路已接通，不证明模型已经学会交易，也不构成收益、Sharpe、显著性或正式训练结论。
本报告生成时，正式 A1/A2/A3 训练仍由 T2 v2 门禁阻断；T2 已于 2026-09-01 通过，见本文末的
后续状态更新。

## 验收身份与范围

| 项目 | 实测值 |
| --- | --- |
| 已验收代码 commit | `2d48bf6db8961998b1450644ef542a293ea8f95d` |
| Agent-R1 gitlink | `b124aa46534cbf2fb8bc8af11405774984c42ac7` |
| 模型 | `Qwen/Qwen3-1.7B` |
| 模型 revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 运行规模 | A1、单 H100 NVL、4 episodes、rollout `n=2`、seed `42` |
| 权威本地 run ID | `20260831T013112864440Z-smoke-a1-qwen3-1p7b-2d48bf6d` |
| 时间 | `2026-08-31T01:31:13Z` 至 `01:56:28Z`，1515.018 秒 |
| 配置哈希 | canonical `2ec544cb4e539cdbc2d3301b255384d3fdd770f5801f8d0ba3194081ac39d844` |
| resolved config 哈希 | `b826d04cc150c3b9e54728f6e7b669c2a48691a9af36f58ca8efe5b19d103753` |

## G1 验收证据

| 门禁 | 实测结果 | 结论 |
| --- | --- | --- |
| 正常终止 | `status=PASSED`、`command_state=FINAL`、`exit_code=returncode=0` | 通过 |
| rollout seed | 259 requests / 259 unique seeds；train 192、validation 67、11 trajectories | 通过 |
| reward 路径 | 第一个 training global step 的 reward/advantage 出现 finite 非零值 | 通过 |
| backward | `backward_count=192` | 通过 |
| optimizer | 2 optimizer steps、2 update records | 通过 |
| 参数变化 | 两次 4096 个参数元素固定样本的 delta L2 为 `6.769137e-6`、`4.462389e-6`，均 finite 且非零 | 通过 |
| 数据边界 | `final_oos_accessed=false`，仅使用 history-only smoke 数据 | 通过 |
| 资源 | peak GPU memory `83,524,895,744` bytes（约 77.79 GiB） | 记录 |

点火前代码门禁也已通过：`181 passed`、Ruff、mypy、`uv lock --check`、diff/context-budget
检查。成功 run 使用 train=4、valid=1；context-budget 覆盖 620 cases，最大 prompt 为
3045/4096 tokens。

## 本轮修复摘要

G1 迭代过程中修复了 vLLM 0.9.2 request drain 与 sleep-cache 生命周期、Ray control-plane hook
隔离、smoke optimizer 状态位置，以及并发 rollout 的逐请求 seed 证据。最后一个 blocker 是 nested
Ray actor 的序列化路径捕获了闭包式 patched 基类方法，导致 actor 进程看不到本地动态注册的
`trading` flow；
[`rollout_seed.py`](../src/agents/rollout_seed.py) 现在只替换 Ray 实际派发的类并保持基类可导入，
[`test_runtime_hooks.py`](../tests/test_runtime_hooks.py) 增加了真实两层 Ray actor 回归路径。

## 证据归档

原始 run 保持在 H100 工作区的 Git-ignored `runs/<run_id>/` 中，不随源码上传。GitHub 中的本报告
只归档关键结果与完整性哈希；需要复核原始日志时，应通过约定的本地文件交接方式取得对应 run。

| 本地证据文件 | SHA-256 |
| --- | --- |
| `summary.json` | `5e895710ca044d9d20f249cdd49d2808057b8935f667dd8e04a0ae22a2647932` |
| `events.log` | `ff82d115554bab3f86919aa76b336ca571ebed054f34b39ab50a65834ffa16f6` |
| `artifacts/preflight_evidence.json` | `152f5e790db1900c355a5ffd5d4f720157a5c7b87a4c441e7e59428fd3a82c3a` |
| `artifacts/update_probe/optimizer-update-413720.jsonl` | `221bda18028f6c871f6d071eb5c2e34c906525de24739fcd43b9ae3bf51c26fb` |
| `artifacts/rollout_seed/request-seeds-414292.jsonl` | `e89b1753e88d973fba49423b9405eb78355a40e8d392dc105827cc8a262b4034` |
| `command.txt` | `ba0dacb989570c2fb8bd126c267db5406c5723e5a708e83188f24b84f30ebc05` |

## 限制与下一步

- 第一个 training global step 确有非零 reward/advantage；第二个 global step 的 train
  reward/advantage 和最终 validation mean reward 为 `0`。第二步同时记录 finite 非零 KL loss、
  gradient 和 parameter delta；这只证明更新链路有效，不能解释为学习质量改善。
- 参数变化来自 4096 个参数元素的固定样本，只证明 optimizer 确实修改了参数，不是全模型质量指标；本次
  `save_freq=-1`，没有生成完整 checkpoint。
- history-only train/valid 不是 strict OOS，且不能排除与模型预训练数据重叠；Final OOS 本次未评估。
- 日志中有一次非致命 Ray worker registration-timeout 记录；随后 worker 正常初始化并完成 run，
  未出现 traceback、OOM 或 terminal error。
- 按 [`EVAL_PROTOCOL.md`](../EVAL_PROTOCOL.md) 的边界，本报告生成时的下一步是 T2 v2 实现与
  评审；该项现已完成。正式训练仍需满足其余独立门禁，且此时仍不访问 Final OOS。

本文件是通过 G1 的代码提交之后生成的纯文档记录；它产生的新 commit 未被该次 smoke 验证，不能把
报告 commit 冒充为已验收代码身份。

## 后续状态更新（2026-09-01）

上述“T2 v2 尚未通过”的表述是本报告生成时的状态。T2 v2 现已在冻结 history Train/Valid 上通过，
详见 [`2026-09-01_T2_V2_RESULTS.md`](2026-09-01_T2_V2_RESULTS.md)。这只解除 T2 blocker，不改变
本报告对 G1 的证据边界，也不自动授权正式训练。
