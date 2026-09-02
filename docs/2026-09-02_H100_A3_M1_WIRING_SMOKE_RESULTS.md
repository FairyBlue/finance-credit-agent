# H100 A3-M1 Wiring Smoke 结果（2026-09-02）

状态：`A3-M1 WIRING SMOKE PASSED`

## 结论与证据边界

A3-M1 / Qwen3-1.7B 的单卡 H100 wiring smoke 在代码提交
`fb07e727b8077c9f25684ad3bfa5222feef84035` 上完整结束。运行真实执行了
M1 immediate-step reward、StepPO step-level GAE、critic 更新、GSPO actor 更新和更新后的
validation；launcher/trainer 最终返回 `0`，且未访问 Final OOS。

这项结果证明 A3-M1 的工程链路和证据门禁已经接通。它不是正式 A3-M1 消融，不能用于收益、
Sharpe、显著性、checkpoint 选择或 A1/A2/A3 优劣比较，也不能把 M1 冒充为 A3 主方法。

协议中的 G1 仍严格指 A1/Qwen3-1.7B wiring smoke，且已于 2026-08-31 独立通过。本次 A3-M1
wiring smoke 是 G1 之后的单独验证；两者的状态和代码身份不能互相替代。

## A3-M1 实现变化

### M1 信号语义与真实 StepPO 接线

- M1 被冻结为 `uniform immediate-step reward v1`。对含 `T` 个已实现 interaction steps 的完整
  轨迹，把终局 `A_fin` 分为 `r_t=A_fin/T`；最后一步吸收浮点余差，使逐步 reward 之和精确等于
  `A_fin`。
- M1 输出是送入 Agent-R1 reward timeline 的 immediate rewards，不是直接或均匀广播的 optimizer
  advantages。上游仍使用 `algorithm.adv_estimator=gae`、真实 critic 和 GSPO actor loss。
- 空轨迹、截断轨迹、非有限 `A_fin`、非守恒 M1 向量、错误 scientific role 或配置漂移都会
  fail closed。
- M2/M3 只保留显式接口并继续抛出 `NotImplementedError`；没有把 M1 重命名或包装成 A3-main。

### 轨迹与更新证据门禁

- 每条完整轨迹写一条 step-credit JSONL，包含实际 action 序列、step 数、`A_fin`、逐步 credit、
  Train/Valid 身份和守恒误差。
- launcher 要求 Train 和 Valid 都有完整、有限、非零的信号，并要求 step-credit 与逐请求 rollout
  seed 证据的一对一 trajectory identity SHA-256 完全一致。
- optimizer probe 为更新记录增加 `role=critic|actor`。A3-M1 成功不再只要求“有一条更新”，而是
  必须同时观测到 critic 和 actor 的有限、非零参数变化。
- run summary 会同时保存 preflight、rollout seed、step credit、双角色 update、最终退出和
  Final OOS 隔离证据；仅有退出码 `0` 不足以判定通过。

### Wiring-only 资源配置

成功 smoke 使用以下 wiring-only 的规模、设备放置与 reference 前向精度控制；它们不是正式实验
超参数：

| 配置 | 值 | 作用 |
|---|---:|---|
| `training.rollout_n` | `1` | 每个 prompt 只生成一条带确定性 request-seed 证据的 wiring 轨迹 |
| `training.train_batch_size` | `1` | 只训练一条完整、非零轨迹 |
| `training.total_training_steps` | `1` | 保留一次完整 critic → actor → post-update validation |
| `training.critic_param_offload` | `false` | 让约 6.41 GiB critic 参数常驻 H100，避免 host `memory.high` 回收风暴 |
| critic optimizer offload | `true` | Adam 状态继续按上游正式管线放在 host，不改变优化器语义 |
| `training.reference_model_dtype` | `bf16` | 与 BF16 forward dtype 对齐，并减少 forward-only reference 的 offloaded host 存储 |

初始实现已经能产生 M1 轨迹和 critic 更新，但诊断显示 host `memory.high` 造成严重回收压力和
控制面延迟；第一次资源修复让 critic 参数常驻 GPU，并把 smoke 限为一个 trainer step。第二轮
把 forward-only reference 显式设为 BF16、把 training batch 从 2 缩到 1，最终保留真实训练语义
的同时把成功 global step 控制在约 154 秒。

## 实现提交

| Commit | 内容 |
|---|---|
| `dca22c5a317e1ce059f78255c544b031387feacf` | 完成 M1 strategy、A3 flow、真实 GAE 接线、step-credit/trajectory evidence、`rollout_n=1` 的 A3-M1 smoke 身份与测试 |
| `d99e7d64a63c0285217f6605e0cc54cd5208e704` | 增加 critic/actor 双角色更新门禁；设置 1 trainer step、critic 参数常驻 GPU |
| `fb07e727b8077c9f25684ad3bfa5222feef84035` | 设置 `train_batch_size=1` 和 BF16 forward-only reference；更新冻结 smoke 身份和配置测试 |

## 验收身份与范围

| 项目 | 实测值 |
|---|---|
| 已验收代码 commit | `fb07e727b8077c9f25684ad3bfa5222feef84035` |
| Agent-R1 gitlink | `b124aa46534cbf2fb8bc8af11405774984c42ac7` |
| 模型 | `Qwen/Qwen3-1.7B` |
| 模型 revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 硬件 | 单 NVIDIA H100 NVL |
| 运行规模 | 4 个可用 Train rows；实际 1 条 Train 轨迹、更新前后各 1 条 Valid 轨迹；seed `42` |
| Trainer | `training/global_step=1`、`rollout_n=1` |
| 权威本地 run ID | `20260902T070540291149Z-smoke-a3-m1-qwen3-1p7b-fb07e727` |
| 时间 | `2026-09-02T07:05:40Z` 至 `07:30:31Z`，1490.826 秒（包含强制 preflight 和模型初始化） |
| 完整 merged config SHA-256 | `c2ac6b396e5eea457c6577be438a452d71e93ce90872c089aa0b41b151aa0ccd` |
| 注册 smoke identity SHA-256 | `ed7237bbf4769095ec56095f017e88d3396509713af149269955a024248996a2` |
| resolved config 文件 SHA-256 | `080e65592dc081f9816b65aea1640640eb012d077c9091662961f15511ee8eca` |

两个 config hash 的 canonicalization 范围不同：run recorder 对完整 resolved config 取 hash；注册
smoke identity 会排除可迁移的 `tracking.log_root`。两者同时出现是预期行为，不是配置漂移。

实际点火命令为：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.agents.arm_a3 \
  --config configs/smoke_a3_m1_1p7b.yaml \
  --expected-commit fb07e727b8077c9f25684ad3bfa5222feef84035
```

launcher 在同一次 actual run 内重新执行了 fail-closed preflight，确认 exact commit、干净 tracked
tree、Agent-R1 gitlink、模型 revision、冻结配置身份、T2 v2、history manifests、processed rows、
620 个 prompt-budget cases 和 Final OOS 隔离后才启动 Ray/Agent-R1。

## Wiring smoke 验收证据

| 门禁 | 实测结果 | 结论 |
|---|---|---|
| 正常终止 | `status=PASSED`、`command_state=FINAL`、`exit_code=returncode=0` | 通过 |
| Trainer step | `training/global_step=1` | 通过 |
| rollout seed | 63 requests / 63 unique seeds；Train 20、Valid 43；共 3 条轨迹 | 通过 |
| M1 step credit | 1 条 Train + 2 条 Valid；63 steps；3/3 完整且守恒 | 通过 |
| 轨迹身份 | seed 与 credit 的 `trajectory_identity_sha256` 都为 `526755cd...eed4d` | 通过 |
| 真实 GAE | advantage mean/min/max 为 `0.0170479 / -1.42425 / 2.10638`，不是均匀 optimizer advantage | 通过 |
| critic 更新 | 独立 `role=critic` record，有限非零梯度与参数变化 | 通过 |
| actor 更新 | 独立 `role=actor` record，有限非零梯度与参数变化 | 通过 |
| post-update validation | `step=1, validate=true` 的完整 22-step Valid 轨迹 | 通过 |
| 数据边界 | `final_oos_accessed=false`，只使用 history smoke Train/Valid | 通过 |
| GPU 资源 | peak allocated `93,829,628,928` bytes（约 87.39 GiB） | 记录 |

### Step-credit 轨迹

| 阶段 | Sample | Steps | `A_fin` / credit sum | 守恒误差 |
|---|---:|---:|---:|---:|
| Initial Valid | 0 | 21 | `-55.28775219319622` | `0.0` |
| Train | 2 | 20 | `1.5457017480377786` | `0.0` |
| Post-update Valid | 0 | 22 | `-55.28775219319622` | `0.0` |

三条轨迹均为有限非零信号。两次 Valid 的 `A_fin` 相同，但 action/step 数不同；各自的实际轨迹、
credit 和 rollout-seed identity 都完整记录。该结果只证明更新后的模型同步和 validation 链路执行，
不代表性能改善。

### 双角色 optimizer 更新

| Role | Record 中的累计 backward count | Grad norm | Parameter delta L2 | Parameter delta max |
|---|---:|---:|---:|---:|
| critic | 20 | `906.3319091796875` | `2.444831989123486e-05` | `9.995885193347931e-06` |
| actor | 40 | `4.048514366149902` | `6.1614555306732655e-06` | `1.000240445137024e-06` |

`backward_count` 是同一 worker 内的累计探针计数；成功判定依据是两个不同 role record 都有 finite、
non-zero gradient 和 parameter delta，而不是把累计 40 误读为 40 个 actor-only batches。

### 资源观察

- critic FSDP 初始化后实测占用约 6.41 GiB device memory，证明 `critic_param_offload=false` 生效。
- 成功 global step 的 `timing_s/update_critic=14.668`、`timing_s/update_actor=5.290`，完整 Train
  generation、更新和 post-update validation 共约 153.84 秒。
- 外部 cgroup 采样从点火前到进程退出均为 `oom=0`、`oom_kill=0`，`memory.high` 事件计数没有
  新增；该 host 采样不属于 launcher summary 的持久字段，因此只作为资源诊断记录，不是算法门禁。

## 本地门禁与预检

最终资源配置提交点火前完成：`246 passed`、Ruff、全 `src` mypy、冻结 smoke identity、lock 与
diff checks。精确 commit 的独立 preflight-only run 也已通过；actual run 随后再次通过相同的强制
preflight，且两次都没有访问 Final OOS。

## 证据归档

原始 run 保持在 H100 工作区的 Git-ignored `runs/<run_id>/` 中，不随源码上传。GitHub 中的本报告
只归档关键结果与完整性哈希；复核原始日志时需取得同一 run 目录。

| 本地证据文件 | SHA-256 |
|---|---|
| `summary.json` | `99329dbf4c624a611a06505af96164fbdbace5acf221b186ebcf5cb27d908fdf` |
| `events.log` | `08e7a8bd273b0c2e174928d8da50687b0b98fbc84d075df82b433cc9d3c37519` |
| `artifacts/preflight_evidence.json` | `ef971d34efbbd18d4ba869ab5dbb0b7e4cdd0397c3682e6cf23224ccd540e786` |
| `artifacts/update_probe/optimizer-update-487042.jsonl` | `4875dd6f06a378e2e7a0463adcc4ffa49d43afac3378453aad477d184a805457` |
| `artifacts/rollout_seed/request-seeds-487978.jsonl` | `6b7768a4e8ffc3b6a9117545731ba87fc4ac4441d61828c410ee147a2affe33a` |
| `artifacts/step_credit/step-credit-487978.jsonl` | `49a36684567c43f64ac0c05bd7ed3ede878dbac8a23356e2c30856d7109d2487` |
| `command.txt` | `1e054cfe0e7f30166f1a47a4c8428a240c74d7f7ae717c9e05410d6aa4607b2b` |
| `resolved_config.yaml` | `080e65592dc081f9816b65aea1640640eb012d077c9091662961f15511ee8eca` |

## 当前状态

| 项目 | 当前状态 |
|---|---|
| 协议定义的 G1（A1 wiring smoke） | `PASSED`；独立绑定 commit `2d48bf6...` |
| A3-M1 wiring smoke | `PASSED`；独立绑定 commit `fb07e727...` |
| A3-M1 正式消融 | 尚未运行 |
| A3-main | `BLOCKED`；M2/M3 和 M1 superiority statistic 尚未完成 |
| Final OOS | 继续关闭 |

## 限制

- 本次 `save_freq=-1`，没有保存完整 checkpoint。4096 个固定参数元素上的非零 delta 只证明
  optimizer 确实修改了参数，不是全模型质量指标。
- history-only smoke Train/Valid 不是 strict OOS，也不能排除与模型预训练数据重叠。
- 日志出现过一次非致命 Ray worker registration-timeout；随后 worker 正常注册并完成完整 run，
  没有 terminal traceback、OOM 或控制面失联。
- 外部 host-memory 采样没有写入 hash-bound run summary；可复核的持久资源证据以 summary 中的
  GPU peak 为准。
- 本次 Valid reward 只用于证明更新后的模型同步和 validation 链路执行；不得把更新前后的 reward
  或轨迹差异解释为性能提升。

## 下一步

下一步仍是实现并预注册 A3-main 的 M2 或 M3、由两位合作者线下冻结 M1 superiority statistic，
随后冻结正式执行身份并完成 formal H100 preflight。只有获得明确授权后才能启动正式四配置比较。

本文件在已验收代码提交之后生成；报告自身的文档 commit 不能冒充 smoke 的 exact validated commit。
