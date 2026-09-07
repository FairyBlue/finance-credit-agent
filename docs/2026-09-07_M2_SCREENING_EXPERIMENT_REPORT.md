# M2 screening 当前实验报告（2026-09-07）

**当前结论：实现和部分工程验证已完成，M2 screening 尚未跑通，效果结论为 INCONCLUSIVE。**
八个主 screening cells 完成 **0/8**；没有 screen-dev 评测，也没有正式 Valid / Final OOS 结果。
不能据此判断 M2 是否优于 M1、M2-plain 或 A2。

截至 **2026-09-07 05:53 UTC**，按用户要求停止实验推进。已核实没有运行中的 `fca-m2-*` 服务，
没有 GPU compute 进程。R2 修复已经提交并完成下述测试，**没有启动 R2 GPU 重跑**。

**实验范围与实现**

本轮依据 [M2 实验计划 v2](2026-09-05_M2_EXPERIMENT_PLAN_V2.md) 和用户的 screening 执行授权。
主设计为 A2、M1、M2-plain、M2 四臂，Qwen3-4B，seeds 42/43；每个 cell 为 32 个 Train
episodes、每 episode 四 siblings、16 次更新，最终在 12 个历史 Train 内的 dev episodes 上
greedy 评测。1.7B 仅用于接线；工程 pilot 不计为主筛选结果。

已实现并提交：

- 独立数据准备、Train-only 校验、新 scorer 标定及数据身份校验；主数据为 32/12，pilot 取同样
  最前两个 Train windows。没有使用正式 Valid 或 Final OOS 选择参数。
- 四臂共用 flow、reward-neutral 观察、严格轨迹完整性和 sibling 身份检查。A2 训练使用 terminal
  raw net return；其余臂使用 uniform Y；最终 evaluator 使用共同 Y。
- M2-plain / M2 residual 在 StepPO GAE 后、whitening 前加入 actor advantage；critic returns
  保持原定义，对完整 logical batch 的真实 steps 做一次 whitening。
- 串行 supervisor、实际 optimizer 更新探针、轨迹及机制证据、完整 checkpoint 标记、最终
  evaluator、完整网格 readout 和完成 cell 的模型归档。后两项已有实现，尚未由完整 4B cell 验证。

CPU 标定使用 **1,408 条代理轨迹、28,160 个账务 steps**，得到 volatility floor
`0.0007613112285332427`、λ=`1.6912508025279984`。calibration payload SHA-256：
`eaf1e50207ba727536a7fabb21d0e3ee3be12bd7569bd3e21e5d5696ab421a7e`。
这些是 Train 标定结果，不是收益证据。实现入口见 [运行手册](2026-09-06_M2_SCREENING_RUNBOOK.md)。

**真实运行记录**

下表的运行均发生在 2026-09-06，时间为 UTC。每次使用独立目录保留失败，未拼接样本补齐结果。

| 运行 | 时间 | 实际进展 | 停止原因 / 状态 |
|---|---|---|---|
| v1：启动检查 | 13:25–13:26 | 未启动模型训练 | 身份检查拒绝历史遗留的 ignored M1 smoke manifest；随后限定豁免并记录其 hash |
| v2：启动检查 | 13:28–13:30 | 未开始模型训练 | Hydra 将 seed 字符串转成整数，Ray runtime env 拒绝；随后修复并验证真实解析链 |
| v3：1.7B A2 wiring | 13:35–13:59 | 8/8 Train 轨迹完整，479 steps；actor / critic 各实际更新一次 | post-update Train greedy 触及 2048 response cap；0/2 完整评估，未保存最终 checkpoint |
| v4：R1，1.7B M2 wiring | 14:25–14:51 | 8/8 Train 轨迹完整，538 steps；两类参数各更新一次；完整 checkpoint 已保存 | post-update Train greedy 触及 4096 cap；0/2 完整评估，整轮 INCOMPLETE |
| v4-pilot：R1，4B M2 诊断 | 14:53–15:13 | 首轮 8/8 Train 轨迹完整，190 steps；占计划两轮的 8/16 | `nvidia-smi` 超时被启动器错误传播，继而 SIGTERM trainer；无实际更新、checkpoint 或评估完成证据 |

两次 1.7B 评估均有一条明确截断，**不是一条成功、一条失败**。v3 的另一条没有完整结果记录，
v4 的另一条有随批次失败而未完成的记录。
共完成 24 条工程训练轨迹，但它们重复使用同两个 Train windows，不是 24 个独立数据窗口。
M1 / M2-plain 的新四臂 GPU 接线、4B M1 pilot、八个主 cells、zero-shot dev 和 dev CPU 基线均未完成。

v3 代码为 `6be14b424db6fa45cfb09f893ce1199b67624b34`；两个 R1 GPU 运行均为
`b90bde9ffee5953c17b6b55b6aba82b152f6e461`。模型固定为：

| 用途 | 模型 | exact revision |
|---|---|---|
| 接线 | Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| screening / 诊断 pilot | Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |

**M2 机制已经验证到哪一步**

v4 的 1.7B M2 提供了实际训练证据：两个 occurrence UID，各四 siblings；40 个 UID×session
组均完整，每条 episode 20 个推进决策。uniform credits 的总和与终局 Y 相等。

| 检查项 | 真实 M2 batch 结果 |
|---|---:|
| 真实 steps / 推进决策 | 538 / 160 |
| diversity gate 开启 | 14/40 组，35% |
| 非零 pooled residual | 24/160 推进行，15% |
| pooled 与 same-gate plain 不同 | 15/160 推进行，9.375% |
| 组内 pooled residual 零和最大误差 | 约 2.22×10⁻¹⁶（独立 float64 重算） |
| actor / critic 实际更新 | 各一次；参数变化探针均检出变化 |

独立重算的 residual 统计与实际 advantage 诊断一致；whitening 的 scope、sample std、epsilon
及完整 batch 方差关系也吻合源码和契约测试。因此，可以确认本轮 M2 接线实际产生了机制差异，
并进入了参数更新链路。**这不证明机制改善了收益。**

4B 首轮虽然完整生成了 8 条轨迹，但没有留下实际 advantage 或 optimizer 证据。在这些轨迹上
离线计算得到的 gate 9/40、非零 residual 36/160、pooling 改变 29/160，均仅是假设计算，不能
写成 4B M2 已训练成功。其生成量为 229,959 response tokens，invalid steps 为 7。

**当前阻碍与修复状态**

1. **1.7B greedy 的实际生成未完成。** v4 的 `formal-train-0001` 第 45 次请求含 2282 prompt
   tokens，生成恰好 4096 tokens。解码显示模型在 `<think>` 内反复讨论空仓及工具选择，未闭合
   reasoning、未生成 tool-call，末 token 不是 EOS。上游 `stop_reason=completed` 不能消除这一
   实际截断。R1 已按共同设置把 cap 从 2048 提高到 4096；再次失败后没有继续自动扩大预算。
   该现象发生在更新后，但尚无证据将其因果归于训练更新或 M2 residual。
2. **4B 被监控故障终止。** 直接故障链已确认：`nvidia-smi` 超过 15 秒 → 未捕获异常 →
   `capture_process` 的 finally 向仍存活的 trainer 发送 SIGTERM。未观察到此前的模型异常或
   OOM。监控命令为何超时尚未确定，不能将“同时存在内存压力”写成已证实的因果关系。
3. **4B 主机内存确有压力，完整容量仍未知。** 成功采样中的主机峰为 58.20 GiB，同一采样窗口
   内 `memory.high` 增加 132,878 次，OOM 计数未增加。当前主机上限 64 GiB、high 57.6 GiB、
   无 swap。原日志只能定位
   到 rollout 结束之后、实际 advantage 证据之前，不能分清 old-logprob、reference 或 value。
   reference 的 FSDP CPUOffload 会分配额外 CPU gradient，这是源码确认的潜在压力来源，尚未
   证明它就是本次转折点。4B 两次更新及 checkpoint 的峰值没有实测通过。

| 资源观测 | 1.7B M2 v4 | 4B M2 v4-pilot |
|---|---:|---:|
| 主机 `memory.current` 采样峰 | 54.44 GiB | 58.20 GiB |
| GPU 物理使用采样峰 | 57.77 GiB | 65.73 GiB |
| 成功采样窗口内新增 `memory.high` / OOM | 0 / 0 | 132,878 / 0 |
| 完整更新 / checkpoint | 已完成一轮 / 已保存 | 未完成 / 未保存 |

这些是约 10 秒加命令耗时的采样峰，不能视为精确瞬时峰；也没有使用历史累计 `memory.peak`
冒充本轮峰值。installed PyTorch 的 host allocator allocated 计数还存在回收路径异常，不能
据其 allocated−reserved 推算 live/cache，容量判断应以 cgroup、PSS 和真实阶段证据为主。

已提交的 [R2 工程修复](2026-09-07_M2_SCREENING_EXECUTION_AMENDMENT_R2.md) 为
`b6f2644223a8ff5edb516a9f1e4bd6c1d6bae29b`：

- GPU 遥测 timeout / NVML 非零退出 / OS 错误记为 `gpu_error`，继续训练；保留真实训练异常与退出码。
- 为 old-logprob、reference、value、critic/actor 更新增加 enter/success/error 与 cgroup 日志。
- 显式设置 CUDA graph shapes `[1,2,4,8]`。R1 的 `max_num_seqs=8` 实测仍预建 67 个 shapes，
  因而先前减少 graph 数量的预期没有兑现；R2 的实际 vLLM CLI 解析检查已通过。

R2 通过 **30 项 launcher 测试 + 45 项资源日志 / runtime 测试**；Ruff 通过，mypy 对 91 个
源文件通过。这里报告的是相关回归检查，不宣称重新执行了整个项目的全部测试。
R2 未进行 GPU 重跑，其资源收益、第二次更新和最终 evaluator 均未验证。候选 optimizer / reference
offload 调整仅有分析草案，未启用。

**断线隔离、保留材料与恢复顺序**

训练使用 H100 本机的 systemd user service；已核验 `Linger=yes`、`stdin=null`、独立 service
cgroup 和文件日志。运行不依赖 SSH 终端存活。没有专门做主动断网注入测试；本次已确认的 4B
停止原因是启动器传播监控超时。按用户停止指令，不再排队启动新实验。

v4 的完整 1.7B M2 checkpoint 保留在
`runs/m2-screening-20260906-v4/wiring-m2-s42/checkpoints/global_step_1/`，含完成标记；未因
评估失败删除。完整标记不等于已经证明真实故障后的 exact resume。主要证据入口：

| 材料 | 本机位置 |
|---|---|
| 最新停止状态 | [M2_SCREENING_LATEST.md](../runs/M2_SCREENING_LATEST.md) |
| 1.7B M2 机制、更新与失败说明 | [v4 工程简报](../runs/m2-screening-20260906-v4/ENGINEERING_OBSERVATIONS.md) |
| 4B 故障与资源审计 | [4B 工程简报](../runs/m2-screening-20260906-v4-pilot/ENGINEERING_OBSERVATIONS.md) |
| 4B 原始摘要 | [summary.json](../runs/m2-screening-20260906-v4-pilot/pilot-m2-s42/summary.json) |
| 1.7B 截断响应诊断 | [failure_analysis.json](../runs/m2-screening-20260906-v4/wiring-m2-s42/failure_analysis.json) |

`runs/` 是忽略目录；上述链接指向 H100 本机证据，不表示这些大体积日志或权重已上传 GitHub。

恢复工作时，优先用已修复的 R2 跑完 4B M2 的两个更新及同 worker 最终 Train 评估，定位主机
压力；如需调整 offload，先验证数值一致性和完整生命周期。随后处理 1.7B 接线中的 greedy 循环、
补齐其余必要接线与 4B M1 pilot。涉及 prompt / thinking / 生成限制的变更应先写入共同设计，
不通过删行、补零或重新抽样把原失败改成成功。所有门禁通过后，才具备启动八个主 cells 的条件。
