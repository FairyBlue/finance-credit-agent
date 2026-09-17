# 阶段二：M2 screening 工程与生成完整性（9 月 6—9 日）

整理日期：2026-09-14。汇总最初筛选、R1—R5 恢复、生成诊断和磁盘处理；
[R1—R5 原修订](../README.md#保留的设计与协议)仍保留，详细过程见[原文归档](../archive/README.md)。

截至本阶段结束，**R5 的六个工程 cells、76 条轨迹通过，主 screening 仍无完整结果**。
主要阻塞由资源问题和 thinking 截断，转为 non-thinking 下几乎不交易、residual 为零。
这份历史总结不代表 9 月 13—14 日冷启动后的结果，后者见[文档导航](../README.md)。

## 筛选原设计与实现

[M2 计划 v2](../2026-09-05_M2_EXPERIMENT_PLAN_V2.md)设计为 A2、M1、M2-plain、M2
四臂、Qwen3-4B、seeds42/43，共八个主 cells；每 cell 32 个 Train windows、每窗四 siblings、
16 次更新，最后在 12 个 Train 内 dev windows 上 greedy 评估。1.7B wiring 和 4B pilot
只检查工程。期间没有正式 Valid / Final OOS 的模型效果评估。

A2 训练使用终局原始净收益，其余使用 uniform Y；最终 evaluator 共用 Y。
M2/plain 在 GAE 后、whitening 前向 actor advantage 加 residual，critic returns 不变；
全 logical batch 的真实 steps 统一 whitening。Train 标定使用 1408 条代理轨迹、28160 步，
floor=`0.0007613112285332427`、λ=`1.6912508025279984`；这些不是模型收益。

## 从失败到 R4 完整 pilot

| 运行 / 修订 | 实际结果 | 原因与处理 |
|---|---|---|
| 初始启动 | 两次未开始训练 | 修复历史 ignored manifest 的身份处理，以及 Hydra seed 类型进入 Ray 环境变量的问题 |
| v3，1.7B A2 | 8 条 Train 完整，actor/critic 各一次；最终评估 0/2 完整 | post-update greedy 触及 2048 response cap |
| v4 / R1，1.7B M2 | 8 条 Train 完整，538 steps，两角色各一次，完整 checkpoint 保存；评估 0/2 完整 | 共同 cap 提到 4096 后仍截断；不能说成另一条评估已成功 |
| v4-pilot，4B M2 | 首批 8 条完整，无实际更新 | `nvidia-smi` 超时异常被启动器传播，finally 终止仍存活的 trainer；不是已证明的模型 OOM |
| v5 / R2 | 首批 8 条完整，未更新 | 遥测超时已不再杀训练；reference 阶段 532.567 秒未返回并持续 host 限流，记录后主动停止 |
| v6 / R3 | reference 约 30.29 秒完成，critic 更新一次；actor 未更新 | pageable reference/Adam 交换缓解卡点，但以 soft limit 计算的容量准入拒绝后续交换，非 OOM |
| v7 / R4 | 4B M2 的 16 条 Train、两轮 actor/critic、两次保存、同 worker 最终 2 条 greedy 全部通过 | CPU 准入按真实 charged bytes 对 hard limit，继续留 2 GiB 余量；该两轮结果不保证长期容量 |
| v9 / R4 | 后续 wiring 和 M1 pilot 通过，进入首个主 cell；4B A2 完成 11 轮、88 条 Train | 第 12 轮训练响应触及 4096 cap，主 cell 停止，主筛选 0/8 完整；step-8 checkpoint 保留 |

v4 的实际 M2 residual 非零为 **24/160** 个推进步骤，pooling 改变 **15/160**；确已进入
更新，但没有完整最终评估，不能据此判断收益。v7 物理显存采样峰为 **95292 MiB**，
保存边界余量很小。NVML/cgroup 采样峰与框架内存记账、瞬时峰不能混称。

## 生成诊断与 R5

固定原 Train 输入，对 exact 4B base 和保留的 step-8 actor 比较；step-8 不是失败时
step-11 的精确权重，独立 vLLM 调度也不同，不能称为精确故障重放。

| 设置 | 实测结果 | 判断 |
|---|---|---|
| 默认 thinking / 仅关闭 thinking | 原 36 次固定请求未复现截断；仅关闭 thinking 的格式合格为 14/18 | 单独缩短生成不足以保证单工具接口 |
| non-thinking + single_tool_v1 | 首步 18/18 合格；后续 320/320 完整轨迹、6619 次请求，零非法/截断 | 全部无买卖；两个模型同窗 siblings 的 Y 一致，plain/pooled residual 均为零 |
| thinking + 相同单工具指令 | 17/18 合格，base seed46 在工具 JSON 中途触及 4096 cap | 条件式 320 条扩展没有启动，保留失败；不能宣布 thinking 已修复 |
| R5 真实训练验收 | 四项 1.7B wiring + 两项 4B pilot，64 条训练、12 条最终评估全部完整 | 共 1984 请求，cap 触发为零，actor/critic 各 8 次实际更新；仍无成功买卖/持仓 |

已确认非空训练 raw_prompt 会绕过环境默认 system 指令，模型输入缺少“每轮一个工具”的要求；
single_tool_v1 补齐了接口合同，没有指定投资方向。随后 R5 共同使用 non-thinking 与该协议。

R5 的 4B M1 为 360 次 hold，4B M2 为 360 次 hold 加 1 次 compute；M2 两轮非零 S 为
**0/320**。1.7B 共 131 次 invalid，当时成功请求缺完整响应/错误原文，不能事后准确分类。
净收益为零不等于 Y 为零：两窗现金 Y 分别为 −4.123584、−4.690993。R5 工程通过后停止，
没有自动把零交易当作机制有效或无效的结论，后续转入[行为诊断](03_M2_BEHAVIOR_DIAGNOSTICS.md)。

## 运行、存储与历史产物

服务器采用独立 systemd user service、Linger=yes、stdin=null、文件日志和进程组清理，
生命周期不依赖 SSH；没有做主动断网注入测试。遥测失败与真实训练失败分开记录。

9 月 8 日按用户授权清理普通缓存、指定旧 VS Code Server、conda 安装包和旧 Qwen2.5-3B，
释放约 **14.53 GiB**；uv 硬链接、Qwen3、失败 checkpoint 和 Python 环境保留。随后管理员
追加 300 GB，9 月 9 日实测总容量约 **621.79 GB / 579.09 GiB**。这是历史读数，不是当前余量。
Agent-R1 源码仅 MiB 量级，agentr1 环境约 6.76 GiB，当时保留；这些旧记录不是新的清理指令。

按当时预登记归档策略，成功 wiring/pilot 的重型模型和 optimizer 曾在完整验收后移除，
只留 marker、inventory 和证据；不能把它们描述为现在仍可恢复的完整 checkpoint。
原失败记录及其保留的 checkpoint 不被后来的成功覆盖。9 月 13—14 日产物保存策略另见近期报告。

## 入口与证据

历史实现入口是 `scripts/prepare_m2_screening.py`、`scripts/run_m2_screening.py`、
`src/agents/screening_launcher.py` 与 `screening_runtime.py`。这里不保留旧 `--stage all`
点火示例；重新执行应使用对应的新计划、配置和新目录，不能直接复用旧命令/身份。

| 证据根目录 | 用途 |
|---|---|
| `runs/m2-screening-20260906-v4/` | 1.7B 失败、真实机制与保留 checkpoint |
| `runs/m2-screening-20260906-v4-pilot/`、`runs/m2-screening-20260907-v5/` | 4B 遥测与 reference 资源故障 |
| `runs/m2-screening-20260907-v6/`、`v7/`、`v9/`（同一命名前缀） | R3 拒绝、R4 完整 pilot、主 cell 第 12 轮截断 |
| `runs/m2-screening-20260908-v10/diagnostics/` | 固定提示、320 条 non-thinking 与 thinking 负例 |
| `runs/m2-screening-20260908-v11/engineering_final_audit.json` | R5 六 cells、76 条轨迹与实际 residual 审计 |
| `runs/disk-audit-20260908/` | 原清理授权/执行/复核及扩容收据 |
