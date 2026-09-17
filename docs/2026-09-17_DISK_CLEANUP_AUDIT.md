# H100 历史模型与磁盘清理候选

检查时间：**2026-09-17 14:36 SGT（UTC+8）**。

**执行更新：2026-09-17 14:55 SGT，用户确认后已清理首选方案的 6 个旧优化器文件。**
文件系统空间回收后，可用容量约 **186.44 GiB**，较清理前增加约 **171.24 GiB**。
三个 checkpoint 的六份 actor/critic 模型文件大小、inode、修改时间未变，原 JSON 元数据 hash 未变。
每个目录新增 `optimizer_pruned.json` 和保留说明，明确不再支持按原优化器状态续训；
原完成收据作为历史记录保留。另两组候选及当前模型均保留。
详见[实际清理收据](../runs/disk-audit-20260917/optimizer_cleanup_receipt.json)。以下为执行前盘点。

当前文件系统约 **578.12 GiB 总容量、562.92 GiB 已用、15.20 GiB 可用，使用率 98%**。
换成十进制约为 620.75 GB 总量、16.33 GB 可用。管理员扩容已经生效，但历史训练产物累计占用了大部分空间。
`runs/` 约 524 GiB，是主要来源；Hugging Face 缓存约 12 GiB，清理环境和小日志不是当前重点。

## 建议先清理：旧实验的优化器状态，约 171.24 GiB

`runs/m2-learning-restart-20260913/` 是**共同 SFT 冷启动之前**的四轮 M1/M2 实验。
最终两臂均没有交易，且当时实际 critic 初始化尚未使用后来修复的共同 seed；
当前实验已经采用 SFT v3 和修复后的共同初始化，不从这批 checkpoint 续训。
依据见[该轮报告](2026-09-13_M2_LEARNING_RESTART_REPORT.md)。

这批保存了三个完整 checkpoint，包含 actor、critic 和两者的 Adam 优化器状态。
建议先只删除下列每个目录中的两份 `optim_world_size_1_rank_0.pt`：

| checkpoint（相对于 `runs/m2-learning-restart-20260913/`） | actor 优化器 | critic 优化器 | 合计 |
|---|---:|---:|---:|
| `m1/checkpoints/global_step_4/` | 29.98 GiB | 27.10 GiB | 57.08 GiB |
| `m2/checkpoints/global_step_2/` | 29.98 GiB | 27.10 GiB | 57.08 GiB |
| `m2/checkpoints/global_step_4/` | 29.98 GiB | 27.10 GiB | 57.08 GiB |
| **6 个文件合计** | | | **171.24 GiB** |

完整文件路径及大小见[候选清单](../runs/disk-audit-20260917/candidate_inventory.json)的 `recommended_scope`。
actor 文件在各 checkpoint 的 `actor/` 子目录，critic 文件在 `critic/` 子目录。

这样可保留旧 actor/critic 权重、配置、日志、轨迹、指标和源码副本，仍有权重供后续重新导出和评估；
**会失去按原 Adam 状态恢复该旧实验的能力**，不能再将其称为可原样续训的完整 checkpoint。
预计可用空间可由约 15.20 GiB 增至 **186.44 GiB**，已足够下一项四轮小试。
删除若获确认，需另写清理收据并标明 checkpoint 已裁剪，保留原始完成收据作为历史记录。

## 如果还要清理旧模型：其他候选

以下是候选 checkpoint 内大型模型与优化器文件的实际分配空间，不是要删除整个实验目录。

| 候选 | 模型与优化器合计 | 其中优化器 | 判断与影响 |
|---|---:|---:|---|
| 9 月 13 日冷启动前 M1/M2 的上述三个 checkpoint | **261.62 GiB** | 171.24 GiB | 已被共同 SFT 路线替代。整批重型文件删除后，将失去这些具体权重的重评能力；只清理优化器已经足够 |
| 9 月 7 日旧 4B A2，`m2-screening-20260907-v9/screening-a2-s42/checkpoints/global_step_8/` | **87.41 GiB** | 57.08 GiB | thinking 阶段第 12 轮截断，保留的是 step 8；与当前共同 SFT 实验起点不同。另有约 8.22 GiB 的 BF16 actor 导出可保留，不能等同于原精度的完整训练状态 |
| 9 月 6 日旧 1.7B M2，`m2-screening-20260906-v4/wiring-m2-s42/checkpoints/global_step_1/` | **36.30 GiB** | 23.33 GiB | 旧 R1 工程失败样本，评估截断，后续已有 R5 工程验收。另有约 3.79 GiB 的 BF16 actor 导出可保留 |

三组重型文件合计 **385.32 GiB**；这是更大范围的可选清理，**已经包含**首选方案的 171.24 GiB，不能相加。
历史背景见[screening 工程总结](history/02_M2_SCREENING_ENGINEERING.md)。旧诊断导出分别位于
`runs/m2-screening-20260908-v10/diagnostics/actor_hf/` 和
`runs/m2-screening-20260907-v8/diagnostics/actor_hf/`，其来源已有 `export_provenance.json` 记录。
本次没有重新加载模型或复跑这些历史验收。

## 当前建议保留

| 产物 | 所在运行目录占用 | 保留原因 |
|---|---:|---|
| 共同 SFT v3 | 7.91 GiB | 下一轮 actor/reference/critic backbone 的共同起点 |
| 最新 Train coverage 的 M1/M2 | 16.52 GiB | 当前结果及后续对照依据 |
| 9 月 14 日三臂 tuning 模型 | 24.73 GiB | 近期信号强度、解码方式和收益回落对照 |
| 共同 SFT 后最初两轮 M1/M2 | 60.24 GiB | 有价值的早期训练对照，暂不优先清理 |
| SFT v1、v2 的模型与 adapter 等 | 合计 15.80 GiB | 冷启动来源及回归比较；收益不如清理旧 optimizer 大 |

同时保留基础 Qwen 模型缓存、Python/agentr1 环境、数据、实验日志与全部评估证据。
当前 `train_coverage` 默认加载 SFT v3；[下一轮计划](2026-09-17_M2_COMMON_VALIDITY_PENALTY_PLAN.md)
也使用 fresh v3，并不依赖上述旧实验的 optimizer。检查时未发现本项目的训练或推理进程；
候选大型文件均为普通文件、硬链接计数为 1，`runs/` 内未发现指向候选目录的符号链接。

空间按 `stat.st_blocks × 512` / `du` 估计，GiB 为二进制单位，与旧报告的文件逻辑字节数可能不同。
实际释放量以执行后 `df` 为准。执行前仅做盘点；后续用户确认的实际清理范围与结果见本文顶部。
