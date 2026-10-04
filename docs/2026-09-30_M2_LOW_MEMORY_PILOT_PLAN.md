# 4B 全参数低显存工程验收计划

用户已同意在约 60 GiB 可用显存的假设下继续实验。先验证分块 Adam 与磁盘暂存是否能支持现有 4B 全参数训练，再恢复更长的共同合法性惩罚实验。两轮验收用于检查资源和训练链路，不据此判定 M2 的收益优势。

## 执行方式

- actor、critic 的 FP32 参数留在 GPU，沿用原梯度累积、全局裁剪和学习率调度。
- 活跃优化器的 FP32 一阶、二阶 moments 留在 pageable CPU；每次最多使用 64 MiB 的 GPU 分块临时张量，调用当前 PyTorch 的 scalar Adam 实现，再回写 moments。
- 在 critic/actor 阶段边界，将非活跃 moments 分块写入本次运行专用磁盘目录。校验长度、SHA256 并完成 fsync 后，才释放 CPU moments；回收本进程空闲主存后，再恢复另一角色的 moments。
- reference 沿用 BF16 CPU，保留 vLLM sleep、microbatch 1、gradient checkpointing 和 remove padding。没有修改 vLLM，也没有改为 LoRA 或量化优化器。
- 新资源标识为 `chunked_adam_disk_v1`。历史运行保持原样，两个实验臂均从共同 SFT v3 新启动。

固定运行目录：`runs/m2-low-memory-pilot-20260930`。代码入口：`python -m src.agents.low_memory_pilot`。

## 固定实验范围

| 项目 | 设置 |
|---|---|
| 基座 | Qwen3 4B，共同 SFT v3 导出 |
| 初始化 | 相同 actor/reference/critic backbone；critic head 使用 `cpu_seed_v1`、seed 42 |
| 两个臂 | M1：η=0；M2：η=0.25 |
| 奖励 | 保留已批准的共同非法请求惩罚 0.05；金融 terminal Y 和 M2 residual 定义不变 |
| 学习率 | actor 2e-7；critic 1e-6 |
| 每臂预算 | 2 轮；每轮 2 windows × 4 siblings，即每臂 16 条训练轨迹 |
| 训练窗口 | 原日期覆盖计划的前四个窗口：0001、0065、0009、0073 |
| 评估 | 第 2 轮完成后，同 worker、原四个日期隔离 Train 验收窗口；各 1 greedy + 4 sampled，即每臂 20 条 |
| 总轨迹 | 32 条训练 + 40 条评估 = 72 条 |
| 生成 | 保持 non-thinking，原温度、token 上限和动作接口 |
| 导出 | 每臂最后一次 actor BF16 HF 导出；不含优化器，不支持断点续训 |

本次不重跑未 RL 的 SFT 基线；已保留该基线。该取舍适用于资源工程验收，正式性能比较仍需要共同基线和完整计划。

## 资源准入与断网保护

每臂启动前检查实际 GPU 空闲量至少 60 GiB；最多等待 12 小时。无需其他用户完全退出，也不停止任何其他用户进程。启动后的每个分块和 CPU moment 分配仍检查真实剩余资源，分别保留 2 GiB GPU 和 2 GiB 主存余量。这不是独占显存预留，其他容器的后续分配仍可能造成失败。

主存硬上限为 64 GiB；一次只保留一个优化器的 CPU moments。磁盘预留同时考虑两角色快照和替换期间的第三套 moments，约 90 GiB，加剩余模型导出和 8 GiB 余量。两个臂的临时状态与失败证据均保留，本次不删除历史模型。

运行由 H100 上独立的 `systemd --user` 服务托管，`Restart=no`，客户端断网不终止服务。程序每 30 秒刷新 `LIVE_STATUS.md`，当前进展一律写新加坡时间。服务或训练失败后停止，保留日志，不在同目录自动重跑或拼接轨迹。

## 验收与读数

1. 连续多步 CPU 和小规模 CUDA 对照：参数、moments、step 与原 AdamW 一致；检查外部梯度累积、裁剪、grad=None、多参数组和学习率调度。
2. actor/critic 交替两轮：验证只有活跃角色的 moments 在 CPU，磁盘恢复保留原状态；损坏、资源不足或阶段错误必须停止。
3. 每臂完成 16 条训练、2 次 critic 和 2 次 actor 更新、模型导出和 20 条同 worker 完整评估；两臂实际初始 critic head hash 相同。
4. 记录每次分块更新及磁盘交换耗时、整臂耗时、训练 worker 的 PyTorch 峰值分配/保留显存、容器主存采样、整卡显存读数。整卡读数包含外部负载，不能直接当作本实验显存。
5. 按 greedy、sampled 分别记录净收益、交易成本、非法请求；检查完整轨迹与奖励守恒。没有“必须买入”或按收益选样本的条件。

小规模数值一致不等于完整训练已通过。若真实 4B 在显存、主存、磁盘或恢复路径上失败，先保留实际失败点和峰值，再与用户讨论调整。完整验收通过后，才有依据估计更长实验的耗时和资源预算。
