# 4B 低显存训练实现与验收记录

当前进展：**2026-10-02 10:36 SGT（UTC+8）**。

**本轮已经失败停止，未完成任何训练轨迹或参数更新。** 9 月 30 日 20:13:24 SGT，队列在空闲显存 62.43 GiB 时放行；20:16:43 SGT，reference 模型初始化触发 CUDA OOM，服务退出。M2 尚未启动，没有新模型导出或收益结果。本次复查没有重启作业。

小规模数值验证和八个完整 Train 窗口 CPU 回放的通过结果仍保留。实际 4B 尚未执行到分块 Adam，不能据此判定其在稳定 60 GiB 条件下能否跑通或会慢多少。10 月 2 日 10:36 SGT 复查时 GPU 空闲约 **76.76 GiB**；这是即时读数，不代表显存预留。

[固定计划](2026-09-30_M2_LOW_MEMORY_PILOT_PLAN.md) ·
[服务器实时状态](../runs/m2-low-memory-pilot-20260930/LIVE_STATUS.md) ·
[最终运行读数](../runs/m2-low-memory-pilot-20260930/FINAL_RESULTS.md)

## 已完成的改动

4B actor/critic 继续全参数训练。活跃 Adam moments 在 CPU，GPU 只处理有界分块；非活跃 moments 写入本次运行专用磁盘目录。磁盘状态经长度、SHA256 和持久化校验后才释放主存；恢复时再次检查。没有改奖励、训练窗口、模型精度或动作逻辑。

保留原全局梯度裁剪和学习率调度。新增阶段校验，禁止 actor/critic 在错误阶段更新；部分更新失败后不能继续使用该优化器。本次只导出最终 actor 推理权重，不把临时 moments 宣称为可恢复 checkpoint。

接入审核发现 verl 实际使用 OmegaConf `ListConfig` 保存 betas，最初的类型校验会阻断首步。现接受标准序列，并用实际 `build_optimizer` 验证，保持原 beta 对象和值不变。这次修复发生在 4B 运行前。

## 验证结果

| 检查 | 结果 |
|---|---|
| 新分块 Adam + 阶段切换集成 | 53 项通过；包含真实 verl 配置、连续更新、全局裁剪、调度和异常停止 |
| 磁盘状态存取 | 22 项通过；覆盖损坏、截断、身份与步骤不匹配、资源拒绝和代际替换 |
| 新两轮控制器 | 17 项通过；包含真实 Hydra 配置、72 条固定预算、60 GiB 队列和历史评估 schema |
| 既有 launcher/reference/placement/exchange/export 回归 | 235 项及 137 个 subtests 通过 |
| H100 小规模数值对照 | 两组模式各 5 步，参数、moments、step 与 stock AdamW 逐位一致 |
| 输入和封存 | 8/8 Train 窗口完整 CPU 回放，最大 prompt 4,589 tokens，源码/数据/模型身份检查通过 |
| 静态检查 | 修改文件 Ruff 和 diff 空白检查通过 |

最终 CUDA 对照包括真实 `ListConfig` betas、两次 microbatch 累积、外部全局裁剪和 grad=None。
PyTorch 峰值分配仅 183,808 bytes，说明这是优化器数值探针，不能代表 4B 训练显存。
[最终代码 CUDA 收据](../runs/m2-low-memory-preparation-20260930/optimizer_cuda_probe_v2.json) 保留源码 SHA256；早期探针记录也保留。

## 本次实际失败点

| 时间或阶段（SGT） | 已核对的证据 |
|---|---|
| 9 月 30 日 20:04:11 | 独立服务启动，开始排队 |
| 20:13:24 | 空闲 62.43 GiB，达到 60 GiB 准入条件 |
| 20:14:38 | trainer 进程启动；与准入相隔约 74 秒，首条整卡占用采样已达 82.36 GiB |
| critic 初始化前 | 本训练 worker 的 PyTorch allocated 为 0，整卡已占用 72.47 GiB |
| critic 初始化后 | 本 worker allocated 为 15.02 GiB，整卡占用 91.91 GiB |
| reference 初始化 | FSDP 拼接参数需要再申请 194 MiB，GPU 仅剩 84.75 MiB，CUDA OOM |
| 20:16:43 | 主服务以退出码 1 停止，未自动重试；0/72 轨迹、0 次更新 |

直接证据是 [trainer.log](../runs/m2-low-memory-pilot-20260930/m1/trainer.log) 的 643、647、795–828 行，以及 GPU 队列和资源采样记录；[复查记录](../runs/m2-low-memory-pilot-20260930/failure_diagnosis_20261002.json) 保存对应文件 hash。

准入后其他 GPU 占用明显增加，原先假设的“持续可用 60 GiB”没有成立。当前检查只读取一次准入快照，之后仍有输入核对和运行环境准备，未在真正加载模型前重新检查。reference 虽然在运行阶段放 CPU，但现有初始化流程先在 GPU 构造 FSDP，再搬到 CPU；因此初始化也需要临时显存。这次尚未进入分块更新或磁盘 moments 交换。

容器主存采样峰值约 20.76 GiB，首尾 `memory.events` 相同，没有本次新增 host OOM 计数。磁盘也没有报错。失败分类是 **GPU 初始化 OOM**。

## 后续处理方向

两臂各跑两轮，合计 32 条训练和 40 条同 worker 评估。重点检查第二轮已有 moments 的磁盘恢复、CPU 硬上限、实际 GPU 峰值、耗时、最终导出和完整轨迹。

下一步应先补充真正启动模型前的即时显存复查和一段连续空闲观察，并覆盖 reference 初始化的临时显存需求，再用新目录重跑同一两轮验收。连续观察仍不能保证其他容器随后不分配显存；若外部负载继续跳变，就需要重新讨论更低显存布局或可保障的运行时段。当前未实施这些变更，也未启动新尝试。

服务 `m2-low-memory-pilot-20260930.service` 当前为 **failed / exit-code 1**，配置为 `Restart=no`、`KillMode=control-group`，用户服务 linger 已启用。服务自行完成了排队、启动和错误退出；没有做实际断网注入测试。[启动收据](../runs/m2-low-memory-pilot-20260930/launch_receipt.json) 保存服务命令和输入 hash。

所有时间按新加坡时区记录。上方 `FINAL_RESULTS.md` 已由服务自动写入失败状态；原始源码、输入、日志和退出记录保留，没有拼接样本或覆盖失败运行。
