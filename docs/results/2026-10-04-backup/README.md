# 2026-10-04 实验结果备份

这是 GitHub 可直接阅读的固定快照；采集时间见 [manifest.json](manifest.json)。
后续运行不会自动更新本目录。文件均从服务器原路径逐字节复制，未重算收益或合并不同实验。

最新 4B 两轮小试已于 **10 月 4 日 17:13:53 SGT（UTC+8）** 完成：72/72 条完整，
M1/M2 各两次 actor 与 critic 更新，最终模型导出验收通过。
greedy 净收益分别为 1.6874% / 1.7704%，sampled 为 1.5063% / 1.6010%。
这是单 seed、Train 内工程结果，尚未证明正式样本外优势；四轮实验未启动。

| 内容 | GitHub 可读证据 |
|---|---|
| 最终报告 | [报告](../../2026-10-04_M2_GPU_HANDOFF_PILOT.md) |
| 两臂收益与成本 | [比较 CSV](runs/m2-low-memory-handoff-20261004/comparison_summary.csv) |
| 完成验收 | [完成读数](runs/m2-low-memory-handoff-20261004/completion_readout.json)、[权威状态](runs/m2-low-memory-handoff-20261004/state.json) |
| 分臂训练与评估 | [M1](runs/m2-low-memory-handoff-20261004/m1/summary.json)、[M2](runs/m2-low-memory-handoff-20261004/m2_eta025/summary.json) |
| 运行设置和来源 | [设置](runs/m2-low-memory-handoff-20261004/settings.json)、[源码身份](runs/m2-low-memory-handoff-20261004/source_identity.json) |
| 占位服务清理 | [清理收据](runs/m2-low-memory-handoff-20261004/reservation_cleanup.json)、[关闭状态](runs/gpu-reservation-20261004/status.json) |
| 原 9 月 30 日失败运行 | [最终读数](runs/m2-low-memory-pilot-20260930/FINAL_RESULTS.md)、[失败复查](runs/m2-low-memory-pilot-20260930/failure_diagnosis_20261002.json) |
| 分块优化器 / 占位 GPU 探针 | [优化器](runs/m2-low-memory-preparation-20260930/optimizer_cuda_probe_v2.json)、[占位服务](runs/gpu-reservation-preparation-20261002/cuda_probe.json) |

模型权重、optimizer、原始行情和完整请求/响应轨迹仍在服务器，未上传。
两份 `actor_export_receipt.json` 保存权重身份与哈希，**不包含权重本身**；
本快照不能恢复训练。源码中的默认模型/历史运行路径仍需相应服务器输入。

复制结果保留原相对链接和本机绝对路径；未纳入快照的目标可能无法在 GitHub 打开。
以本页入口查看已备份证据。manifest 为每份有效载荷保存原路径、大小和 SHA-256，
不对自身递归哈希。可在服务器仓库根目录运行
`python3 docs/results/2026-10-04-backup/_build_snapshot.py` 重建快照；重建会更新采集时间。

本次备份前在现有环境隐藏 CUDA，执行优化器、状态存储、占位、交接、运行控制器及
screening launcher/reference/placement 的 11 份定向测试：**394 passed、2 subtests passed**，
耗时 111.85 秒。修改与新增代码的 Ruff 检查通过，快照有效载荷大小和 SHA-256 核对通过。
这项回归没有启动新 GPU 训练，也不代表已验证全新克隆的 CPU 依赖安装。
