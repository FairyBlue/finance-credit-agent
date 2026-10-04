# 4B 显存交接与低显存训练验收：最终报告

当前进展：**2026-10-04 21:37 SGT（UTC+8）**。实验于 **2026-10-04 17:13:53 SGT** 完成。

**工程验收通过：72/72 条轨迹完整，M1/M2 各两轮训练、最终评估和 actor 导出全部完成。**
训练服务正常退出，显存占位服务已关闭并释放全部张量；当前没有本批训练作业。
M2 在这批 Train 评估的 greedy、sampled 平均含成本净收益均略高于 M1，幅度约 0.08–0.09
个百分点。此结果来自单 seed、两轮的小试，不构成正式性能优势结论。

## 最终收益与行为读数

每臂在相同四个日期隔离的 Train 评估窗口上完成 4 条 greedy、16 条 sampled。
本表收益已扣交易成本；只用本批结果作 M1/M2 比较。

| 解码方式 | M1 平均含成本净收益 | M2 平均含成本净收益 | M2−M1 | M1 非法请求 | M2 非法请求 |
|---|---:|---:|---:|---:|---:|
| greedy | 1.6874% | 1.7704% | +0.0830 pp | 0/80 | 0/80 |
| sampled | 1.5063% | 1.6010% | +0.0948 pp | 21/341 | 18/338 |

平均轨迹交易成本：greedy 的 M1/M2 分别为 **5972.18 / 6032.20** 账户美元，sampled
分别为 **5513.49 / 5649.33**。M2 费用略高，净收益仍略高。训练非法请求则为
M1 **19/339**、M2 **20/340**；不能只凭本批验收非法数较少就认定共同拒单罚已解决行为问题。
本批未重跑未 RL 的 SFT 基线，也未包含 A1 或正式 Valid/Test。

GitHub 阅读入口：[小型结果快照](results/2026-10-04-backup/README.md)，包含两臂摘要、
比较 CSV、完成验收和模型导出哈希；模型权重与完整运行目录仍在服务器。

## 训练和资源验收

| 项目 | M1 | M2（η=0.25） |
|---|---:|---:|
| 完整训练轨迹 | 16/16 | 16/16 |
| 完整最终评估 | 20/20 | 20/20 |
| critic / actor 实际更新 | 2 / 2 | 2 / 2 |
| 整臂耗时（包含前置准备） | 33.0 分钟 | 35.7 分钟 |
| worker PyTorch 峰值 allocated | 69.47 GiB | 69.44 GiB |
| worker PyTorch 峰值 reserved | 72.58 GiB | 72.59 GiB |
| placement 采样主存峰值 | 57.19 GiB | 57.58 GiB |

整批从 16:05:14 到 17:13:53，约 **68.6 分钟**。两臂共同 critic 初始 head hash 相同；
奖励守恒、共同拒单罚、M2 residual、训练预算和最终完整性均由运行控制器验收通过。

全部 8 次分块 Adam 更新成功；每次 993 块、最大 GPU 临时缓冲 64 MiB，实测更新本身
约 **6.7–18.2 秒**。有真实 moments 的阶段切换约 **131–190 秒**，第二轮恢复已通过。
主要额外时间来自大状态写盘/恢复；没有同批 stock Adam 速度对照，不能报告可靠的减速倍数。

两臂都没有新增 host OOM、OOM kill 或 memory.max 事件；memory.high 分别增加 49/68，
说明曾触及主存软限额回收或节流。两个最终 actor 的完整文件及 tensor 哈希复验通过，
各 399 tensors、约 8.22 GiB BF16 权重；导出不含优化器，不能作为完整训练恢复 checkpoint。

## 显存交接结论

准备、模型核验和 Ray 启动期间，独立 holder 持有 59 GiB。M1 首次 critic 加载前，
worker 在 16:07:46.512–16:07:47.311 暂停并释放占位，立即确认实际空闲 **75.7156 GiB**；
M2 的新 worker 也独立验收通过。reference/actor 初始化和训练期间均保持暂停、零持有。
整批完成后，holder 于 17:13:53 主动关闭，早于原租期到期；无自动回补或重复训练。

本次已经越过旧 reference 初始化 OOM，并验证完整两轮的优化器状态交换、恢复与导出。
**此次实际放行约 75.7 GiB 空闲；实测峰值约 69.5 GiB allocated、72.6 GiB reserved，
不能据此宣称严格 60 GiB 可运行。** 共享 GPU 的后续外部分配仍可能影响下一次运行。
没有修改 vLLM；断网保护依靠 `systemd --user` 独立服务、`Linger=yes`、`Restart=no`。
未做实际网络断连注入测试。

## 固定设置与证据

沿用[原两轮计划](2026-09-30_M2_LOW_MEMORY_PILOT_PLAN.md)：共同 SFT v3、4B 全参数、
共同 critic seed42、non-thinking；M1 η=0、M2 η=0.25；共同拒单罚 0.05；actor LR 2e-7、
critic LR 1e-6。训练窗口为 0001/0065/0009/0073，评估为 0113/0117/0121/0125，均属原 Train。
未拼接旧失败运行，未因本批收益结果改设置或追加轮数。

交接/控制器/placement 共 72 项及 2 个子测试、8/8 输入回放通过。
运行后的源码和输入封存核验也通过。

[完成读数](../runs/m2-low-memory-handoff-20261004/completion_readout.json) ·
[比较 CSV](../runs/m2-low-memory-handoff-20261004/comparison_summary.csv) ·
[权威状态与验收结果](../runs/m2-low-memory-handoff-20261004/state.json) ·
[最终运行页](../runs/m2-low-memory-handoff-20261004/FINAL_RESULTS.md) ·
[占位清理](../runs/m2-low-memory-handoff-20261004/reservation_cleanup.json)

两臂 actor 位于各自 `checkpoints/global_step_2/actor/huggingface/`，导出回执和 hash 位于
同级 `actor_export_receipt.json`。当前工程结果支持继续准备原定四轮共同拒单罚对照；
后续主要检查跨轮收益/非法率是否保持改善，保留共同 SFT 基线，而不是据两轮即宣布 M2 胜出。
