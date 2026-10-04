# H100 显存占位试验

当前进展：**2026-10-02 11:05 SGT（UTC+8）**。

用户同意先持有 30 GiB，再逐步申请后续空闲显存。本次已启动独立服务，
并实际达到 30 GiB；随后将目标提高到 59 GiB。**11:05 已持有 59 GiB，状态 HOLDING，
OOM 次数为 0**；当时 driver 空闲约 2.74 GiB。最新实际持有量和状态见
[实时状态](../runs/gpu-reservation-20261002/LIVE_STATUS.md)，每 5 秒刷新。
**4B 训练尚未重启；这是显存占位试验，尚未接入训练释放流程。**

## 执行方式

- 服务：`fca-gpu-reservation-20261002.service`，PID `2905858`。
- 启动：2026-10-02 11:00:20 SGT；到期：**2026-10-02 17:00:20 SGT**。
- 单进程持有空 `uint8` CUDA 张量，不加载模型，不执行推理或训练。
- 每秒最多申请一块 1 GiB；目标与 allocator 显存占用上限均为 **59 GiB**。
  CUDA context 不计入 allocator 上限，另留约 1 GiB 预算；总进程占用应看实际读数。
- 每次申请前检查：新增分配后应仍有至少 2 GiB driver 空闲。
  这是申请时的检查，其他进程后续分配仍可能消耗这些余量。
- 不足时等待；OOM 后退避 30 秒并保留已有块；其他错误退出并释放。
- `release` 先暂停增长再释放，避免将交给训练的空间立即申请回来。
- `systemd --user` 托管，已确认 `Linger=yes`；终端或网络断开不影响服务。
  内部租期 6 小时，systemd 最长运行 6 小时 2 分钟，不自动重启、不自动延长。

## 控制命令

在仓库根目录执行；`status` 查询，`pause` 只暂停增长，`stop` 全部释放并退出：

```bash
.venv/bin/python -m src.agents.gpu_reservation_service status --directory runs/gpu-reservation-20261002
.venv/bin/python -m src.agents.gpu_reservation_service pause --directory runs/gpu-reservation-20261002
.venv/bin/python -m src.agents.gpu_reservation_service release --directory runs/gpu-reservation-20261002 --gib 5
.venv/bin/python -m src.agents.gpu_reservation_service target --directory runs/gpu-reservation-20261002 --gib 59 --resume
.venv/bin/python -m src.agents.gpu_reservation_service stop --directory runs/gpu-reservation-20261002
```

也可直接停止本服务：

```bash
systemctl --user stop fca-gpu-reservation-20261002.service
```

释放按完整块进行。控制命令可指定 `--request-id`：如果应答因断连丢失，
使用相同 ID 重试不会重复执行释放。命令日志保存在运行目录。

## 已完成验证

- 核心 allocator 控制 25 项测试通过；独立服务及 Unix socket 控制 25 项测试通过；Ruff 通过。
- H100 小额试验最高持有 0.5 GiB：申请、释放、暂停后不回补、重复释放请求去重、
  恢复增长及主动退出通过；另一独立进程的租期到期退出通过。
- 小额释放后，allocator 已分配/保留量均归零；进程正常退出。
- 首次小额启动遇到 Unix socket 路径过长，已改为短 runtime 路径并保留失败记录。
- 正式服务已达到初始 30 GiB；提高目标后已达到 59 GiB，状态 HOLDING，无分配错误。

证据：[小额 GPU 验证](../runs/gpu-reservation-preparation-20261002/cuda_probe.json)、
[启动收据](../runs/gpu-reservation-20261002/launch_receipt.json)、
[初始持有快照](../runs/gpu-reservation-20261002/initial_holding.json)、
[提高目标收据](../runs/gpu-reservation-20261002/growth_target_receipt.json)、
[最新结构化状态](../runs/gpu-reservation-20261002/status.json)。

## 与训练衔接

独立进程释放的空间会向所有进程重新开放，不能保证交给本项目，也不预留计算吞吐。
训练接入前需先完成准备和源码核验，再在真实 worker 即将加载模型时暂停并释放占位，
立即复查空闲量；训练期间保持占位暂停，确认训练与 Ray GPU worker 退出后才恢复增长。
现有训练控制器要求 driver 空闲至少 60 GiB，不能在占位服务仍持有显存时直接启动，
否则会被自身占位阻挡。尤其不能在 vLLM 显存分析期间释放占位，以免影响其容量测量。

上一项低显存训练仍保留原失败结论：0 条轨迹、0 次更新，reference 初始化时 OOM；
本次占位成功不代表低显存训练验收通过。
