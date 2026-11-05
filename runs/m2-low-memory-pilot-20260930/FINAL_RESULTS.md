# 4B 低显存两轮工程验收

当前进展（新加坡时间）：2026-09-30T20:16:43.435569+08:00

状态：**FAILED**；当前臂：m1。

固定预算：M1、M2 各 16 条训练 + 20 条同 worker 评估，共 72 条。
4B 全参数；共同 SFT v3；non-thinking；CPU 活跃 Adam moments + 磁盘非活跃 moments。
模型导出不含优化器；本轮不支持断点续训或自动重跑。

GPU 准入读数：62.43 GiB 空闲（门槛 60 GiB）。

- m1：FAILED；已落盘轨迹 0/36。
  错误：RuntimeError: m1 trainer exited 1; evidence retained
- m2_eta025：PENDING；已落盘轨迹 0/36。

停止原因：RuntimeError: m1 trainer exited 1; evidence retained

本轮衡量资源路径能否跑通；两轮 Train 读数不能证明 M2 优于 M1。
阶段显存包含本进程指标；nvidia-smi 的整卡读数还包含其他用户进程。
