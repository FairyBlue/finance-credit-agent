# 4B SFT / M1 / M2 参数网格

当前进展（新加坡时间）：2026-10-06T18:23:29.892784+08:00

状态：**FAILED**；当前臂：m2_lr1e7_eta0125。
本批 seed：42。

固定预算：共 972 条；每训练臂 32 条训练 + 120 条评估；SFT0 60 条评估。
4B 全参数；共同 SFT v3；non-thinking；CPU 活跃 Adam moments + 磁盘非活跃 moments。
模型导出不含优化器；本轮不支持断点续训或自动重跑。

GPU 准入读数：74.36 GiB 空闲（门槛 60 GiB）。

- sft0：COMPLETE；已落盘轨迹 60/60。
  训练 worker 的 PyTorch 峰值分配：64.50 GiB；整臂耗时 12.1 分钟。
  greedy：平均净收益 0.2394%；非法请求 8/228。
  sampled：平均净收益 0.4828%；非法请求 60/1020。
- m1_lr1e7：COMPLETE；已落盘轨迹 152/152。
  训练 worker 的 PyTorch 峰值分配：69.46 GiB；整臂耗时 68.5 分钟。
  greedy：平均净收益 0.5805%；非法请求 6/206。
  sampled：平均净收益 0.4693%；非法请求 47/1007。
- m2_lr1e7_eta0125：FAILED；已落盘轨迹 76/152。
  错误：RuntimeError: m2_lr1e7_eta0125 trainer exited 1; evidence retained
- m2_lr1e7_eta025：PENDING；已落盘轨迹 0/152。
- m1_lr2e7：PENDING；已落盘轨迹 0/152。
- m2_lr2e7_eta0125：PENDING；已落盘轨迹 0/152。
- m2_lr2e7_eta025：PENDING；已落盘轨迹 0/152。

停止原因：RuntimeError: m2_lr1e7_eta0125 trainer exited 1; evidence retained

本轮比较共同 SFT 起点、M1 和 M2；单 seed Train 读数不替代正式验证。
阶段显存包含本进程指标；nvidia-smi 的整卡读数还包含其他用户进程。
