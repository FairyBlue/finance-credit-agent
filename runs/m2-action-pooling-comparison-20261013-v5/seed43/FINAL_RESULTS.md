# 4B SFT / M1 / M2 参数网格

当前进展（新加坡时间）：2026-10-13T17:33:08.880924+08:00

状态：**FAILED**；当前臂：m2_plain_lr1e7_eta025。
本批 seed：43。

固定预算：共 668 条；每训练臂 32 条训练 + 120 条评估；SFT0 60 条评估。
4B 全参数；共同 SFT v3；non-thinking；CPU 活跃 Adam moments + 磁盘非活跃 moments。
模型导出不含优化器；本轮不支持断点续训或自动重跑。

GPU 准入读数：3.53 GiB 空闲（门槛 60 GiB）。

- sft0：COMPLETE；已落盘轨迹 60/60。
  训练 worker 的 PyTorch 峰值分配：64.50 GiB；整臂耗时 13.2 分钟。
  greedy：平均净收益 0.6940%；非法请求 41/281。
  sampled：平均净收益 0.9580%；非法请求 153/1113。
- m2_lr1e7_eta025：COMPLETE；已落盘轨迹 152/152。
  训练 worker 的 PyTorch 峰值分配：69.89 GiB；整臂耗时 84.5 分钟。
  greedy：平均净收益 0.7168%；非法请求 50/290。
  sampled：平均净收益 0.7384%；非法请求 154/1114。
- m2_plain_lr1e7_eta025：PENDING；已落盘轨迹 0/152。
- m1_lr2e7：PENDING；已落盘轨迹 0/152。
- m1_lr1e7：PENDING；已落盘轨迹 0/152。

停止原因：TimeoutError: native GPU budget did not become available within 1800 seconds

本轮比较共同 SFT 起点、M1 和 M2；单 seed Train 读数不替代正式验证。
阶段显存包含本进程指标；nvidia-smi 的整卡读数还包含其他用户进程。
