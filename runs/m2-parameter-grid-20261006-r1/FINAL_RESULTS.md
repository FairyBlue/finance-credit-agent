# 4B M1/M2 参数网格训练

当前进展（新加坡时间）：2026-10-06T18:23:32.331675+08:00

状态：**FAILED**；当前 seed：42。
当前臂：无运行臂。
已落盘轨迹 288/1944；已验收臂 2/14。

已验收臂中的未完成评估：5；金融覆盖状态：PENDING。

固定预算：seed42/43；每 seed 一个未 RL 的 SFT 和六个四轮 RL 臂。
Actor LR 1e-7/2e-7；每个 LR 对照 M1、M2 η=0.125/0.25；critic LR 1e-6。
每臂 32 条训练；第 2/4 轮各评估 12 个 Train 窗口，greedy1 + sampled4。
每 seed 共 972 条，全批 1944 条；只使用原 Train，未启动 seed44 或样本外验证。
同一服务串行执行；网络或客户端断开不影响服务；无自动重跑或断点续训。

R1 仅隔离真实 256-step 未完成评估，失败保留并单列统计；训练仍须完整。
未完成评估没有有效终局收益；落盘数包含失败尝试，不能当成完整轨迹数。
原失败批次另有 50 次尝试保留，R1 加原批次的计划总数为 1994。

整批已运行 1.99 小时。
按当前落盘速度粗估剩余 11.5 小时。
估时包含当前共享 GPU 等待；SFT 与 RL 阶段速度不同，早期估时波动较大。

- seed42 / sft0：COMPLETE；已落盘 60/60。
  未完成评估 1；金融覆盖 INCOMPLETE。
- seed42 / m1_lr1e7：COMPLETE；已落盘 152/152。
  未完成评估 4；金融覆盖 INCOMPLETE。
- seed42 / m2_lr1e7_eta0125：FAILED；已落盘 76/152。
  停止原因：RuntimeError: m2_lr1e7_eta0125 trainer exited 1; evidence retained
- seed42 / m2_lr1e7_eta025：PENDING；已落盘 0/152。
- seed42 / m1_lr2e7：PENDING；已落盘 0/152。
- seed42 / m2_lr2e7_eta0125：PENDING；已落盘 0/152。
- seed42 / m2_lr2e7_eta025：PENDING；已落盘 0/152。
- seed43 / sft0：PENDING；已落盘 0/60。
- seed43 / m2_lr2e7_eta025：PENDING；已落盘 0/152。
- seed43 / m2_lr2e7_eta0125：PENDING；已落盘 0/152。
- seed43 / m1_lr2e7：PENDING；已落盘 0/152。
- seed43 / m2_lr1e7_eta025：PENDING；已落盘 0/152。
- seed43 / m2_lr1e7_eta0125：PENDING；已落盘 0/152。
- seed43 / m1_lr1e7：PENDING；已落盘 0/152。

整批停止原因：RuntimeError: m2_lr1e7_eta0125 trainer exited 1; evidence retained

落盘条数是进度读数；完整性和收益以验收后的 grid_readout.json 为准。
结果用于检查训练稳定范围和下一步改进线索，不能替代正式性能验证。
