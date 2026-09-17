# M2：共同拒单罚四轮小试执行记录

当前进展：**2026-09-17 15:23 SGT（UTC+8）**。

**实现和 CPU 验收完成，独立服务已启动，当前等待共享 H100 空闲；GPU 训练尚未开始。**
权威实时进度见 [LIVE_STATUS](../runs/m2-common-validity-penalty-20260917/LIVE_STATUS.md)，
每 45 秒刷新；执行结束后同目录自动生成 `FINAL_RESULTS.md` 和 `completion_validation.json`。
本页记录启动时状态，结果以这些动态文件为准。

## 本次实际完成

- 按用户确认，只删除冷启动前旧实验的 6 个 optimizer，释放约 **171.24 GiB**，
  可用空间恢复到约 **186.44 GiB**。旧模型权重、日志和结果保留；
  [清理收据](../runs/disk-audit-20260917/optimizer_cleanup_receipt.json)记录逐文件范围与复核。
- 新增独立实验身份 `common_sft_validity_penalty_v1`，目录
  `runs/m2-common-validity-penalty-20260917/`，按[既定计划](2026-09-17_M2_COMMON_VALIDITY_PENALTY_PLAN.md)执行。
- 训练使用 `r′ = Y/N − 0.05 × invalid`，每个环境拒绝请求只扣一次，
  两臂的 GAE 和 critic returns 都使用罚后奖励；M2 sibling residual 使用原始金融 Y。
  默认关闭新选项时保留原路径，评估仍用原金融 Y 和含成本净收益。
- 逐步保留原金融 reward、罚项、训练 reward、U/S、最终 actor advantage 和 critic return；
  真实 actor loss 诊断将三项 reward 与实际 loss A 一起记录，用于确认 GPU 上的训练接线。

## 启动前验证

| 检查 | 结果 |
|---|---|
| runtime、actor probe 与历史兼容性 | 101 项测试通过 |
| 新运行器、固定预算、失败停止、证据与数值容差 | 28 项测试通过 |
| GPU 排队、不重复点火、遥测失败不误放行 | 4 项测试通过 |
| 读数、完整分母、奖励对齐、历史对照、监视器 | 25 项测试通过 |
| CPU 完整回放 | 8 个训练窗口 + 4 个验收窗口全部通过；最大 prompt 4589 tokens，小于 8192 |
| 与原实验数据一致性 | 新 8 条输入与原 coverage 前四轮逐字段相同，仅实验角色标签不同；验收 parquet 字节相同 |
| 独立消费者烟测 | 真实 Agent-R1 batch 经新 runtime 输出，运行器与读数均能校验 M1/M2/SFT0 |
| 源码与输入封存 | 80 个源文件、7 项合同输入；seal 检查通过 |
| 新目录读数 | 0/164，尚未执行；无证据错误，能读取旧 SFT0 与同臂第 2/4 轮历史参照 |

合计 **158 项测试通过**，对应代码 lint 通过。
首次交互式 CPU 预检因漏设 `third_party/agent-r1` 的 PYTHONPATH 在 import 阶段退出，
补齐与既有服务相同的路径后，12 窗回放通过；此前没有 GPU 训练或采样。
记录见 [准备与验证](../runs/m2-common-validity-penalty-20260917/prelaunch_validation.json)、
[输入一致性](../runs/m2-common-validity-penalty-20260917/input_equivalence.json)。

## 执行安排与排队

| 阶段 | 固定预算 |
|---|---|
| 未 RL 的共同 SFT v3 | 4 窗 ×（1 greedy + 4 sampled）= 20 条验收 |
| M1 | 4 轮 × 2 窗 × 4 siblings = 32 条训练；第 2/4 轮各 20 条验收 |
| M2，η=0.25 | 同 M1，独立从相同 SFT v3 和 critic seed42 开始 |
| 总计 | **64 条训练 + 100 条验收 = 164 条**；两臂最终导出第 4 轮 actor |

服务于 **2026-09-17 15:22:41 SGT** 启动：

- 主服务：`m2-common-validity-penalty-20260917.service`。
- 监视服务：`m2-common-validity-penalty-20260917-monitor.service`。
- 两服务均为 `systemd --user` 托管，`Linger=yes`、stdin=null、文件日志、进程组清理，生命周期独立于客户端。
- 当前共享显存占用约 **17097 MiB**；旧同配置 M1/M2 实测物理显存峰值约 **88302 MiB**，
  二者合计超过这张卡的 95830 MiB，因而保留原准入条件，等待占用低于 1024 MiB。
- 队列每 45 秒检查，最多等 12 小时；主服务最长 16 小时含排队，监视器最长 17 小时。
  未获得 GPU 时不会开始训练；未停止其他任务，不自动重试失败实验。
- 按上一轮耗时估算，拿到 GPU 后约需 **1.5–2 小时**；排队时间另计。

[启动收据](../runs/m2-common-validity-penalty-20260917/launch_receipt.json)记录实际命令、环境与封存身份。
训练开始后会验证真实 GPU 更新与奖励证据；目前没有新收益结果。

## 16:35 SGT：能否绕过排队直接启动

用户询问共享 GPU 持续占用时能否直接训练。本次只读核对后，**不建议原配置强行启动**。

- 当前 GPU 可用 **78217 MiB（76.38 GiB）**，占用 **17103 MiB**，利用率虽为 0%，占用的显存仍未释放。
- 按原运行的相位日志对齐物理显存采样，仅前四轮的峰值就已达 **88240 MiB（86.17 GiB）**，
  高于当前可用量约 **9.79 GiB**；第一轮也已达 84740 MiB，高于当前可用量。
- 主要峰值位于 `update_critic` / `update_actor`，不是只发生在生成时。
  当前执行方式已有非活动 Adam 状态与 reference 的 CPU 交换；vLLM 在进入训练时会释放其缓存。
  因此单独降低生成缓存比例，不能认为就能消除训练阶段的缺口。
- 以上是旧运行的实测采样，而非未来精确最低需求；但目前没有支持“只关掉排队门禁就可可靠完成”的证据。

如果要在这份剩余显存上运行，需要另做训练状态分阶段搬运等资源适配，并同时核对 64 GiB 主机内存限制。
M1/M2 应共同采用同一新执行方式，先做短工程验收，再进入原四轮对照；金融目标与 M2 公式可以保持现有设定。
本次未修改已封存代码、未启动强制训练、未操作其他任务，原独立队列继续等待。
逐阶段测量见 [共享 GPU 可行性核对](../runs/m2-common-validity-penalty-20260917/shared_gpu_assessment.json)。
