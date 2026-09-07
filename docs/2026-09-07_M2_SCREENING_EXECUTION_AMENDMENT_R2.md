# M2 screening 工程执行修订 R2（2026-09-07）

**当前状态：用户已要求停止实验并整理报告。R2 代码及相关测试已完成，但没有启动 GPU 重跑。**
实际结果见 [当前实验报告](2026-09-07_M2_SCREENING_EXPERIMENT_REPORT.md)。

R1 的科学契约继续生效：四臂、scorer、数据、seeds、模型 revisions、dtype、学习率、n=4、
逻辑 batch、4096 response cap、greedy evaluator 及严格完成性均不变。本修订属于计划 §5.2
允许的共同工程修复，不是新的效果结论或人类签字。

## 实际触发证据

`b90bde9ffee5953c17b6b55b6aba82b152f6e461` 的 1.7B M2 接线完成 8/8 Train 轨迹、两类参数
更新和完整 checkpoint；post-update greedy 仍在 4096 cap 内陷入重复 reasoning，无工具调用，
整轮维持 INCOMPLETE。停止自动提高 cap，未进入任何主 screening/dev。

为区分小模型行为与实际 4B 可行性，提前运行原预算内的 4B M2 Train-only pilot，意图记录于
`runs/m2-screening-20260906-v4-pilot/diagnostic_intent.json`。这一步没有解除接线门禁。
4B 首轮 8/8 轨迹完整结束，共 190 agent steps；尚未进入实际参数更新。2026-09-06 15:13 UTC
启动器因 `nvidia-smi` 查询超过 15 秒抛出 TimeoutExpired，其 finally 分支终止了 trainer。
因此这是监控故障导致的工程失败，不能写成策略截断或资源 pilot 通过。

采样主机峰 58.20 GiB，memory.high 事件增加 132,878，未有 OOM；GPU 采样峰 65.73 GiB。
发生限流时 GPU 从约 56 GiB 降至约 25 GiB，支持 offload/参考 forward 内存压力的推断，
但原日志不足以确定命名阶段。4B 两次更新及保存的容量仍未验证。

## 修复与新增证据

1. GPU 遥测为 best-effort：timeout、NVML 非零退出、OS 错误写入 `gpu_error`，继续等待 trainer。
   不把采样缺失写成 GPU=0。训练真实退出码、异常及清理规则保持有效。
2. 为 old-logprob、reference、value、critic/actor 更新记录 enter/success/error 阶段与 cgroup
   状态；日志失败不改变模型计算的返回值或异常。此信息用于定位主机压力，不能替代完整验收。
3. 共同显式设置 vLLM `compilation_config.cudagraph_capture_sizes=[1,2,4,8]`。
   R1 的 `max_num_seqs=8` 未减少 V1 默认 67 个 graph shapes；本修订修复这一未兑现的资源优化，
   实际启动日志仍需复核。不得将 graph 数量减少本身当成 4B 容量通过。

原失败及完整轨迹证据保留。恢复后，R2 计划使用新的运行目录，从相同 base model 开始一次
4B M2 工程重试；目前尚未执行。
这不是已证明的 exact resume，也不复用旧 siblings 或补齐旧结果。该重试属于额外工程成本，
不能混入原主筛选的 1024 条训练预算。主筛选仍需全部必要门禁通过后另行启动。

## 保留的限制

1.7B 四臂接线未全部通过；4B 的第二次更新内存峰与 checkpoint 尚未验证。不得以本修订跳过。
参考模型的 FSDP CPUOffload 会预分配 CPU gradient，可能产生不必要的 pinned 内存；当前尚未
改变这一布局，先用命名阶段证据定位。运行中不修改冻结源码，也不因失败自动放宽生成边界。

服务仍由 H100 上 systemd user manager 托管，Linger=yes，stdin=null，日志落盘。
网络中断不依赖重启 trainer 处理；模型/资源/监控错误与网络中断分别记录。
