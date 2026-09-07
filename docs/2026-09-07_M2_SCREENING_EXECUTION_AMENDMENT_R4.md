# M2 screening 工程执行修订 R4（2026-09-07）

本次仅修正R3的CPU容量准入检查。模型、数据、scorer、算法、generation、资源放置和2 GiB
transfer reserve均沿用R3；共同profile仍为 `pageable_adam_v1`，以新revision/commit冻结执行身份。
不修改主机cgroup配置，不启用resume，不自动进入主screening/dev。

## R3的实际进展与失败

`runs/m2-screening-20260907-v6`，代码 `28a89801fba67338b8cd4467f40f3041bfc484a5`：

- 首批8/8 Train轨迹完整，共190 steps/160 advancing steps，438,194 prompt和222,151 response
  tokens。实际M2诊断与独立复算一致：gate19/40，nonzero S76/160，pooling changed58/160。
- old log-prob26.21秒、reference30.29秒、value23.23秒、critic更新95.87秒均成功。
  reference参数8,044,936,192 bytes完整搬回pageable CPU，无额外CPU gradient。
  R2的reference卡点已在本次执行通过，但尚不能宣称完整资源pilot通过。
- actor更新前，critic Adam已有30,564,872,200 bytes转入CPU，剩1,614,893,056 bytes在GPU。
  准入检查拒绝下一块403,723,264 bytes，并使协调器永久失败，后续未继续更新或保存。
  最终launcher FAILED / supervisor STOPPED；只有一次critic更新，没有actor更新/完整round、
  checkpoint或评估。没有发生OOM、memory.high增量或GPU监控错误。
- 失败边界memory.current=59,494,535,168 bytes（55.4086 GiB）；旧检查要求在memory.high
  57.6 GiB之下再保留2 GiB，因此下一块差约0.1846 GiB。memory.max实际为64 GiB。
  这是准入检查主动拒绝，不是内存分配器已报告OOM。原失败与半交换状态日志保留，不复用。

证据：v6下 `first_round_audit.json`、`resource_failure_audit.json`、placement和phase日志。
Torch allocator统计包含vLLM睡眠分配池的记账，不能直接等同物理显存；同时保留driver-free与
NVML采样。10秒采样也可能遗漏瞬时峰，不能用其较低峰值证明后续容量。

## 检查修正及边界

CPU每笔同步搬运前使用真实已计费的 `memory.current`，要求：

`memory.max - memory.current >= next_storage_bytes + 2 GiB`。

必须存在有限、有效的hard limit；没有足够余量时仍拒绝。GPU的driver-free准入规则不变。
不预先抵扣page cache，不假设任何缓存一定能回收，不改变copy顺序、Tensor值或state keys。
memory.high是回收/限流阈值，继续作为性能压力指标记录，不再作为第二个hard allocation ceiling。
本机memory.high、memory.max、swap以及其他进程配置均不变。
语义依据：[Linux cgroup v2 memory接口](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files)。

基于失败时快照，假设其他占用不变，完成剩余交换约56.9126 GiB，距64 GiB硬限7.0874 GiB；
保留2 GiB后尚余5.0874 GiB。按本模型最大storage约1.4490 GiB估算，单笔额外checkpoint staging
会到约58.3616 GiB。该估算不约束文件页缓存、dirty writeback、进程增长或异步分配；不能替代
真实两轮更新、rollout、reference及checkpoint验证。若soft pressure持续且阶段不推进，保留
诊断并处理；不会据此关闭监控、绕过完成性检查或自动补齐样本。

## 验证与下一次运行

CPU回归覆盖v6实际拒绝值、hard-limit reserve精确边界、soft threshold独立性、不计缓存信用、
拒绝unlimited/无效capacity；沿用交换值/生命周期/失败锁和worker边界测试。
共38项CPU测试、137个subtests通过；全项目Ruff及64个src/scripts文件mypy通过。
R3真实CUDA/FSDP数值对照保持适用，其copy/update/reference实现未改变；本次改变的是CPU准入门槛。

冻结新身份后，从相同base开始新的4B M2 Train-only两轮pilot，仍使用两个既定Train windows、
seed42、n4、逐round保存及最终两条Train greedy。原失败不改写、不续接半交换optimizer、不混入
1024条主筛选预算。主screening仍0/8，1.7B接线与其他必要门禁未解除。
