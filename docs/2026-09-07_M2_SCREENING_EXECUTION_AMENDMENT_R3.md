# M2 screening 工程执行修订 R3（2026-09-07）

本修订依据用户恢复授权及计划 §5.2，先验证 H100 / 64 GiB 主机限制下的共同资源布局。
主筛选仍为 0/8；1.7B 四臂接线、4B 两次更新及最终 Train greedy 的门禁仍须满足。
本文件登记工程更改，不构成 M2 有效性结论，也不自动开放 screening/dev。

## R2 实际诊断

运行 `runs/m2-screening-20260907-v5`，代码 `dc9c9abba01fdd24b0f63bbc9530aa00aaecd6d8`：

- 首轮 8/8 Train 轨迹完整；两轮目标为16条，目前完成8条。共188 steps、229,012 response
  tokens、433,885 prompt tokens，invalid action8次。
- old log-prob 于12:30:02–12:30:29 UTC完成（26.995秒）；reference log-prob随后进入，
  至12:39:21人为停止时532.567秒未返回。value、实际参数更新、checkpoint和评估均未执行。
- 主机采样峰58.343 GiB，memory.high从启动至operator stop增加1,683,019，OOM/OOM kill均为0。
  worker共享内存PSS在reference阶段明显增加。命名阶段证据支持reference内存压力定位，
  不能单凭这些数据证明每一笔分配的归属或排除其他性能因素。
- 一次15秒NVML timeout被R2捕获，此后仍有12次成功GPU采样。因此监控超时没有终止trainer。
- 已先落盘 `operator_stop.json`，再向核实的trainer PGID发送SIGTERM。SIGTERM日志晚于记录
  88.769毫秒，最终退出码−6在记录后4.533秒写出。未见SIGTERM前崩溃；不推断内部abort机制。
  本轮是主动停止的资源诊断，launcher FAILED / supervisor STOPPED，不是自然策略失败。

完整证据留在运行目录及 `audit.json`。R1/R2原失败和1.7B完整checkpoint继续保留。

## 新共同资源布局

配置身份为 `resources.profile=pageable_adam_v1`，适用于所有四臂及所有阶段。
不改变模型revision、权重精度、Adam算法/学习率、逻辑batch、样本、reward、GAE、S或whitening。
response cap4096、prompt4096、n4、greedy、严格完成性均沿用R1。不得用重试补齐失败样本。

1. **Reference参数放在普通CPU内存。** 当前固定verl强制reference FSDP CPUOffload，
   而固定torch在懒初始化时额外分配并pin CPU gradient，即使reference只做no_grad推理。
   仅对reference、optim_config=None、BF16、FSDP1调用调整构建路径，禁用该强制CPUOffload；
   原actor构建不变。reference仍使用同一base权重，没有optimizer/scheduler。
   参数在reference forward前同步搬到GPU，结束后同步搬回pageable CPU；更新FSDP参数视图，
   验证无_cpu_grad、无pinned CPU参数、local shard及共享weight aliases指向新storage。
2. **两个训练模型参数常驻GPU，只保留当前更新模型的Adam moments。** 同时关闭actor/critic
   的四个原生parameter/optimizer offload开关。inactive optimizer的exp_avg/exp_avg_sq保存在
   pageable CPU，FP32值、shape、state keys和step标量不变。第一次空Adam state合法。
   仅在完整worker update入口、两边grad均为None时交换；不增加optimizer/scheduler step。
3. **按字节交错交换。** 按slot记录位置和尺寸，不保存旧tensor列表。outgoing GPU→CPU和incoming
   CPU→GPU按累计字节差交错，每次同步复制、替换同一state slot、释放旧引用，再执行下一笔。
   不能先整份搬入或整份搬出，否则任一设备都可能出现两套Adam的瞬时峰。
   不要求actor/critic的moment数量或shape一一对应。拒绝共享/不连续/非FP32等不支持状态。
   每笔复制检查目标余量并保留2 GiB；它只是交换admission门槛，不是所有阶段已测安全余量。
4. **vLLM预留从0.28改为0.20。** 两套训练参数常驻后，需要为rollout权重同步及snapshot留空间。
   max_num_seqs8、engine max_model_len8192及4个CUDA graph shapes保持。实际KV容量由新启动日志
   核实；预留比例不等于保证8条满长请求可同时驻留。不会因此裁剪prompt/response或丢请求。

约4B payload估算：FP32参数每套14.985 GiB，Adam moments每套29.970 GiB，BF16 reference
7.492 GiB。旧布局的第二轮critic GPU payload约89.91 GiB（不含激活/运行时），主机也偏紧。
新布局普通无梯度阶段GPU约59.94 GiB、CPU约37.46 GiB；active梯度或单模型checkpoint snapshot
再增加约14.985 GiB。这些是静态估算，实际FSDP storage、allocator缓存、KV、进程及文件页缓存
必须实测。reference单项修复先做小模型验证，再用联合布局验证两轮pilot，避免仅首轮成功。

## 安装与失败边界

适配仅由显式screening Ray worker环境开关启用；legacy入口与父launcher不安装。
固定torch/verl版本和依赖源码SHA、FSDP1/NO_SHARD/world_size1、full fine-tuning、no LoRA、
FP32 scalar AdamW（foreach/fused/capturable/amsgrad均关闭）等条件不符时拒绝启用。

同一worker进程中的actor/critic共享协调器，以PID、execution identity和角色绑定。
reference不登记成训练actor。重复注册、live gradients、重入或转移失败使后续更新/保存失败；
不自动rollback或继续半交换状态。异常清理不能掩盖原始异常。日志仅写标量元数据。

继续使用原checkpoint内容及完成标记，保存前校验布局，实际逐round落盘仍是pilot验收条件。
本profile的checkpoint恢复顺序尚未验证，因此拒绝resume/load；不能宣称exact resume。
网络中断由独立systemd user service、Linger=yes、stdin=null和文件日志保障，不靠重启训练。

## 验证与执行

先完成针对状态数值、旧storage释放、别名、装饰器/角色隔离、原异常和失败锁定的CPU测试；
再用真实小型CUDA/FSDP模型验证reference重复迁移后输出/权重及CPUOffload差异，并验证交错Adam
多次更新与同一普通AdamW对照的参数/moments/step一致。小模型通过不代表4B容量通过。

冻结代码和配置后，从相同exact base在新目录启动4B M2 Train-only工程pilot：相同两Train windows、
seed42、n4、两个完整update rounds、每round保存、同worker最终两条Train greedy评估。
记录first/second update、reference、rollout、checkpoint、memory.high/OOM和交换字节/device证据。
没有主screening/dev后续队列；若再失败，保留结果并据具体阶段诊断，不自动扩cap。

最终验证结果与新运行身份在恢复进展记录及本机 `runs/M2_SCREENING_LATEST.md` 中追加。
