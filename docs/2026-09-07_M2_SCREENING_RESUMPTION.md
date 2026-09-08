# M2 screening 恢复执行记录（2026-09-07）

> **2026-09-08 更新：** 下文按时间保留历史进度。v9 四臂 1.7B wiring 与 4B M1 pilot 随后均已
> 通过，必要门禁于 9 月 7 日 16:54 UTC 验收后进入主 screening。首个 4B A2 / seed42 cell
> 完成 11 轮更新，第 12 轮的一条 Train 响应触及 4096 cap，18:54 UTC 自动停止。
> 当前不是仍在跑 M1 pilot；主筛选未完成，原失败和 step-8 完整 checkpoint 保留。
> 新修复工作见 [9 月 8 日生成诊断计划](2026-09-08_M2_GENERATION_DIAGNOSTIC_PLAN.md)。

用户阅读当前报告并认可“先处理 4B 工程问题，再判断生成完整性问题”的顺序，现明确授权恢复。
本记录更新此前暂停状态；[05:53 UTC 实验报告](2026-09-07_M2_SCREENING_EXPERIMENT_REPORT.md)
仍作为暂停时的历史快照保留。

本次首先运行 R2 的 4B M2 Train-only 工程 pilot：相同两个 Train windows、seed 42、n=4、
两个完整 update rounds，逐 round 保存，并在同一 worker 中执行最终两条 Train greedy 评估。
generation、scorer、dtype、学习率和预算边界均沿用 R1/R2。模型从相同 exact base 初始化，
使用新目录，不把原失败重命名为成功，也不拼接旧样本。

运行入口为 `scripts/run_m2_screening.py --stage pilot --only-arm M2`；命名阶段日志用于定位
old-logprob、reference、value 和更新阶段的主机压力。R2 的 GPU 遥测错误不会主动终止 trainer。
如需进一步调整共同资源布局，将记录依据、验证和新代码版本后重跑必要 pilot。

当前授权的处理顺序不代表 1.7B 接线门禁已经通过，也不自动启动主 screening/dev。若 4B 也
出现生成不完整，先处理共同生成契约；若后续以 4B 接线替代 1.7B 门禁，需同步更新计划。

H100 继续使用 systemd user service、Linger=yes、stdin=null 和文件日志。当前进度、运行目录、
service、实际 commit 和失败证据更新在本机 `runs/M2_SCREENING_LATEST.md` 及对应 JSON 中。

## 13:03 UTC：R2 诊断结束，R3 组件验证完成

R2 首轮8条Train轨迹完整（188 steps），old log-prob完成；reference阶段连续532.567秒未返回，
同时持续触发主机memory.high。12:39:21 UTC记录原因并向核实的trainer PGID发送SIGTERM，
最终launcher FAILED / supervisor STOPPED。没有实际更新、checkpoint或评估，不算pilot通过。
一次NVML timeout被捕获后仍继续采样，R2修复已在实际故障中奏效。完整审计在
`runs/m2-screening-20260907-v5/audit.json`；资源峰和停止因果边界详见
[R3登记](2026-09-07_M2_SCREENING_EXECUTION_AMENDMENT_R3.md)。

R3先通过以下组件验证，再进入实际4B容量验证：

- 启动器37项、reference CPU29项、exchange/placement CPU30项、startup7项检查通过。
  exchange另含137个subtests。集成时修正了一个测试mock的旧模块路径，原失败日志保留。
- 实际sitecustomize安装通过，在明确开启profile时按reference→placement顺序生效，未初始化CUDA；
  默认入口保持原行为。全项目Ruff、64个src/scripts文件mypy通过。
- 随机82,304参数Qwen3、BF16、tied embedding/head、FP32 buffer、FlashAttention2/remove-padding，
  经真实builder→decorated worker→DataProto/DPActor调用，4次worker调用/11次microbatch forward的
  reference log-prob与原CPUOffload逐位一致；全部权重和共享aliases不变。
  原路径分配164,608 bytes CPU gradient，candidate为0，参数完整回到pageable CPU。
  首次probe缺少CC环境导致Triton编译失败，补生产已核实编译器后通过，原失败证据保留。
- 真实小型FP32 FSDP、BF16 forward、scalar AdamW，交错交换后连续3轮critic→actor（6次更新）
  的loss、参数、moments及step与同一普通AdamW对照逐位一致，包含首次空状态和不同模型尺寸。
- 对PyTorch退出引用警告做了3组独立生命周期隔离，各20轮：原CPUOffload、candidate、
  candidate加state_dict均在删除worker/GC时出现相同3条警告，循环中没有警告。
  各组预热后18轮CUDA allocated恒定，删除后相同；未发现迁移特有的累积。
  精确C++引用原因未定位，不能将小模型验证推成无限期或完整4B/Ray生命周期安全证明。

GPU证据、源文件SHA和隔离日志在 `runs/reference_candidate/`、`runs/optimizer_candidate/`。
reference原型与生产模块仅docstrings不同，已校验其余AST相同。组件验证未使用交易数据或真实
4B权重，未覆盖完整Ray/vLLM、checkpoint容量和主筛选效果；这些仍由新的两轮Train-only pilot验收。

## 13:32 UTC：R3首批进展与R4容量检查修正

R3/v6实际4B首批8条完整，reference30.29秒及critic一次参数更新均成功；actor更新前，
按soft limit预留2 GiB的CPU准入规则拒绝最后几块Adam状态。整轮FAILED，无actor更新、checkpoint
或评估；没有OOM、memory.high增量或GPU监控错误。失败状态及全部证据保留。

[R4](2026-09-07_M2_SCREENING_EXECUTION_AMENDMENT_R4.md)仅将CPU容量检查改为实际charged bytes
对64 GiB hard limit，继续留2 GiB、继续观察soft pressure，不提前抵扣缓存。38项CPU测试及
137个subtests、Ruff、mypy通过；独立审阅支持受监测两轮pilot，未承诺后续阶段的容量。
新身份从相同base重跑，不恢复半交换状态，不自动启动主筛选。最新服务和目录见本机LATEST记录。

## 14:10 UTC：R4 的 4B M2 工程 pilot 通过

`runs/m2-screening-20260907-v7/pilot-m2-s42` 使用冻结版本 `3828a97`，launcher于
14:10:29 UTC记录PASSED，supervisor于14:10:33 UTC记录PASSED，退出码为0。两个Train窗口、
seed 42、每窗口n=4的两批共16条轨迹完整；actor和critic各完成两次有实际参数变化的更新。
两批共385 steps、460,344 response tokens，M2实际诊断与完整sibling分组复算一致。

随后在同一worker中完成最终两条Train greedy评估（step 2、n=1、temperature=0）。
共48个请求、52,367 response tokens，invalid为0；最长响应2,626 tokens，均低于4096上限，
两条轨迹均到达规定终局。这里确认的是本次工程pilot完成，不能据此判断M2效果或整个主筛选
的生成完成率。详细证据在该run root的 `two_round_audit.json` 和 `final_pilot_audit.json`。

两轮均完成checkpoint写入；最终checkpoint通过完成标记、身份、必要文件及尺寸校验后，
按预登记的pilot权重丢弃策略归档。模型、Adam、RNG及data.pt已删除，保留完成标记、文件清单和
模型/tokenizer元数据。当前归档目录不支持恢复，本轮也未证明exact恢复重放。

主screening仍为 **0/8**；**1.7B四臂接线门禁和4B M1 pilot仍未通过**，不因本次M2成功自动
启动主筛选。Dev/正式Valid及Final OOS未访问。

资源审计随后确认：两套Adam完整物化后的两次双向交换各有148个storage完成同步copy/release；
第二次checkpoint保存97.74秒。device-wide NVML物理显存采样峰95,292 MiB，CPU采样峰约57.23 GiB，
memory.high新增6次，max/OOM/oom_kill无新增。保存边界CUDA driver-free最低约27.44 MiB，
rollout模式切换成功后回升约13.43 GiB；现有记账无法精确拆分峰值中的可回收缓存。
GPU数值为整卡使用量，不能全归因于trainer；Torch含vLLM分配池的统计不作为物理峰值。
这次两轮完整通过仍不保证主筛选所有batch/保存峰值均能容纳。
证据在v7下 `resource_two_round_audit.json`、`final_pilot_audit.json` 和归档清单。

## 14:21 UTC：进入1.7B固定输入诊断

旧R1失败请求44的prompt已由原环境重建：初始prompt和失败prompt均匹配原SHA；后者为2282 tokens，
SHA为 `674fc0ab8636f599ecc6e386291bd22e4d52ecde557cd683b3b4307efcf81a76`。
重建的是等价可见状态，不声称恢复全部未知的历史tool参数。只使用原Train episode。

单独将保留的step-1 actor离线导出为BF16，311个权重键和shape校验通过，原7个checkpoint文件和
FAILED状态未改动。在 `runs/m2-screening-20260907-v8/diagnostics/PROTOCOL.md` 中预先登记两个
固定请求：base及导出actor各一次greedy，同一原prompt、tokenizer、EOS规则和4096 cap，
不改变thinking或采样。使用独立systemd服务、新的standalone V1进程；与历史训练worker和两条
并发评测存在执行差异，故不承诺历史逐token复现，也不把单响应成功当作完整接线通过。

## 14:28 UTC：固定prompt未复现，验收当前R4下的1.7B完整接线

两个预定诊断请求均自然结束：base 1014 tokens，step-1 actor 636 tokens；均闭合think，
生成一个tool call并以有效EOS结束，有效停止集合均为[151643,151645]。没有修改prompt、thinking
或4096 cap。原4096-token循环未在本次standalone条件下复现，不能据此归因为M2更新，也不能
宣称历史错误已消失。两进程资源参数相同，但实际KV容量和原两并发执行不同，保留该限制。

下一次只运行 **一次 fresh R4 1.7B M2完整wiring**：相同两个Train windows、seed42、n4、
8条训练、critic/actor各一次更新、完整checkpoint及同worker最终2条Train greedy。
这是当前共同R4 reference/Adam/CUDA graph资源执行方式在1.7B上的首次完整门禁验收；
旧1.7B失败来自R1，standalone诊断不替代这项门禁。算法、scorer、generation和样本边界不变。
新目录新身份，不恢复旧checkpoint，不拼接样本。若相同截断再现，停止该候选并保留证据，
不以重复同配置直到成功的方式消除失败；主screening仍待全部必要门禁。
