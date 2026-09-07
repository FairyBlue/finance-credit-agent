# M2 screening 恢复执行记录（2026-09-07）

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
