# M2 R5：单工具、non-thinking 的有限训练工程候选（2026-09-08）

**状态：CPU 准备完成；9月8日获准清理后磁盘已达113.64GiB，通过112GiB磁盘门槛。**
尚未启动R5训练，启动前仍需核对当时GPU/主机资源及冻结身份；不是主screening通过或已恢复。
下文99GiB/BLOCKED数字为清理前的准备记录，执行更新见
[空间清理记录](2026-09-08_DISK_CLEANUP_AUDIT.md)。
本修订登记 G2 候选的真实训练门禁，配置为 `configs/m2_screening_r5_generation_pilot.yaml`。
默认 `configs/m2_screening.yaml` 继续保留 R4，所有新运行必须明确指定本候选配置。
本次执行只到下表规定的工程检查结束；不使用 `--stage all`，不自动启动主筛选或 Dev。
这里是当前修复工作的执行边界，用户原先对实现与调试 screening 的授权持续有效。

## 依据与待回答的问题

[9 月 8 日诊断报告](2026-09-08_M2_GENERATION_DIAGNOSTIC_REPORT.md) 记录：

- G2：`single_tool_v1 + enable_thinking=False` 的固定 18 请求全部通过；4B base 和 step-8
  actor 各 160 条完整 Train 轨迹，共 6619 请求，自然 EOS、非法动作 0，但 buy/sell 均为 0。
- G3：相同指令保留 thinking，在固定 18 请求中出现一次 4096-token 截断，17/18；按登记
  停止条件式 320 条扩展，未补抽或扩大上限。G3 不用于本次训练候选。

G2 的原完整性门禁已经通过，不能事后用交易数量将其改判失败。不过，它只验证了冻结
策略的空仓交互，尚未验证参数更新及同 worker 的训练后生成。R5 用有限工程预算回答这
个问题，同时记录实际动作和 residual 是否得到执行；不继续搜索 prompt 到出现交易为止。

G2 推理的 step-8 actor 并非原失败时已经更新 11 次的 policy，也不是 R5 初始化来源。
旧 R4 失败 checkpoint、G1/G2/G3 的全部正负结果和准备阶段失败证据保留。

既有 G2 采样轨迹的 CPU 复核确认，同窗四 siblings 的 Y 相同，plain/pooled residual
按既定容差均为零；hold/abstain 的 family diversity 本身不产生 residual。该复核使用
真实账户观察、完整应用轨迹和冻结 scorer/标定重建终局现金，未另行生成轨迹，也没有
将 outcome_only 的零 reward 当作 Y=0。证据
`runs/m2-screening-20260908-v10/diagnostics/single_tool_v1/g2_frozen_residual_audit.json`，
SHA256 `e535043bd5ebd948a9863a73f9909b1f62578acb6979cf061991caa05f27d770`。
它是未覆盖机制的限制说明，不是实际更新验证或事后新增门槛。

## 共同配置

相对 R4 唯一的模型可见改动是共同使用 `single_tool_v1` 指令与显式
`generation.enable_thinking=False`。helper 文件 SHA256 为
`299e0d197142ab2b8e169cb7a3d98dc8eb91b223a08e34fb27af884b95b7c20c`，规则 SHA256 为
`3ca4fc62f06f96c03bbe704103dea741ff74ee5a626b3b6aebd849cd3eee6b45`。
指令通过消息副本注入，四臂、Train 最终 greedy 使用同一生产路径；接口也支持后续经
单独决定的共同 screening/eval/zero-shot 配置，不在本次执行这些阶段。

保留 R4 的 pageable Adam/reference/CUDA graph 资源方式、BF16、actor/critic 学习率
1e-6/1e-5、scorer 与已冻结标定、GAE/residual/whitening 算法、两窗口 batch 与 n=4。
prompt=4096、response=4096、context=8192、最多256 agent steps；采样温度1、top_p=1、
top_k=-1，最终 Train greedy 温度0。4096 长度保护、真实终局检查和失败保留规则不变。
使用原生 non-thinking 模板（prompt 含原生空 thinking 前缀），不对已生成响应补写标签或
强制截停，不新增 logits 处理器、动作屏蔽或改变既定采样参数；不修改奖励以强制交易。

准备好的 Train 数据及标定继续沿用原身份，不重选窗口或重标定 scorer：

- `train.parquet` SHA256：`10c4265418681421d572a30e529f00c957086ddadaa4657f62ec83f594ff78fa`。
- prepared manifest payload SHA256：`0e5be980d74e2a6dd5b1ab423acb29a90d4cb0bbebf4e35bbe203f12accc7dae`。
- calibration payload SHA256：`eaf1e50207ba727536a7fabb21d0e3ee3be12bd7569bd3e21e5d5696ab421a7e`。

## 固定预算与顺序

所有新 cells 使用新目录及冻结代码/配置身份，从对应的同一个 exact base 开始，不恢复
旧 checkpoint、不继承上一 arm 权重、不抵扣旧 R4 已通过的门禁。

| 顺序 | cells | 每 cell 更新轮数 | Train 轨迹总数 | 同 worker 最终 Train greedy |
| --- | --- | ---: | ---: | ---: |
| 1 | 1.7B，A2 / M1 / M2-plain / M2，seed42 | 1 | 4×8=32 | 4×2=8 |
| 2 | 4B，M1 / M2，seed42 | 2 | 2×16=32 | 2×2=4 |

合计 **64 条训练 + 12 条 Train greedy = 76 条工程轨迹**。1.7B revision 为
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；4B revision 为
`1cfa9a7208912126459214e8b04321603b3df60c`。窗口、seed 推导及更新/保存顺序沿用原门禁。
第一阶段完整通过及证据复核后才执行第二阶段；任一失败停止后续队列，不重复同配置直到成功。

每轮核实 actor/critic 的真实更新、参数变化及完整 checkpoint。最终保存后，在同一
worker 保留既定 Adam 生命周期执行最终两条 Train greedy；独立导出推理不能替代此验收。
记录 GPU/host 压力、token/step 数、响应和终局完整性。PASSED cell 沿用原经过验证的归档
流程；FAILED cell 不自动删除权重。诊断用有界 exporter 的成功不替代生产归档准入。

## 机制读数与停止边界

沿用原计划的 family histogram、diversity、pooling opportunity、plain/pooled 差异、
nonzero_S_rate、mean_abs_S、U/S 与一次最终 whitening 诊断，另报告真实买卖与持仓覆盖。
不引入最低买卖次数、最低 diversity 比例或新收益硬门槛。

如果两轮 pilot 全部完整但 S 为零，结论为“本次工程链路可运行，pilot 未覆盖 residual
作用”，不能写成“M2 已验证”或“M2 无效”。原计划针对完整主筛选各 seed 的
`NO-RESIDUAL-EXPOSURE` 与 `POOLING-NOT-EXERCISED` 规则继续保留，不直接套用于两轮 pilot。

**本批 76 条结束后停止并写出读数，先判断机制覆盖及剩余问题，再决定主筛选执行。**
不追加更新直到出现买卖；不访问 screen-dev、正式 Valid 或 Final OOS。若后续需要再改
候选，先登记理由、预算与四臂共同设置，不使用收益挑选 prompt。

## 资源与持久运行

服务器现有约99GiB 可用空间；生产 supervisor 每个新 cell 前要求至少112GiB。用户已表示
将向管理员申请增加100GB，尚未确认完成。准备工作与 CPU 验证可以先完成，**训练要等实际
空间检查通过**；不降低 guard、不删除原失败 checkpoint，也不将申请视为已扩容。

沿用服务器 systemd user service、Linger=yes、stdin=null、KillMode=control-group、
Restart=no 和文件日志，不依赖本机 SSH。启动时冻结 commit/config SHA、运行根目录、
完整命令、环境及 unit 属性。只分别使用现有 supervisor 的 `--stage wiring` 和
`--stage pilot`，明确传入 `--settings configs/m2_screening_r5_generation_pilot.yaml`。
运行前还需重新核对资源、代码身份及已有候选证据；本文件不是实际服务启动收据。

## CPU 准备结果

候选配置 SHA256：`c59d0c3c9b9f8107bbba1ba1b6151d5501c194fde7eb0f0722a3fd38d8c26715`。
6 个必要 cells 经真实 `build_cell_config → build_cell_command → Hydra compose` 和
Ray RuntimeEnv schema 验证通过。生成模板、四臂、fresh base、Train-only 评估路径、
64+12预算及完整 R4 资源/scorer 设置一致。没有调用 launcher、ray.init 或 GPU。

CPU 收据为 `runs/m2-screening-20260908-v10/diagnostics/r5_preparation/cpu_preparation.json`，
SHA256 `68b504ed793b6d006cdbc65e9aeb97ce244b7bfcf5859cc50af4668a7a6a7bc6`。
同目录 `preview_validated/` 保留 resolved 命令和配置。它们使用提交前代码身份及预览目录，
**不能直接当正式启动命令使用**；实际运行时需在冻结 commit 和新真实路径重新生成身份。

首次 CPU checker 对 Hydra 将 `resume_from_path=None` 表示为字符串的假设错误已经修正，
原日志保留。真实 `resume_mode=disable` 分支直接跳过恢复；没有改变生产参数或运行模型。
最终 CPU 配置检查为 PASSED，磁盘检查仍为 BLOCKED（99.11GiB < 112GiB），两者分别记录。
