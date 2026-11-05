# 共同初始化与 4B 学习（2026-09-12—10-08）

整理日期：2026-10-18。本文合并提示探索、共同 SFT、短 RL、4B 两 seed 对照及早期网格的关键结果。
最新结论与下一步见[当前状态](../CURRENT_STATUS.md)，逐次报告、冻结方案和来源哈希见[历史归档](../archive/README.md)。
`runs/` 路径相对仓库根目录，完整产物主要在服务器；日期内的等待、容量及模型保留状态均是历史记录。

**共同 SFT v3 通过合法性门槛，4B 全参数训练随后跑通；两 seed 四轮出现局部正向结果，sampled 的 M2 增量未稳定重复。**
早期八轮 RL 的收益回落、拒单恢复和三股教师先验均保留；工程通过不等于方法收益成立。
10/08 R5 的启动快照已被后续完成结果取代，固定网格及勘误见[恢复与动作池化中的 R5 汇总](RECOVERY_AND_ACTION_POOLING.md)。

## 从提示探索转向共同 SFT

| 实验 | 实际结果 | 失败与边界 |
|---|---|---|
| 9/12 提示迭代 | 48/48 完整；现金起点原提示/指引/指引+示例分别 0/8、6/8、8/8 成交，净收益 0%/+0.859%/+0.172% | 未训练权重；已有持仓管理未改善；示例组短期反向交易 10 次、指引组 2 次；两次 sell 实际反向 buy |
| 9/13 base 重启 | M1/M2 各四次双角色更新；64 条训练、12 条评估完整，最终均零成交/零收益 | 唯一买入在首次更新前；M2 首批非零 S 44/160、后三批全零；critic head 当时未实际使用配置 seed |
| SFT v1 | 原模型 100 条全无成交；补训后 99/100 完整、594 笔成交；完整 greedy/sample 净收益 +3.326%/+2.448% | 非法 207/846（24.47%），一条拒单循环耗尽请求；编译器失败补 CC 后同模型恢复，非响应截断 |
| SFT v2 | 新协议前后共 370 条观察；补训后 185/185 完整；两组非法率 20.79%→12.93%、17.55%→11.43% | 均未过 5% 门槛；旧完整 greedy/sample 收益 +3.1914%/+2.6424%→+0.9973%/+1.8196%，成本上升 |
| SFT v3 | 185/185 完整，两组非法率 1.23%/2.21%；40 条错误前缀回合完整、非法率 1.64%，八项固定门槛 PASS | 采集原有 39/40 完整、一条 REQUEST_CAP；旧完整 sampled 收益 1.8196%→1.3044%，八组成本均上升 |

订单方向边界已修复：next-open 如需反向成交则零成交、零费用并返回原因，未加人工最小调仓幅度。
重启之后的 `cpu_seed_v1` 在 critic 构建时设共同 CPU seed 并恢复外层 RNG；短 RL 核对三组初始 head 哈希一致，不能追溯改写旧重启身份。
共同观察为 `history_features_v1`、non-thinking；`single_tool_portfolio_v1` 说明目标总权重、买卖方向及拒单不推进日期等接口事实，不自动改写非法动作。

## 冷启动数据与共同先验

| 版本 | 真实训练预算 |
|---|---|
| v1 | Train 前 20 窗/93 个去重日期，372 条决策 + 96 条 API 示范 = 468 条；LoRA r16/alpha32，七类 attention/MLP 投影，lr1e-4、3 epochs、90 更新 |
| v2 | 现金/单股/多股及四类错误反馈状态，加旧数据共 740 条；从 v1 开始，lr5e-5、2 epochs、94 更新 |
| v3 | 从模型实际访问状态选 472 个不同 prompt（错误及后续 190、普通 282）+ 原 740 = 1212 条；从 v2 开始，lr5e-5、2 epochs、152 更新 |

三轮仅监督 completion+EOS，BF16 固定预算，无示范截断、未按验收收益延训。离线教师在 GS/MSFT/HD 合法网格和 hold 中，
按未来三个 session 的真实成本后终值选标签；未来价格未进入输入。v3 新标签 buy248/sell176/hold48，buy 全为 20%、sell 全清仓，
验收 909 笔成交仍只覆盖三股。这是带交易先验的共同初始化，不能称纯 JSON 训练，也不能计作 M2 的贡献。
教师/训练日期不晚于 2023-05-22；另一组 Train 为 05-23—06-13（完整片段 15 sessions），旧回归为 06-14—08-01（完整回合 21 sessions）。
两套均反复查看，窗口/siblings 非独立，非新鲜 OOS；模型成交/成本排除构造前缀，短回合从接管资产计收益，含前缀指标另列。

## 两轮、八轮与日期覆盖

共同 SFT v3 保留零更新基线。M1 的 S 为零；M2 使用 `Whiten(U+ηS)`，S 只进 actor，critic returns 不加 S。
下表为原四个 Train 验收窗 `0113/0117/0121/0125`（均前缀 `formal-train-`）、含成本平均净收益；执行路径分别比较。

| 固定终点 | SFT | M1 | M2 η1 | M2 η0.25 |
|---|---:|---:|---:|---:|
| 同 worker，两轮 greedy | 1.648694% | 1.990035% | 1.652102% | — |
| 同 worker，八轮 greedy | 1.648694% | 0.729263% | 0.312604% | 0.172757% |
| 八轮模型 standalone sampled | 1.4238% | 0.4577% | 0.3210% | 0.6545% |
| 八轮模型 standalone greedy | 1.6187% | 0.0333% | 0.7128% | 0.7944% |

两轮 44/44 完整，M2 非零 S 180/320，但低于 M1 0.337933 个百分点。critic 首轮 half-MSE 在 LR1e-5 下 0.222238→60.663093，
LR1e-6 下 0.221099→0.189285；两组各两轮/44 条完整，按同批拟合的事前规则选 1e-6，未按收益选 LR。
随后八轮主比较 244/244 完整；η0.25 第二轮曾 2.281889%，最终 0.172757%，未用事后最佳轮替代终点；仅第八轮 actor 导出。
交叉评估 240 次/228 完整，12 条缺失全为训练日期 greedy 拒单循环、达 128 请求上限；另一组日期三种 RL sampled 均低于 SFT。
worker/standalone 的 greedy 排序改变，不能直接归因 vLLM。9/15 当前采样 T1/p1/k−1 为 192/192 完整，推荐 .7/.8/20 为 177/192、
15 条训练日期拒单循环；收益有升有降，未解决 RL 低于 SFT，保留当前配置。所有响应正常 stop、最长 28 tokens，失败非 4096-token 截断。

9/15—16 两次覆盖尝试各 28/308 后因新增诊断误拒 GSPO/空 multi_modal_inputs 停止，均只有一次 M1 critic 更新、零 actor 更新。
修正后 fresh 执行的新目录 9/16 18:55 完成 **308/308**（128 训练+180 验收），两臂各八次 actor/critic 更新，不拼接旧轨迹/optimizer。
actor LR 降到 2e-7、critic 1e-6、η0.25，16 个 Train 窗覆盖 154 个交易日；日期与 LR 同时变化，不能单独归因。

| 固定第八轮 | SFT | M1 | M2 η0.25 |
|---|---:|---:|---:|
| Greedy 净收益 | 1.618652% | 1.565643% | 1.823533% |
| Sampled 净收益 | 1.426767% | 1.263738% | 1.287828% |
| 训练非法/请求 | — | 111/1391 | 95/1375 |

M2−M1 为 greedy +0.257889、sampled +0.024090 个百分点，但 M2 sampled 仍低于 SFT 0.138939 个百分点，金融 Y 略低于 M1。
第 2→8 轮 sampled 分别降 0.369704/0.466113 个百分点，手续费仅解释 2.65%/0.42%，主要算术差额来自持仓路径，非逐渐全现金。
180 条验收/3442 笔成交/2766 训练步重算未发现新记账或执行 bug。非法步骤实际 A 为正为 M1 34/111、M2 38/95，
其中更新后平均 logp 上升仅 17/34、18/38；直接 reward/A 符号不足以证明强化。M2 非法直接 S 全零，离线 η0 后 95/95 A 符号不变，
但仍可能经全批 whitening 尺度间接影响。critic 偏差/窗口均值尚未定位为收益根因；账面分解与少数 lane 的不利路径不构成训练因果证明。

## 拒单罚与低显存训练

9/17 登记新共同目标 `r′=Y/N−0.05×invalid`，末步保留余差，总和为 `Y−0.05×N_invalid`；两臂 GAE/critic 用 r′，
M2 residual 仍用原金融 Y，评估仍报原净收益/Y。此为共同目标修订，非原公式 bug 修复，亦不保证每条非法 A 为负。
该批原预算 164 条，准备通过后排队，最终 **9/18 03:22:57 WAIT_TIMEOUT、0/164、零更新**；原“仍排队”已失效。

`chunked_adam_disk_v1` 保留 actor/critic FP32 全参数、全局裁剪/调度；Adam FP32 moments 活跃时 pageable CPU、非活跃时经 SHA/fsync 验证落盘，
GPU 临时分块≤64 MiB，配合 BF16 reference、vLLM sleep、microbatch1 和 checkpointing，未改 LoRA 或量化 optimizer。
9/30 低显存 pilot 在 reference 初始化 OOM，0/72、未运行到分块 Adam；放行时 62.43 GiB 随外部占用下降，不能据此证明 60 GiB 可训。
10/04 handoff pilot **72/72** 完整，两臂各 16 训练/20 评估、各两次双角色更新；首次加载交接空闲 **75.7156 GiB**，峰值约 69.5 GiB allocated/72.6 reserved，
没有严格 60 GiB 验收。八次分块更新/第二轮 moments 恢复通过，约 68.6 分钟；主存峰 57.19/57.58 GiB，无新增 host OOM，曾触软限。
本批 M1/M2 greedy 1.6874%/1.7704%、sampled 1.5063%/1.6010%；未重跑 SFT0，仅工程及同批两臂比较，无 stock Adam 速度对照。

## 两 seed 四轮：局部正向，sampled 未稳定重复

共同 SFT v3 fresh 起点，actor/reference/critic backbone 相同，同 seed 初始 critic head 哈希一致；actor LR2e-7、critic1e-6，M1η0/M2η0.25，
共同拒单罚、原金融 Y residual；八个训练窗 `0001/0065/0009/0073/0017/0149/0025/0157`，原四个验收窗，每 seed 164 条。
最终第四轮事前固定，非按最佳收益择点；各臂四次双角色更新，两 seed 各 164/164 完整。

| 第四轮净收益 | seed42 greedy | seed42 sampled | seed43 greedy | seed43 sampled |
|---|---:|---:|---:|---:|
| SFT0 | 1.6187% | 1.5094% | 1.6187% | 1.1089% |
| M1 | 1.7704% | 1.6794% | 1.7018% | 1.4229% |
| M2 | 1.8481% | 1.8861% | 1.9529% | 1.4261% |
| M2−M1（百分点） | +0.0777 | +0.2067 | +0.2511 | +0.0033 |

seed43 sampled 仅约 0.33 基点，近乎持平。seed42 第二轮 M2 曾低于 M1，seed43 第二轮两臂 sampled 低于 SFT，不能称持续改善。
四窗只覆盖 33 个不同交易日且互相重叠，非独立胜窗；所有均值低于约 4.2216% 等权 benchmark。sampled 非法率 M1/M2：
seed42 为 11/331、7/327，seed43 为 13/333、16/336；回撤两 seed 较低但非法改善未重复，seed43 M2 金融 Y −5.3316 低于 M1 −5.3005。
seed43 原服务因只读验收残留 run_seed==42 误报 FAILED；显式 expected_seed 修复后复验原 SFT/M1 的 92 条，recovery 仅 fresh 执行 M2 的 72 条，未重训前两臂。
BF16 actor 各约 8.22 GiB，非包含 optimizer/critic/RNG 的续训 checkpoint；临时 moments 仅在导出和证据验收后按收据退休。

## 扩大网格、未完成评估与资源协调

10/06 固定两 seed × actor LR1e-7/2e-7 × M1/M2η0.125/M2η0.25，共 12 次四轮 RL、1944 次尝试（384 训练、1560 评估）。
12 个非重叠 21-session dev 窗覆盖 2023-11-01—2024-10-31 的 252 个交易日，与训练日期隔离；奖励标定已接触，仍非 untouched/OOS。
原批仅 50 次 SFT（49 完整）因 greedy MSFT20.137266% 反复 buy到20% 耗尽请求停止，零训练更新；这是拒单恢复问题，正常短响应。
R1 `retain_prefix_v1` 保留 INCOMPLETE prefix、金融收益/Y 为空，传输用 0 单列；所有评估槽位均尝试，共同完整窗口比较须报排除数。
`COMPLETE_WITH_EVALUATION_FAILURES` 不是性能 PASS；训练不完整/工程或身份异常仍停止。R1 最终共享显存骤降 OOM，288 条落盘、282 完整、
6 条 greedy 未完成，48 条训练全完整；M1 LR1e-7 四轮已完成，M2η0.125 仅两轮，缺第四轮/seed43，不能选 η 或稳定参数范围。
新模型用绑定 SFT v3 的无损 tensor XOR 归档，真实约 8.82GB→119MiB 并精确恢复；归档不可直接加载，依赖完整基座，不含续训状态。
10/07 持久 holder 目标76GiB、上限80GiB、留2GiB驱动空闲；接手时暂停增长、分配前释放并复查，子进程仍在时不补回，但释放存在竞争窗口，非独占保证。
10/08 R2/R3/R4 因自身驻留识别/CPU设备文件误拒/vLLM cache 初始化先后失败，均零轨迹；R5 16:42 启动、16:52 容量等待仅历史快照，
其后 **10/09 07:30 完成固定1944次网格，含29条未完成greedy评估**；性能和10/11勘误见恢复与动作池化汇总，未把旧失败拼入新批。

## 证据、存储与结论范围

| 事项 | 原始证据（`runs/` 下） |
|---|---|
| 提示、重启与共同 SFT | `m2-train-practice-20260912/readout.json`；`m2-learning-restart-20260913/critic_initialization_review/`；`m2-coldstart-20260913/`、`m2-coldstart-v2-20260913/`、`m2-coldstart-v3-20260914/` 的 readout/manifest/逐请求与 validation |
| 短 RL、critic、交叉/采样 | `m2-common-sft-short-rl-20260914/`；`m2-critic-lr-diagnostic-20260914/critic_fit_analysis.json`；`m2-training-tuning-20260914/FINAL_RESULTS.md`；`m2-policy-transfer-eval-20260914-continuation/joint_readout.json`；`m2-sampling-profile-eval-20260915/FINAL_RESULTS.md` |
| 覆盖/拒单罚终态 | `m2-train-coverage-20260916-v2/FINAL_RESULTS.md`；`m2-train-coverage-diagnosis-20260917/{behavior,signals,sampling}/`；`m2-common-validity-penalty-20260917/{FINAL_RESULTS.md,completion_validation.json}` |
| 低显存与四轮 | `m2-low-memory-pilot-20260930/`；`m2-low-memory-handoff-20261004/completion_readout.json`；`m2-sft-four-round-comparison-20261004/comparison_readout.json`；`m2-seed43-revalidation-20261005/`；`m2-sft-four-round-comparison-seed43-20261005-recovery/` |
| 网格、归档、持久协调 | `m2-grid-preparation-20261006/{failure_analysis.json,archive-roundtrip-v2/roundtrip_receipt.json}`；`m2-parameter-grid-20261006-r1/`；`gpu-reservation-persistent-20261007/`；`m2-parameter-grid-20261008-r5/` |
| 清理与存储收据 | `disk-audit-20260917/optimizer_cleanup_receipt.json`；`disk-audit-20261004/scratch_cleanup_receipt.json`；`disk-audit-20261008/` 下 optimizer/storage/scratch 收据 |

经确认的历史清理包括 9/17 六份旧 Adam 约171.24GiB、10/05 pilot moment scratch 约114.16GiB；模型保留并不保留原 Adam 续训能力。
10/08 旧80.41GiB optimizer 删除后实际可用未增加，ZFS原因未验证；失败M2的715个raw实际回收约33GiB，另389个当时保留，不能混报。
SFT v3 也是 XOR 解码基座，不能因旧模型清单而删除依赖；当前存储以新收据为准。服务用独立 systemd/Linger，未做真实断网注入。
以上仅 Train 内探索、少量训练 seed、反复查看的开发评估，无完整 A1/A2 和正式 Valid/Final OOS；教师先验属于所有臂共同起点。
合法性、非零 S、真实更新和局部正向均值支持继续检验，均不单独证明 M2 的稳定方法优势。
