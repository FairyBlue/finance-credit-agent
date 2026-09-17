# 共同 SFT 后固定两轮 M1/M2 短 RL 计划

制定时间：2026-09-14 01:51 SGT（UTC+8）。**制定时尚未应用 RL 候选补丁、尚未启动 RL。**
当时 v3 SFT 与模型导出已完成，验收仍在进行；实际执行进度另记于 runs/M2_SCREENING_LATEST.md。

本轮回答：在同一个能完成合法自主交易的共同 SFT 起点上，M1 和 M2 各训练两轮后，
交易行为与含成本收益如何变化。共同教师先验保持一致，M2 的差别仍来自既有 step
credit/reward 算法。先完成这个小预算对照，不增加手工投资策略，不自动延长训练。

## 启动条件和共同模型

必须同时满足：

- `runs/m2-coldstart-v3-20260914/readout.json` 的原共同初始化门槛为 **PASS**：
  两套共 185 条验收全部完整、各套模型非法率不超过 5%；40 条错误前缀轨迹全部完整
  且接管后非法率不超过 5%；各套至少半数 sampled contexts 的四兄弟完整且收益范围
  大于 `1e-8`；至少半数完整 full sampled 轨迹在两个以上日期真实成交。
- v3 控制器完成，当前独立服务已退出；确认 H100 没有该服务遗留训练/评估进程。
- 下述模型校验及短 RL 资源检查通过。验收 FAIL 则保留结果，此计划不启动。

共同 actor、reference、critic backbone 和 tokenizer 使用：

`/home/tianming/projects/finance-credit-agent/runs/m2-coldstart-v3-20260914/merged_model`

实际 `merge_receipt.json` SHA-256：

`cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db`

该 receipt 于 01:42:00 SGT 完成，记录 v2 merged backbone 加 v3 adapter 的最终导出。
入口显式采用 `model.source=local_export`，检查 receipt 身份及完整权重/config/tokenizer
文件清单、字节数和 SHA-256；不冒充 HF commit。任何一臂均不从另一臂 checkpoint 开始，
也不加载 SFT optimizer。两臂各创建新的 optimizer。

## 固定预算与共同条件

| 顺序 | 起点与更新 | 自主训练轨迹 | 同 worker 最终 greedy 评估 |
|---|---|---:|---:|
| SFT baseline | 同一 SFT，0 次 RL 更新 | 0 | 4 |
| M1 | 从同一 SFT 独立开始，固定两轮；actor/critic 各更新两次 | 16 | 4 |
| M2 | 从同一 SFT 独立开始，固定两轮；actor/critic 各更新两次 | 16 | 4 |

baseline 走与训练相同的框架/worker，以 M1 的 `val_only` 路径执行；不会训练或保存
RL checkpoint。每臂每轮两个 Train 窗口、每窗口 `n=4`，最终评估在两轮更新之后进行。
同一批轨迹不跨臂拼接；失败不重抽，不接续旧 checkpoint，不运行旧 M2 自适应 gate。

复用既有 Train splitter，准备 `0001/0005/0009/0013/0017/0021/0025/0029` 八个窗口；
本次两轮实际使用前四个 `formal-train-0001/0005/0009/0013`。评估固定为
`formal-train-0113/0117/0121/0125`，全部 21 sessions；May 23–June 13 区间不并入训练。
不访问正式 Test。

共同条件固定为 seed42、`history_features_v1`、`single_tool_portfolio_v1`、non-thinking，
沿用现有 scorer、账本、generation 预算、pageable-Adam 与 vLLM 资源方式。
`cpu_seed_v1` 初始化新 critic value head。各 cell 在 HF 加载后、FSDP 包装前记录小型
value head 的实际 shape/dtype/seed/SHA-256；M1、M2 的初始 head hash 必须与 baseline
一致。只读取这层约数千个值，不做 4B 整模 GPU→CPU 复制；主干身份由共同导出文件校验。

## 候选接入与执行

仅在启动条件满足后，按以下顺序应用已隔离测试的候选；第三项使用 **r2 替代 r1**：

1. `runs/m2-coldstart-v2-20260913/rl_preparation/local_export_candidate.patch`
2. `runs/m2-coldstart-v2-20260913/checkpoint_preparation/research_model_extra_v1.patch`
3. `runs/m2-coldstart-v2-20260913/rl_integration_preparation/common_sft_fixed2_integration_r2.patch`

三项依次 SHA-256 为：

```text
73cf3d4515023ea6630e9b192b39fd82ff944d5e4d21878d24cb21d856b9ba98
b13987fa1411bcf8bd22315bddfbafbedabf65aca50530ea8e66f0490e37b4d3
e4a190d67cf3a6b72b18d458daf22dbe71a1da9b3ba06f824ce4a9444cb870d9
```

r2 额外封存 `prepared/manifest.json`，防止两臂间同时误改数据与当前校验清单。
原补丁及旧验证记录均保留，详见隔离目录的 `README_R2.md`、`validation_r2.json`。
应用后运行相关 CPU 入口与保存/身份检查；这些通过不代替本轮 GPU 验证。
接入时将 `coldstart_learning.settings` 的 `plan` 元数据路径从隔离 README 改为本文件
`docs/2026-09-14_M2_COMMON_SFT_SHORT_RL_PLAN.md`，随后测试并封存实际源码；这不改变
训练算法或预算。运行目录始终显式传入，不沿用候选代码中的旧默认目录。

新运行目录为 `runs/m2-common-sft-short-rl-20260914`。在项目根目录执行：

```bash
PYTHONPATH=.:third_party/agent-r1 .venv/bin/python -m src.agents.coldstart_learning prepare --directory runs/m2-common-sft-short-rl-20260914 --model /home/tianming/projects/finance-credit-agent/runs/m2-coldstart-v3-20260914/merged_model --merge-receipt-sha256 cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db
PYTHONPATH=.:third_party/agent-r1 .venv/bin/python -m src.agents.coldstart_learning preflight --directory runs/m2-common-sft-short-rl-20260914
PYTHONPATH=.:third_party/agent-r1 .venv/bin/python -m src.agents.coldstart_learning seal --directory runs/m2-common-sft-short-rl-20260914
```

随后用独立 systemd user service 执行 `coldstart_learning run --directory`，指向上述
新目录。沿用隔离服务模板，替换目录和 unit 名为 `m2-common-sft-short-rl-20260914`；
使用绝对 Python/项目路径、已验证 CC 编译器和离线模型环境，`Linger=yes`、`Restart=no`、
`KillMode=control-group`，四小时总启动超时。日志、SGT 时间和状态落盘，SSH/本机网络
断开不会终止服务。不把服务模板存在或 CPU 预检通过记作 RL 已启动。

## 产物、资源与读数

每臂只在第二轮保存一次 `research_model_extra_v1`：actor/critic 权重、scheduler/RNG、
dataloader 及 HF/FSDP 元数据。明确 `optimizer_included=false`、`resume_supported=false`，
**这是不可恢复训练的研究产物**。不保存 Adam 状态；评估前先保存，评估失败仍保留
已更新的模型和证据，不退回完整保存模式、不删除旧文件。

两臂预计合计约 62.8 GiB。入口要求 baseline/M1 前至少 80 GiB 可用空间、M2 前至少
48 GiB；另确认 H100 与主机可用内存适合原 RL 资源配置，不直接套用 SFT 的内存上限。
空间或内存不足则停止并记录，不通过清理历史模型继续。保留真实 actor/critic 更新、
完整轨迹、M2 非零 credit 信号、初始 head 身份及最终保存证据。

报告三者的完整性、非法率、hold/买卖、实际成交、股票覆盖、仓位集中度、交易成本和
含成本净收益；将 M1/M2 相对共同 SFT 的变化及两臂差异一起呈现。收益相同、收益下降、
credit 信号缺失或任一工程失败都如实保留，固定两轮结束后再决定下一步。

共同 SFT 含原 Train 内三日未来收益教师标签、20%/清仓倾向及 API 示范，本轮不能把
这些共同先验的效果归于 M2。四个评估窗口已用于诊断，是重复 Train 回归读数；此次
没有 A1 对照，也不支持正式 OOS、M2 优于 M1/A1 或奖励方法有效性的最终结论。
