# 基础接线与行为诊断（2026-08-27—09-12）

整理日期：2026-10-18。本文合并 H100/M1 接线、M2 screening 工程、交易行为和概率诊断。
最新结论与下一步见[当前状态](../CURRENT_STATUS.md)，逐次报告、冻结方案及来源哈希见[历史归档](../archive/README.md)。
`runs/` 路径相对仓库根目录，完整产物主要保存在服务器；这里的资源与服务状态均是历史读数。

**本阶段接通了真实训练与 M2 信号链路，但初始 screening 没有完成，模型主要保持现金。**
工具、观察与记忆确有缺陷并已修复；事实理解、合法生成和工程通过均不能代替交易与收益改善。
后续因此转向[共同初始化与 4B 学习](INITIALIZATION_AND_4B_LEARNING.md)。

## 接线与数据门禁

| 项目 | 已完成结果 | 结论边界 |
|---|---|---|
| H100 与输入预算 | 真实 tokenizer、紧凑观察、滚动历史等修复后，620 个预算案例最大 3045/4096 tokens | 原首步 5141、历史累积至 40285；ASCII 检查漏过 4548-token Unicode 反例，预算必须覆盖完整允许路径 |
| G1 / A1 / 1.7B | 11 条轨迹、259 次请求、2 次真实 optimizer 更新；代码 `2d48bf6db8961998b1450644ef542a293ea8f95d` | 未保存完整 checkpoint，第二步 reward/advantage 为零；仅证明更新链路 |
| T2 v2 | balanced accuracy 0.499357；强/弱正控制 1.0/0.595495，联合规则通过 | 未显著检出泄漏不等于证明无泄漏；结果在 `configs/t2_v2_result.json` |
| formal manifest | Train 441、Valid 60 个 21-session 窗；Test 1 个 83-session Final OOS | 准备 Test 不等于评估 Test |
| M1 / 1.7B | 1 条 Train、更新前后各 1 条 Valid；actor/critic 各更新一次，代码 `fb07e727b8077c9f25684ad3bfa5222feef84035` | 两次 Valid 终局分数相同，是 reward/GAE/更新接线 smoke |

M1 把终局分数分配为 immediate-step rewards，末步吸收浮点余差；之后仍计算真实 GAE、更新 critic/actor，
不是均匀广播 optimizer advantage。环境固定 exact snapshot/offline、manifest 与 expected commit；安装后用 `--no-sync` 防止漂移。

## M2 screening：工程通过，正式筛选未完成

原设计为 A2/M1/M2-plain/M2 × seeds42/43、Qwen3-4B，共八个主 cells；每 cell 32 个 Train 窗、
每窗四 siblings、16 次更新，在 12 个 Train 内 dev 窗评估。A2 用原始净收益，其他臂用 uniform Y，最终共用 Y。
M2/plain 的 residual 在 GAE 后、whitening 前只加到 actor，critic returns 不变；全 logical batch 的真实 steps 统一 whitening。
Train 代理标定为 1408 条轨迹/28160 步，floor=0.0007613112285332427、λ=1.6912508025279984，均非模型收益。

| 阶段 | 实际结果与失败 |
|---|---|
| v3/v4，1.7B | 训练各 8 条完整并真实更新；v4 的 S 非零 24/160、pooling 改变 15/160；最终评估均 0/2 完整，4096 cap 仍截断 |
| 4B v4-pilot/v5 | 首批各 8 条完整、零更新；先是遥测超时误杀 trainer，后是 reference 532.567 秒未返回并持续 host 限流，主动停止；不是已证明的模型 OOM |
| v6/R3 → v7/R4 | v6 critic 更新一次、actor 被容量准入拒绝；R4 改按真实 charged bytes/hard limit 后，16 条 Train、两轮双角色更新及 2 条 greedy 全通过 |
| v9 主 cell | A2 完成 11 轮、88 条 Train，第 12 轮触及 4096 cap；主筛选仍 0/8 完整，保留 step-8 actor；R4 显存采样峰 95292 MiB 不代表瞬时峰 |
| non-thinking + single_tool_v1 | 固定诊断 320/320 轨迹、6619 请求完整且零非法/截断，但全无买卖；同窗 siblings Y 相同，plain/pooled residual 全零 |
| thinking 对照 | 单工具指令仍 17/18 合格，一次工具 JSON 中途触及 cap；未启动条件式扩展，不能说 thinking 已修复 |
| R5 工程验收 | 四项 1.7B wiring + 两项 4B pilot，64 条训练/12 条评估、1984 请求全完整；actor/critic 各 8 次更新，仍无成功买卖/持仓；M2 非零 S 为 0/320 |

非空训练 raw_prompt 原先绕过默认 system 指令，`single_tool_v1` 补齐“每轮一个工具”合同，未指定交易方向。
R5 的现金净收益为零，但两窗 Y 为 −4.123584/−4.690993；现金不等于零训练分数。1.7B 的 131 次 invalid 缺完整原文，不能事后准确分类。
step-8 actor 与截断时 step-11 权重不同，独立 vLLM 调度也不同，固定诊断不是精确故障重放；本阶段无正式 Valid/Final OOS 效果评估。

## 行为诊断：读懂事实仍不等于形成交易

完整工具 schema、真实请求/错误日志和有界 `history_memory_v1` 已补齐：最多 20 条 compute/retrieve 结果、检索最多 2 条、4096 字节，
保存 as_of 并隔离 siblings/reset。可计算 close 数量与真实 PIT 一致，未用未来价格补齐；新增历史说明却与 1.7B 工具循环相关，未直接升为默认。
买卖已通过真实解析、环境和 next-open 成交验证；未发现写死 hold、4B seed 碰撞或缓存重复回答的证据。

| 诊断 | 结果与解释 |
|---|---|
| schema/首步 | 48/48 响应完整；4B 旧/新 schema 和追加中性目标各 8/8 hold；修接口没有直接修行为 |
| 完整观察/因素 | 32 条中 25 完整；2593 次响应均正常 stop。因素诊断 96 条中 70 终局、26 耗尽请求，94/96 无持仓；失败主要是工具循环而非 token 截断 |
| 1.7B 语义 | 64 条中 memory_only 15/16 完整，availability 5/16，time_semantics/组合均 0/16；987 次合法 compute 重复同日同参数 |
| 4B 事实理解 | 64 次答题、448/448 事实正确；答案只评分，不执行交易，不是 64 条交易轨迹 |
| 4B 定义/目标转移 | 64/64 完整，只有原 features 对照一笔 AXP 成交；新增定义/目标的 48 条全现金；AXP 净收益 0.291835%、benchmark 3.882613%、Y −4.340712 |
| 4B 单步覆盖 | 1024/1024 完整，23 笔 buy 成交、1 次空仓 sell 被拒；20 笔是同一窗口 AMZN 50%，其余 V 两笔/GS 一笔，覆盖少且集中 |

1024 次是逐次重置后各执行一步，非完整轨迹；相同 AMZN 也非独立市场案例。30% 集中度是评分软惩罚，50% 合法不等于风险合理。
hold/abstain 都保持仓位并推进日期，文本比例不是持仓改善。六种脚本策略×两窗的 240 次推进、20 笔成交、180 项终局字段重算通过：
未发现账务/奖励强制现金占优，但脚本收益不能解释 fresh 模型选择。模型给出的“信息不足”也不是经验证的内部因果解释。

## 概率/decode 排查的收束

固定 8 条、234 tokens，阈值始终为逐 token 差≤0.1、整条 logprob 和差≤0.5：原 vLLM teacher-forced 对历史生成 **7/8** 通过，
最大差 0.194178；批量/逐条/逐前缀评分一致。串行 vLLM 为 6/8，HF BF16 完整/KV 为 2/8，HF FP32 为 8/8（最大差约 0.000029445）。
FP32 对旧 vLLM 仍仅 7/8，不能替换旧分布；组合开关试验也不能归因于单一优化。
实际 vLLM 原始概率与公开返回 **234/234 完全一致**，含 226 个 decode tokens；未发现公开概率读取错误，亦无证据把大量 hold 归因于 vLLM。
同 worker 新配对和异进程恢复评分先后因字段接口错误 **INCOMPLETE**；修正通过 CPU 接口检查，未再次 GPU 验收。
新生成与历史 token 相同只能复用旧评分作参考，不能补成新校准通过。未改 vLLM 默认实现、未执行交易，停止扩展底层排查，转向实际生成与初始化。

## 证据入口与保留范围

| 事项 | 原始证据（`runs/` 下） |
|---|---|
| G1 / M1 | `20260831T013112864440Z-smoke-a1-qwen3-1p7b-2d48bf6d/`；`20260902T070540291149Z-smoke-a3-m1-qwen3-1p7b-fb07e727/` |
| screening 失败与 R4/R5 | `m2-screening-20260906-v4/`、`m2-screening-20260906-v4-pilot/`、`m2-screening-20260907-v5/`、同日 `v6/`、`v7/`、`v9/`；`m2-screening-20260908-v10/diagnostics/`、`m2-screening-20260908-v11/engineering_final_audit.json` |
| 工具/观察/奖励 | `m2-hold-diagnostic-20260909/`；`m2-decision-diagnostic-20260909/{probe-v1/final_readout.json,reward-v1/result.json}` |
| 语义/转移/单步覆盖 | `m2-semantics-diagnostic-20260912/`、`m2-transfer-diagnostic-20260912/`、`m2-action-support-diagnostic-20260912/` 下的 `final_readout.json` |
| 概率与未完成配对 | `m2-logprob-alignment-20260912/final_readout.json`；`m2-decode-paired-20260912-v3/paired/`；`m2-decode-recovery-20260912/final_readout.json` |

旧队列已禁用；后续独立 systemd user service/文件日志不依赖 SSH，但未注入真实断网测试。9/8 清理约 14.53 GiB、9/9 扩容后的容量仅属历史记录。
成功 wiring/pilot 的重型 checkpoint 有些已按当时策略退休，仅留 marker/inventory；失败与未完成记录保留，不能把历史路径存在当作当前可恢复模型。
