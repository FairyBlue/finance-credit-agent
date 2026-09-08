# M2 生成完整性诊断报告（2026-09-08）

**当前结论：关闭 thinking + 单工具指令的候选已完成 320 条轨迹、6619 次请求，没有截断或非法动作；
但没有任何买卖，暂不将它设为正式默认值。下一步单独检查保留 thinking 的相同工具指令。**
原 R4 主筛选仍为 0/8 完整 cells，没有开始 screen-dev 评估。生产配置尚未切换到新候选。
本报告随当前诊断更新；原失败及历史报告保留。

## 原故障与诊断范围

4B A2 / seed42 完成 88 条 Train 轨迹、actor/critic 各 11 次实际更新，第 12 轮的一条训练
响应达到 4096 tokens，未结束 thinking、未输出工具调用，也没有 EOS。原完整性保护正确
停止了该 cell；没有证据表明这是 EOS 检测的假阳性。

本次优先处理阻塞主筛选的 4B。失败初始状态的 2227-token 输入已逐 token 重建并匹配
原 prompt SHA。对照使用 exact 4B base 与保留的 step-8 actor：后者离线导出后完成
399 个权重键、全部 FP32→BF16 数值逐位校验。**step 8 不是失败时更新 11 次后的精确权重**，
独立 vLLM 的批处理也不同于原训练 worker，因此这里不宣称精确重放失败。

## 两个候选的结果

每个模型/设置均为事先固定的 8 个采样 seed 加 1 个 greedy，共 9 次初始提示响应。
输入/输出上限及采样参数保持原约定，不根据交易回报选配置。

| 设置 | base：完整 / 格式合格 | step-8 actor：完整 / 格式合格 | 结论 |
| --- | --- | --- | --- |
| 原默认 thinking | 9/9；4/9 | 9/9；6/9 | 此处未复现截断；有多调用及参数错误 |
| 仅关闭 thinking | 9/9；7/9 | 9/9；7/9 | 未达到 18/18 格式门禁，原条件式轨迹扩展不启动 |
| 关闭 thinking + single_tool_v1 | 9/9；9/9 | 9/9；9/9 | 18/18 同时满足 EOS、单工具、真实参数 schema 及无额外文字 |

原 36 次响应均自然结束，不能据此推断整个训练 schedule 的失败率。仅关闭 thinking 时
仍出现无调用的计划文字及一次多个调用，说明缩短响应本身不足以保证接口正确。

已确认的接口说明缺口是：非空训练 `raw_prompt` 使环境跳过默认 system 指令，实际模型
输入缺少“每轮一个工具”的要求；Qwen 的工具模板允许一个或多个调用。新可选协议补上
这一要求，并说明 session 编号不限制 PIT 历史、compute 的 lookback 为 2–60。
它不指定持仓方向，不更改数据、工具实现、scorer、学习率或训练样本。

新候选 18 次响应最长 34 tokens；两模型各为 8 个 hold 和 1 个 compute。这个初始状态
的行为分布不证明策略有效，参数格式通过也不等于整条交易轨迹可完成。

## 工程验证及完整轨迹检查

共同生产路径相关 196 项测试、Ruff、mypy 通过；实际 prompt 方法链与 tokenizer 得到
预登记的 2364 tokens，四臂使用同一入口，处理消息副本不会累积指令。

首次新 probe 在任何模型请求前因诊断 JSON 排序导致输入不一致而停止，新增响应为 0。
保留旧失败目录后修正保序落盘，并增加保存后重读及启动前 token 核验；13 项 CPU 测试通过。
候选文本、目标 token IDs、seed 和 18 请求预算不变，旧 36 次结果不受影响。

完整轨迹 runner 已通过 40 项 CPU 测试及独立审阅。真实环境 row0/row22 各做一次非法动作
及 20 次固定 hold，终局和 42 个保存 prompt 经最终代码复核通过；这是 CPU 接口测试，
模型请求为 0，不计入生成轨迹预算。

**完整轨迹检查及最终证据复核已完成，controller 为 COMPLETE320，actor 退出码为 0。**
每模型运行原 32 个 Train 窗口 ×（4 条采样 + 1 条 greedy），base 全部完成后才运行 actor。
两模型的 submitted / returned / applied 数量一致，没有响应截断或终局失败。

| 模型 | 完整轨迹 | 请求数 | 响应长度：最短 / 中位 / 最长 | buy / sell | hold / abstain / compute | invalid |
| --- | ---: | ---: | --- | --- | --- | ---: |
| 4B base | 160/160 | 3309 | 15 / 15 / 35 | 0 / 0 | 3102 / 98 / 109 | 0 |
| step-8 actor | 160/160 | 3310 | 15 / 15 / 35 | 0 / 0 | 3116 / 84 / 110 | 0 |

所有轨迹均有 21 个 ledger sessions、20 次合法推进；所有响应自然 EOS，输入最长 2526 tokens。
两模型均没有 retrieve。320 条执行覆盖了空仓路径，尚未覆盖实际买卖与持仓路径。
使用真实环境、PIT 边界、原 seed 推导及 4096/8192/256 上限；任何响应或轨迹不完整就停止
候选，保留同批已生成输出及未执行网格，不补抽、不替换动作、不更新参数。

此阶段不替代新配置下从 base 开始的训练 wiring/pilot，也不访问 screen-dev/正式 Valid/
Final OOS 做评估。详细边界及启动 SHA 见
[单工具协议诊断登记](2026-09-08_M2_SINGLE_TOOL_DIAGNOSTIC_PLAN.md)。

作为历史背景，原 R4 的 88 条完整训练轨迹都有买卖，总计 buy 131、sell 36；它们跨不同
prompt 和持续更新的 policy，因此不能把差异全部归因于关闭 thinking。G2 原先登记的
完成性门禁已经通过，不事后增加交易数量门槛，也不据此裁定正式 M2 的 residual exposure。
不过，当前证据不足以支持立即采用该设置进行 M2。

新 [G3 诊断登记](2026-09-08_M2_THINKING_SINGLE_TOOL_DIAGNOSTIC_PLAN.md) 保持完全相同的
单工具指令，明确启用 thinking。新增固定 18 请求全部通过后，才执行该候选自己的条件式
320 条 Train 检查；旧对照保留，不重新抽样，不使用收益或 Dev 结果选择候选。

## 资源及后续

服务通过 H100 的 systemd user service、Linger、stdin=null 和文件日志运行，生命周期
独立于本机 SSH；未声称做过物理断网注入测试。原 FAILED checkpoint 和诊断证据保留。

当前可用磁盘约 99 GiB，生产 supervisor 在每个新 cell 前要求至少 112 GiB，涵盖 wiring、
pilot、主筛选和 zero-shot。用户将向管理员申请增加 100 GB，结果待通知；不能认为已扩容。
现有空间可完成当前独立推理诊断，后续训练须等待实际空间检查通过。

待 G3 结果及交互覆盖评估后，再决定并登记共同 R5，重新执行必要的四臂 1.7B wiring、4B M1/M2 pilot，
每个新 cell 从对应的相同 base 初始化，并验证同 worker 的训练后 greedy。通过必要门禁后才能以新身份重启
主筛选；旧 R4 失败不会改写为成功。

证据根目录：`runs/m2-screening-20260908-v10/diagnostics/`。原对照详见
`fixed_matrix_readout.json`；新候选见 `single_tool_v1/acceptance_v2/acceptance.json`；
完整 320 条统计见 `single_tool_v1/full_trajectory_readout.json`，历史动作背景见
`single_tool_v1/original_r4_action_context.json`；运行入口和最新服务见
本机 `runs/M2_SCREENING_LATEST.md`。
