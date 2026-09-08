# M2 单工具协议候选诊断（2026-09-08，Train-only）

本文在新候选的 GPU 请求执行前登记。它接续
[生成完整性诊断](2026-09-08_M2_GENERATION_DIAGNOSTIC_PLAN.md)，不修改前一候选的验收结果。
主 screening 仍停止；这是生成协议修复的诊断，尚不是 R5 正式实验执行修订。

## 前一候选的结果及本次依据

原定 36 次请求已全部观察。两模型、两模板均正常 EOS 结束；没有在这一初始状态的独立推理中
复现原来的 4096-token 截断。工具格式按单个完整调用及真实参数 schema 检查：

| 模型 | 原 thinking：完整 / 格式通过 | 仅关闭 thinking：完整 / 格式通过 |
| --- | --- | --- |
| exact 4B base | 9/9；4/9 | 9/9；7/9 |
| step-8 actor | 9/9；6/9 | 9/9；7/9 |

仅关闭 thinking 的候选未达到 18/18 格式门禁；`eligible_for_multi_train_validation=false`。
原计划中以该候选为条件的 320 条轨迹扩展不启动。输出过长的风险与正常结束后的格式不合格
是不同问题；目前不能认定原截断已经修好。

实际输入中有一个明确的接口说明缺口：非空数据 `raw_prompt` 使 `TradingEnv.reset` 跳过
默认 system 指令，模型没有收到该处的“每轮一个工具”要求；Qwen 工具模板允许一个或多个
调用。已观察到无工具的计划文字、一次多个调用，以及 `compute(lookback=1)`。
模型还会把 episode 的 session 0 误认为只存在一个历史 bar；真实 compute 访问的是当前
回放时间以前的收盘价。失败初始状态在 2023 年 5 月，已存在此前历史；这不代表最早的
2023 年 1 月窗口也有足够历史。

## 新候选：single_tool_v1 + non-thinking

只增加明确的共同 system 指令，并使用 `enable_thinking=False`。规则要求每轮只输出一个
`<tool_call>` JSON，然后等待工具结果；不输出计划、评论或未来调用清单；说明 session
编号不限制 PIT 历史、compute lookback 为 2–60、成交在下一开盘且禁止做空/杠杆。
没有指定买卖或持仓方向，不修改工具实现、参数校验、交易环境、奖励、学习率或数据。

生产实现位于 `src/agents/screening_prompt.py`，协议名 `single_tool_v1`；规则文本 SHA256：
`3ca4fc62f06f96c03bbe704103dea741ff74ee5a626b3b6aebd849cd3eee6b45`。
其完整文本及实际 tokenizer 输出写入候选 prompt artifact。生成时只复制并扩展消息，不改变
环境保存的用户/工具消息；四臂、训练、最终评估和 zero-shot 使用同一可选入口。
默认 `legacy` 保持原提示，正式 `configs/m2_screening.yaml` 此时仍为原 R4。

## 固定初始提示验证：新增 18 次请求

沿用已逐 token 重建的失败初始状态、原 tools 和真实参数 schema。旧 prompt artifact 的
SHA256 为 `4f8dbd6ee0bed1041da7a024749e03eac7c26c163cf63845be003a8542754170`。
使用实际生产 helper 和 exact 4B tokenizer 构造新 prompt，启动前保存 helper/规则/prompt
文件 SHA、完整 token IDs 及其 SHA；不得根据新输出再改提示。

启动前冻结的新 prompt 为 2364 tokens，token SHA256
`aa5b8ad01c33807d77fb1e4113d40bd998b3ef50e5b8245f686cc8aff031e4a4`；prompt JSON SHA256
`f7d2ddb52c93ef1a5d4272c05a505b0cac63e61356c5385c0af60bc142d0f879`。
probe 脚本 SHA256 `2a85274ae1d77706067f503c1f6e4f151a8ceaadf76ee5524070c8a8c1316ccd`。
新 probe 的 11 项 CPU 测试通过；共同生产路径相关 196 项测试、Ruff 和三文件 mypy 通过。

- 模型仍为 exact Qwen3-4B base `1cfa9a7208912126459214e8b04321603b3df60c`，及已完成
  399/399 权重键、全部 FP32→BF16 bitwise 校验的 step-8 actor 导出。它不是失败时的 step-11
  policy；原 FAILED checkpoint 和证据保留。
- 每模型新建独立 vLLM engine，先一批 8 个采样请求，再 1 个 greedy 请求，共 18 个。
  采样 seed 按序固定为 `[4299994105846001998,42,43,44,45,46,47,48]`，greedy seed=42。
  temperature 分别为 1 / 0；top_p=1、top_k=-1、repetition_penalty=1、n=1。
- 仍使用 vLLM 0.9.2 / V1 / BF16、GPU utilization=0.20、max_num_seqs=8、graph sizes
  [1,2,4,8]、context=8192、prompt limit=4096、response cap=4096、ignore_eos=False。
- 旧 non-thinking 18 次结果保留为已有对照，不重新抽取。新旧 engine 的批次/缓存状态可能
  不同，不能承诺逐 token 配对，也不能把差异全归因为单条说明。
- 验收要求新候选 **18/18** 响应均原生 `finish_reason=stop`，最后 token 为有效 EOS，
  长度严格小于 4096；恰好一个完整工具调用且通过真实参数 schema；调用标签外不得有
  非空计划/评论文字。最后一项是本候选新增的明确要求，旧对照仍保留原指标定义。
- 每请求记录 raw finish/stop reason、完整 token/text、prompt/response SHA、seed/请求身份、
  工具格式及额外文字。length/未知终止、格式或额外文字失败均保留；不补抽、不更换 seed。

新输出放在 `runs/m2-screening-20260908-v10/diagnostics/single_tool_v1/`。旧两份 matrix 必须
均为 OBSERVED 且旧 acceptance 明确不通过；新 gate 独立聚合新 18 次结果及对应源 SHA。
单个工具的参数格式通过并不等于交易可执行或策略有效；本阶段不计算回报来选配置。

## 条件式完整 Train 轨迹检查：预定最多 320 条

仅在上述新候选 18/18 门禁通过后，才启动本节；不得沿用旧候选的失败 gate。
base 先、step-8 actor 后，每模型按原顺序执行全部 32 个 screen-train 窗口，每窗口 4 条
采样和 1 条 greedy，共 160/model、320 条。参数不更新，不访问 screen-dev/正式 Valid/Final OOS。
输入限定已冻结 `artifacts/m2_screening/train.parquet`，SHA256
`10c4265418681421d572a30e529f00c957086ddadaa4657f62ec83f594ff78fa`。

每模型先完成所有采样，再完成所有 greedy。采样按相邻两个窗口 ×4 siblings 的 batch8；
greedy 按相邻两个窗口的 batch2。当前 batch 全部终局前不补入新轨迹。run seed=42，请求
seed 使用现有 `finance-rollout-seed-v1`，诊断 identity 固定 step=0、原 sample_index 0..31、
rollout_n 和 validate；env_step 每次实际请求递增，不把诊断 identity 当作训练轮次。

使用真实 `TradingEnv.reset/step`、真实 tokenizer 和固定 single_tool_v1 + non-thinking。
环境代码、Train 输入、helper、runner、模型导出和新 acceptance 均绑定版本/哈希；执行前
保存完整网格和启动依据。采样/greedy、4096 cap、8192 context、最多 256 agent steps 及
vLLM 资源参数沿用上节。单 prompt 诊断与此处完整轨迹的用途不同：轨迹中自然结束的
invalid/malformed action 交给原环境反馈，记录后继续，不合成 hold、不跳过错误、不补抽。

每条完整轨迹须到达真实终局：21 个 ledger sessions、20 次合法推进，decision sessions
恰为 0..19。每步记录请求、原始输出、真实动作/invalid 及推进；环境奖励即使计算也不作为
候选门禁。任一响应/轨迹不完整、预算越界或 runtime 异常即停止整个候选；已生成的同批
输出全部保存，明确是否已应用到环境；其余成员标记中止，后续网格标记未开始。没有重试、
补位、续接 checkpoint 或追加成功样本。若 base 阶段失败，actor 阶段不启动。

这 320 条是预定上限。全部通过才支持下一步在新共同执行修订下重跑必要 wiring/pilot；
独立推理没有真实更新和常驻 Adam，不替代同 worker 的训练后评估，也不能证明后续训练
不会再截断。若失败，保留全部证据，另行登记下一步。

## 资源与后续执行边界

诊断通过服务器独立 systemd user service、Linger=yes、stdin=null 和文件日志运行，断开
本机 SSH 不影响服务生命周期；未声称做过物理断网注入测试。现有空间可做此阶段诊断。
用户将联系管理员申请增加 100 GB，结果待通知；不能当作扩容已经完成。

若候选与完整轨迹验证支持修复，再登记共同 R5，四臂从相同 base 重新做必要接线/pilot，
不修改原 R4 失败。主筛选还需通过这些门禁及实际资源检查。原 FAILED 数据和 checkpoint
不删除，不降低生产磁盘/CPU guard 以绕过资源不足。
