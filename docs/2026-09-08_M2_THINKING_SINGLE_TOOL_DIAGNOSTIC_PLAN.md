# M2 保留 thinking 的单工具诊断（G3，2026-09-08）

本文在 G3 新模型请求执行前登记。当前正式配置仍为 R4；本次为独立 Train-only 诊断。
它接续 [G2 单工具 + non-thinking 诊断](2026-09-08_M2_SINGLE_TOOL_DIAGNOSTIC_PLAN.md)，
不修改 G1/G2 的结果，也不把已通过的完整性门禁改为失败。

## 为什么继续检查另一个候选

G2 的 18 次固定请求全部通过，base 和 step-8 actor 的各 160 条轨迹及最终证据复核已完成。
两模型分别产生 3309 / 3310 次请求，全部自然 EOS、非法动作为 0，但 buy/sell 均为 0，
只有 hold、abstain 和 compute。其完成性证据尚未覆盖真实买卖及持仓路径。

作为历史背景，原 R4 主 A2 的 88 条完整训练轨迹都有买卖，合计 buy 131、sell 36。
旧数据跨多个 prompt 和持续更新的 policy，不能与 G2 frozen-base 检查作同分布因果对照。
这里没有根据交易回报挑参数，也不新增买卖次数、diversity 比例或收益硬门槛；主计划中
M2 每 seed 的 residual exposure 判据继续保留。

G3 要回答的是：**补齐相同工具接口说明，同时保留原 thinking 模式，是否能维持完整交互，
以及它会产生哪些实际动作。** 在证据更充分前，暂不把 G2 non-thinking 设置为正式默认值。

## 候选、输入及固定请求

G3 使用同一个生产 `single_tool_v1` helper，文件 SHA256
`299e0d197142ab2b8e169cb7a3d98dc8eb91b223a08e34fb27af884b95b7c20c`，规则 SHA256
`3ca4fc62f06f96c03bbe704103dea741ff74ee5a626b3b6aebd849cd3eee6b45`。
唯一新变量是明确 `enable_thinking=True`，恢复原 thinking 模式；不修改指令文本、工具
schema/执行器、数据、scorer、学习率或生成上限。用真实 tokenizer 与保留字段顺序的
messages/tools 构造 prompt，保存后重读核对全部 token IDs，启动前冻结其 SHA。
实际 CPU tokenizer 已确认输入为 2360 tokens，token SHA256
`bc4aef8b652f51e0eb62cd2ae31ba3371d3f7b936f44ae24d16e81a53f0a2e22`；
末尾为 `<|im_start|>assistant\n`，没有预先打开 `<think>`，故下述前缀结构检查针对完整 completion。

模型仍为 exact Qwen3-4B base `1cfa9a7208912126459214e8b04321603b3df60c` 和已逐值验证的
step-8 actor BF16 导出。它不是原失败时的 step-11 policy，不承诺精确重现原失败。
旧 G1/G2 单提示、G2 轨迹结果作为已有对照保留；不重新抽取这些对照。

每模型固定先一批 8 个采样请求，seed 依序为
`[4299994105846001998,42,43,44,45,46,47,48]`，再 1 个 greedy（seed42），共 **新增 18 次**。
采样 temperature=1，greedy=0；top_p=1、top_k=-1、repetition_penalty=1、n=1。
各模型使用 fresh vLLM 0.9.2 / V1 / BF16，GPU utilization=.20、max_num_seqs=8、
graph sizes=[1,2,4,8]、max_num_batched_tokens=8192；prompt limit=4096、response cap=4096、
context=8192、ignore_eos=False。没有补抽、更换 seed 或参数更新。

## 固定提示验收：18/18

每响应须原生 `finish_reason=stop`，最后 token 为有效 EOS，长度严格小于 4096；完整
decoded response 中，按真实环境使用的工具解析规则恰有一个合格调用及真实参数 schema。
不能先删掉 thinking 中多余的工具调用，再把原环境会拒绝的响应判为通过。

允许至多一个完整、非嵌套的原生 `<think>…</think>` 前缀；存在该前缀时必须闭合，其外
只允许单工具调用与空白。没有 thinking 段而直接产生有效工具调用也可。未闭合/nested
thinking、工具标签外其他叙述、格式错误或未知终止均不通过。这一模式的“无额外文字”
指标明确指 **thinking 之外**，与 G2 完全无叙述的指标分开记录。

逐请求保存完整原始 token/text、两种解码文本、prompt/response SHA、有效 EOS、raw
finish/stop reason、seed、工具解析及 thinking 结构；记录原始信息，不替模型补闭合标签。
只有全部 18/18 通过才进入下一节。若不通过，保留全部预定请求结果，条件扩展不启动。

### GPU 启动前冻结

新目录下 `run_single_tool_thinking_probe.py` 的 SHA256 为
`90aa0baf4212fdeb8854ba5c8058ae67a25261da8cd72829a7fc79054e1dba7b`，
`prompt.json` 文件 SHA256 为
`8891a9138ea0c593ed43cc9bbe890a2bd5d7b0be9237207577f0001683eb8641`。
39 项 CPU 测试及 Ruff 通过；保存后重新分词与 2360 个预期 IDs 完全一致。
启动还需实际旧 G2 receipt/新 prompt/生产方法链的 CPU 复核通过。启动命令、环境、
commit、文件 SHA 和服务属性落盘；本段登记先于任何新 G3 模型请求。

## 条件式完整 Train 轨迹：预定最多 320 条

G3 采用自己的新 gate，不借用 G2 的通过结果。新 18 请求通过后，base 先、actor 后，各跑
原 32 个 screen-train 窗口 ×（4 条采样 + 1 条 greedy），160/model、320 条上限。
每模型先全部采样，再全部 greedy；采样相邻两窗口 ×4 为 batch8，greedy 相邻两窗口为 batch2。
当前 batch 完成前不补位。任一响应/轨迹不完整即停止整个候选；base 失败则 actor 不启动。

沿用 `finance-rollout-seed-v1`、run seed42、诊断 identity step0、原 sample_index0..31、
rollout_n/validate 与实际 env_step。使用真实原环境与 PIT 数据、固定单工具 helper、
`enable_thinking=True` 及上节采样参数；最多256 agent steps，输入/输出/context预算不变。

输入限定已冻结 `artifacts/m2_screening/train.parquet`（SHA256
`10c4265418681421d572a30e529f00c957086ddadaa4657f62ec83f594ff78fa`）。原环境 source 为
`a549c6d3df4aab9cb38cc503b14a8f27b28bbcae`、Agent-R1 为
`b124aa46534cbf2fb8bc8af11405774984c42ac7`；导出、helper、prompt、runner、gate 与数据均
绑定 SHA。使用真实 decoded response 调用原 `env.step`；自然结束但格式错误的 action
按原 invalid 反馈继续，不能剥离 thinking 中的额外调用来修改环境实际收到的动作。

终局仍须21个ledger sessions、20次合法推进，decision sessions恰为0..19。已生成同批
输出先全部保存，随后才检查/应用，保留失败责任、是否执行及未运行网格。没有合成 hold、
补样、恢复旧 checkpoint、额外优化或根据收益选择配置。

单独报告完成性、token长度、invalid、buy/sell/hold/abstain/compute/retrieve分布及实际
交易覆盖，不把这些动作统计变成新增硬门槛。两模型各160全部完成只是后续新执行修订的
工程依据；它仍不替代真实更新及同worker训练后评估，也不证明原截断从此不会出现。

## 持久运行与后续

产物目录 `runs/m2-screening-20260908-v10/diagnostics/single_tool_thinking_v1/`。服务器
独立 systemd 服务运行，基于完整 base160 证据才自动启动一次 actor，失败停止、日志与
退出码落盘；不依赖本机 SSH，不自动启动训练或 Dev。

当前磁盘约99GiB，生产新cell要求至少112GiB。用户申请额外100GB仍待反馈；本次诊断
不降低该门槛，不删除原FAILED证据。完成上述检查后再决定共同生成修订，并按原计划从
相同base重新做四臂1.7B wiring及4B M1/M2 pilot。正式screening还须门禁及资源检查通过。
