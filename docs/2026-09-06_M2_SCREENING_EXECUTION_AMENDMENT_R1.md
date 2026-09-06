# M2 screening 执行修订 R1（2026-09-06）

本记录由执行 agent 根据已获授权的实施任务及计划 §5.2 起草、执行；不是新增的人类签字。
该条款允许先更新共同设计和预算、重做必要 pilot，再进入 dev 评测。本次调整共同生成限制、
资源设置及工程 pilot 更新次数，不改变 M2 定义、训练目标、数据选择、seeds 或完成性标准，
供合作者次日连同失败证据复核。

## 触发证据

原执行 revision 0 的 `runs/m2-screening-20260906-v3/wiring-a2-s42`，代码
`6be14b424db6fa45cfb09f893ce1199b67624b34`：

- 新 scorer/n=4 的 8 条 Train 轨迹全部完整，共 479 agent steps；actor、critic 各实际更新一次。
- 更新后的 Train greedy evaluator 中，`formal-train-0005` 第 22 次请求使用 2259 prompt tokens，
  输出恰好达到 2048-token cap，episode 尚未终止，严格门禁判 `INCOMPLETE`。
- 上游把终止原因折叠为 `stop_reason=completed`，无法排除 length truncation；不能因可解析或
  字符串标记而接受该请求。新版本继续拒绝达到 cap 的响应，不放宽完成性。
- 无新增 OOM/memory.high 事件；GPU 的 10 秒采样峰约 51.58 GiB，Torch allocated 峰约 62.37 GiB，
  主机 cgroup 采样峰约 54.29 GiB。allocator 指标与物理显存采样是不同量，4B 容量仍需实测。
- 失败发生在 1.7B wiring 的 Train 输入上；尚未开始主 screening 或 screen-dev evaluator。
  final checkpoint 尚未写出，不能把已更新模型当成可恢复的完整 checkpoint。

## R1 的共同设置与预算

| 设置 | 原起点 | R1（所有臂、seeds、zero-shot 共用） |
|---|---:|---:|
| prompt tokens / request | 4096 | 4096 |
| response tokens / request | 2048 | 4096 |
| 入口强制 context tokens / request | 6144 | 8192 |
| agent steps / trajectory | 256 | 256 |
| siblings / train occurrence | 4 | 4 |
| 主训练 / final dev / zero-shot trajectories | 1024 / 96 / 12 | 1024 / 96 / 12 |
| 最坏主训练 response-token 上界 | 536,870,912 | 1,073,741,824 |
| 最坏 dev + zero-shot response-token 上界 | 56,623,104 | 113,246,208 |

上界由 trajectories × 256 × response cap 得到，只是预算边界，不是预计实际生成量或费用。
实际时间与 token 成本继续按完整 4B pilot 记录；不得宣称原预算、固定 trajectories 等于固定算力。
原上游把 engine capacity 重设为模型的 40960；R1 用其最后合并的 `engine_kwargs` 显式覆盖为
8192，并设 `max_num_seqs=8` 对应最大并发八轨迹，避免为默认 512 并发准备 CUDA graphs。
Flow 同时逐请求强制上表边界；实际 engine CLI 要在运行日志中复核。新增 optimizer 前后的
pinned host allocator stats，以区分存活 offload 状态与缓存，辅助 4B 容量诊断。

AdamW 共同改用 `override_optimizer_config={foreach:false,fused:false}`。原 CUDA 默认 foreach
会产生约一份参数大小的额外 tensor-list 临时内存；4B 为约 15 GiB。此调整保留 dtype、lr、betas、
weight decay 和优化公式，但属于实现数值路径变化，因此全部 R1 臂统一重跑，不混入旧模型结果。

4B 工程 pilot 增为同最前两个 Train episodes 的两个 epochs / 两个完整 update rounds，每 round
仍为八条新采样轨迹，最终 post-update Train greedy 仍为两条。两臂合计 pilot train 从 16 条增加为
32 条，post-update eval 仍为 4 条；每 round 保存一次 checkpoint，保留最近一个。1.7B wiring
保持每臂一个 round。原因是首次 critic 更新时 actor Adam 状态尚未创建，单次更新加评测会漏掉
第二次 critic 更新时两套 Adam 同时存在的峰值；该额外 round 是主筛选前的必要资源验证。

## 执行与停止规则

1. 原 revision 0 的三次启动及所有资源/轨迹证据保留。前两次是模型加载前的工程启动失败；
   v3 的 A2 post-update evaluator 是真实完成性失败，维持 INCOMPLETE，不补零、不改判。
2. R1 从相同 exact base model 重新初始化；不拼接旧 siblings、复用已更新权重或替换旧评测行。
3. 首先验证 R1 的 1.7B M2 接线，再补全四臂 wiring；然后验证 4B M1/M2 pilot。
   只有必要验证完整通过才启动 R1 的八个主 cells，全部采用同一设置和冻结代码。
4. 本次只登记一次共同 cap 扩展。若 R1 仍因生成/agent/context 上限不完整，则保留失败并停止
   自动扩大预算，形成讨论所需证据；不以无限加长或重采样追求通过。
5. 资源布局可继续按 §5.2 实测调整并记录，但不把 1.7B 接线结果当成 4B screening。

本次没有执行正式 Valid 或 Final OOS。此修订不授权正式训练或最终测试。
