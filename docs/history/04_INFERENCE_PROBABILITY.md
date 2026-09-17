# 阶段四：概率测量与 decode 排查收束（9 月 12 日）

整理日期：2026-09-14。合并 logprob 对齐、decode V1—V3、恢复评分及导入归因更正；
详细数表、历史计划和失败原文见[归档](../archive/README.md)。

**未发现 vLLM 公开生成概率读取错误，也没有证据证明 vLLM 导致大量 hold。**
精确生成/完整评分配对仍未完成；后续按用户讨论保留现有推理框架，停止扩展底层数值排查，
回到 Train 内共同探索与初始化。这里不把缺失结果补成通过。

## 有效结果

固定四提示×两个 Train session4 输入，各取此前 sibling0 的完整输出，共 8 条、234 tokens。
比较始终使用逐 token 差≤0.1、整条 logprob 和差≤0.5 的原阈值，没有据结果放宽。

| 比较 | 结果 | 能支持什么 |
|---|---|---|
| 原 vLLM teacher-forced 评分 vs 历史生成 | 7/8 通过，最大逐 token 差 0.194178 | 失败位于 abstain 理由中的 data token；不是因位于理由文本就可忽略 |
| 原 vLLM 批量、逐条、逐前缀评分 | 三种评分在全部 234 tokens 上相同 | 拆分 prompt 评分没有消除差异；逐前缀评分仍是 prefill，不等于 KV decode |
| 串行 vLLM vs 历史生成 | 6/8 通过，最大差 0.451961 | 同时关闭多个优化的组合干预，不能单独归因于某个开关 |
| HF BF16 完整前向 vs KV 前向 | 2/8 通过，最大差 0.451957 | 同一低精度实现的执行路径也可存在明显数值差异 |
| HF FP32 完整前向 vs KV 前向 | 8/8 通过，最大差约 0.000029445 | 支持精度与执行路径相互作用；FP32 对旧 vLLM 仍仅 7/8，不是对旧分布的替代修复 |
| 实际 vLLM 原始概率 vs 公开返回值 | 234/234 一致，最大差 0；含 8 个首 token 和 226 个 KV decode token | 本批原始概率观测正确；没有拿错目标 token 或受约束后的概率 |

decode 观察通过公开 worker 扩展、字符串 RPC 和项目只读回调完成。只在 logits 副本上
计算概率，不改实际 token、输出或随机数状态；已安装 vLLM 源码哈希未变。观察会增加同步开销，
不承诺与旧运行调度完全一致，也未覆盖真正进入训练 loss 的概率比值/KL 全链路。

## 保留的工程失败与更正

| 阶段 | 实际发生的事 | 最终状态 |
|---|---|---|
| V1 | 顶层 scripts 导入冲突，创建 LLM 前停止，0 模型请求 | 冲突实际来自已安装 **verl** 发布的 scripts 包；早期归因 Agent-R1 有误 |
| V2 | 模型加载后 msgpack 拒绝函数对象，0 模型请求 | 改用公开命名 worker 扩展和字符串方法 |
| V3 同 worker 配对 | 8 条自然生成/原始观察已保存，后处理误读 finish_reason，正确字段为 raw_finish_reason | **INCOMPLETE**，完整评分尚未提交；已退出 worker 不能假称恢复了同 worker 配对 |
| 异进程恢复评分 | 已取得 8 个返回对象，但 suffix_indices / selected_suffix_indices 接口不符，数值未落盘 | **INCOMPLETE**，不提供新的概率结果 |

两个序列化字段后来已修正，使用实际共享序列化器的完整 CPU 接口检查通过，
但修正入口当时未再次 GPU 验收。单元检查不能替代缺失的真实配对结果。

8 条新生成恰好与历史 token 序列相同，因此可将新 decode 与**此前已保存**的完整评分做
CPU 参考比较，仍为 7/8、最大差 0.194178。它不是上述失败阶段缺失的新评分，不能宣布
整体校准通过。没有执行环境动作，没有新增交易覆盖或收益读数。

## 对后续工作的影响

保留[阶段三](03_M2_BEHAVIOR_DIAGNOSTICS.md)中 23/1024 单步真实成交等行为证据，
不使用未校准的有限候选文本评分冒充整个 buy/sell 类别概率，也不据此筛选提示。
没有通过修改 vLLM、切换默认 FP32 或延长 cap 来解决交易问题。

后续共同初始化以真实生成、合法执行、持仓和完整轨迹为主要依据；
[9 月 12 日 Train 实用计划](../2026-09-12_M2_TRAIN_PRACTICE_PLAN.md)保留原路径，
其结果及之后回到模型学习的修订见[近期文档](../README.md#近期文档原样保留)。
若未来确需精确配对评分，应对修正入口另做验收，不把这项历史 INCOMPLETE 当作已关闭。

## 证据入口

- `runs/m2-action-support-diagnostic-20260912/calibration_diagnostic.json`：最初 7/8 校准。
- `runs/m2-logprob-alignment-20260912/final_readout.json`：vLLM / HF BF16 / HF FP32 四条件。
- `runs/m2-decode-paired-20260912-v3/paired/`：真实生成及原始观察。
- `runs/m2-decode-recovery-20260912/final_readout.json`：decode 审计、历史参考与评分失败。
- `scripts/diagnostics/m2_decode_paired_checked.py`：当时修正但未再次 GPU 验收的入口。
