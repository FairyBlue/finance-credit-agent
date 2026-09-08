# M2 生成完整性诊断（2026-09-08，Train-only）

用户授权继续修复响应截断。优先诊断阻塞主 screening 的 4B；1.7B 可作通用实现检查，不能
代替 4B 验收。本文在下列新推理请求执行前登记；不覆盖原失败，也不把诊断请求当作完整轨迹。

## 已确认的边界

v9 四臂 1.7B wiring 均已通过，4B M1 pilot 于 2026-09-07 16:54 UTC 通过；v7 M2 pilot
在保留原身份、核对 runtime/config 相同后复用。主 screening 于 16:54 UTC 开始，首个
`screening-a2-s42` 于 18:54 UTC 失败，controller STOPPED，其余主 cells 未开始。

该 cell 完成 88 条 Train 轨迹、actor/critic 各 11 次实际更新；第 12 轮的
`formal-train-0089 / sample_index=22 / rollout_n=3` 首个请求失败。prompt 2227 tokens，
SHA256 `8f4911a0dc45ec369a1e0527cf0347f47ce97648372d3fbcb7f8bc5bbe98d805`；response 4096
tokens，含开头 `<think>`，无结束 think、工具调用或有效 EOS。cap guard 正确拒绝。
旧 1.7B M2 的 post-update greedy 失败与此有相同完整性表现，但尚无相同根因证据。

当前最新完整 4B checkpoint 是 **step 8**，不是失败请求使用的 step 11 policy；因此不能
声称用其精确复现失败权重，也不能用对照差异定量归因第 9–11 次更新。原 FAILED summary、
原始证据及完整 checkpoint 均保留。此阶段不恢复训练、不进入 screen-dev/正式 Valid/Final OOS。

## 固定请求对照

只使用该失败初始状态，从原 Train 数据重建 messages/tools 和 tokenizer 输入，先验证上述
prompt SHA。模型为 exact Qwen3-4B base `1cfa9a7208912126459214e8b04321603b3df60c`，以及
保留的 step-8 actor 离线 BF16 导出。导出须验证键、形状和 FP32→BF16 全数值一致，不删除源文件。

每个模型比较两种模板：原 `apply_chat_template_kwargs={}`，以及仅改
`enable_thinking=False`。该开关是 Qwen 官方支持的模式；这里只把它作为一个候选变量，
不先修改正式实验配置，不根据交易回报选择候选。

- 每种模板 8 个采样请求，seed 固定为
  `[4299994105846001998, 42, 43, 44, 45, 46, 47, 48]`，按此顺序并发一批；
  temperature=1、top_p=1、top_k=-1、repetition_penalty=1、n=1。
- 每种模板另 1 个 greedy 请求，seed=42、temperature=0，其余相同。
- 固定顺序：原模板采样、原模板 greedy、候选模板采样、候选模板 greedy。
  共 **2 models × 2 templates × 9 = 36 requests**，不补抽、不挑选成功 seed。
- 单请求上限仍 4096，prompt 上限 4096，context 8192，ignore_eos=False。独立 vLLM 0.9.2
  V1 / BF16 / GPU reservation 0.20 / max_num_seqs=8 / graph sizes [1,2,4,8]；离线运行。

记录每个请求的完整 tokens/text、prompt SHA、有效 EOS、原生 finish_reason/stop_reason、
输出长度、工具调用数量及参数格式。主要看完整性和工具格式，不计算 Dev 指标或策略优越性。
任何 length/未知终止保留为失败；不能以 stop 字段或 EOS 之外的文本猜测完成。

独立推理不保留训练 worker/Adam 状态，并发构成也与原两窗口各 n4 的训练 batch 不同；
即使使用原 seed，仍不承诺逐 token 重放。单 prompt 对照不能估计整个 schedule 的失败率。
如果原设置没有复现，也应如实报告；候选仅在所有预定请求均完成且输出工具格式合格时才进入
下一阶段更广的 Train 轨迹检查。若失败，保留全部结果并根据证据另行登记下一步。

## 后续验收原则

若诊断支持更改生成模式，须对四臂的训练、最终评测及 zero-shot 采用同一明确模式，记录
新的共同 execution amendment、输入模板和预算检查；学习率、scorer、residual、样本/seed
预算不因某个臂失败单独改变。必要的新 wiring/pilot 从相同 base 开始，旧失败不改名为通过。
主 screening 只能在必要门禁通过后以新身份重新开始。不会把单请求成功当成问题完全解决。

诊断使用 H100 的独立 systemd 服务，stdin=null、Linger=yes、文件日志；失败停止并保留证据。
产物目录 `runs/m2-screening-20260908-v10/diagnostics/`。生产归档及原 checkpoint 清理策略不在
本次诊断中放宽。若 CPU 导出资源不足，使用经验证的有界诊断导出方式，或报告阻塞。

官方依据：[Qwen3-4B 模型说明](https://huggingface.co/Qwen/Qwen3-4B) 提供
`enable_thinking=False`，并提醒 thinking 模式 greedy 可能重复。该说明不证明本项目失败原因；
项目结论仍取决于本地完整证据及上述对照。
