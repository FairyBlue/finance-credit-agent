# 强SFT后的局部credit验证

状态：`COMPLETE`；阶段：`COMPLETE`。

所有候选从同一个完整教师SFT出发，任务、工具、终局reward不变。
父轨迹采样与信息干预探测本身不构成step-credit学习收益。

## 局部信号探测

父轨迹生成token：3308883。
配对续写生成token：1314181。
退役失败配对生成token（仍计同一200万上限）：57188。
信号出口：`FAIL`。

局部信号通过后才进入ordinary RL / sham / 真实credit / 打乱credit对照；
失败或预算下无法判断时停止，不为非零信号反复筛题或修改干预。

## 固定终点

| 模型 | 区域 | 状态 | 完整成功 | 指标/触发/执行正确 | 新增生成token |
|---|---|---|---:|---|---:|
| sft | dev | NOT_RUN_AFTER_SIGNAL_GATE | None / None | None/None/None | 0 |
| rloo_seed42 | dev | SKIPPED | None / None | None/None/None | 0 |
| sham_seed42 | dev | SKIPPED | None / None | None/None/None | 0 |
| m3_seed42 | dev | SKIPPED | None / None | None/None/None | 0 |
| shuffled_seed42 | dev | SKIPPED | None / None | None/None/None | 0 |
| sft | test | NOT_OPEN_AFTER_SIGNAL_GATE | None / None | None/None/None | 0 |
| rloo_seed42 | test | NOT_OPEN_AFTER_SIGNAL_GATE | None / None | None/None/None | 0 |
| sham_seed42 | test | NOT_OPEN_AFTER_SIGNAL_GATE | None / None | None/None/None | 0 |
| m3_seed42 | test | NOT_OPEN_AFTER_SIGNAL_GATE | None / None | None/None/None | 0 |
| shuffled_seed42 | test | NOT_OPEN_AFTER_SIGNAL_GATE | None / None | None/None/None | 0 |

前一轮SFT历史参考：开发82/100、旧Test72/100；不是本轮新保留测试结果。

## 训练与决定

| 臂 | 状态 | 实际生成token | 有效更新 |
|---|---|---:|---:|
| rloo | SKIPPED | NA | NA |
| sham | SKIPPED | NA | NA |
| m3 | SKIPPED | NA | NA |
| shuffled | SKIPPED | NA | NA |

最终决定：`LOCAL_EFFECT_OBSERVED_BUT_PRESET_TRAINING_GATE_NOT_MET`。

所有父/辅助输入输出记账；额外过程真值不进入模型提示。实际消耗不同不称严格等算力。
若通过gate，四臂首批复用同一强SFT的32×4父轨迹；共享生成只记一次全局成本，
各臂数据暴露单列，不将复用说成免费计算或独立复验。每臂新采样上限275万，
父probe上限500万、配对probe上限200万，含四臂的新训练生成总上限1800万。
实际共享probe＋新臂生成：4680252；新增评估生成（另列）：0。
SFT Dev若精确复用上一轮同权重同任务确定性评估，新增成本0、来源明确。
新Test仅在所有终点与训练选择冻结后开放；旧Test保持历史结果。

[状态](state.json) · [完整读数](summary/readout.json) · [决定](decision.json)
