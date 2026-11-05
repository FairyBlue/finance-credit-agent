# 有限预算MC状态baseline与分配：实际读数

状态：`COMPLETE`；阶段：`None`。

所有训练臂从同一完整SFT开始；MC只替换所选动作前的状态baseline。
全部真实父轨迹请求保留更新，辅助轨迹不作为额外SFT标签。
这是MC适配和分配原型，不是完整VinePPO复现或显式动作Q估计。

## 第一步接线

完整独立MC baseline：8。
恢复前/后Adam有效步数：4 / 8。
接线父数据复用旧128条强SFT轨迹，新增父生成0；原3308883 token作为历史暴露单列。
其选点清单是仅依据历史动作前缀回放生成，并非当时采样前已记录；
第二步的新父采样才在模型生成当前动作之前记录选点。

## 固定终点

| 模型 | 区域 | 状态 | 成功/计划 | 完成 | 指标/触发/执行 | 新增生成token |
|---|---|---|---:|---:|---|---:|
| sft | dev | COMPLETE | 82/100 | 100 | 82/92/96 | 0 |
| rloo_seed42 | dev | COMPLETE | 80/100 | 100 | 82/92/94 | 2559158 |
| mc_uniform_seed42 | dev | COMPLETE | 82/100 | 100 | 82/92/96 | 2559405 |
| mc_priority_seed42 | dev | COMPLETE | 81/100 | 100 | 82/92/95 | 2559294 |
| sft | test | NOT_OPEN_BEFORE_CONFIRMATION_AND_FREEZE | None/None | None | None/None/None | 0 |
| rloo_seed42 | test | NOT_OPEN_BEFORE_CONFIRMATION_AND_FREEZE | None/None | None | None/None/None | 0 |
| mc_uniform_seed42 | test | NOT_OPEN_BEFORE_CONFIRMATION_AND_FREEZE | None/None | None | None/None/None | 0 |
| mc_priority_seed42 | test | NOT_OPEN_BEFORE_CONFIRMATION_AND_FREEZE | None/None | None | None/None/None | 0 |

## 训练及生成账目

| 臂 | 状态 | 父token | 辅助token | 总token | 有效更新 |
|---|---|---:|---:|---:|---:|
| rloo_seed42 | COMPLETE | 2419532 | 0 | 2419532 | 20 |
| mc_uniform_seed42 | COMPLETE | 1902639 | 502739 | 2405378 | 13 |
| mc_priority_seed42 | COMPLETE | 1896636 | 496860 | 2393496 | 13 |

新增接线/训练/评估token：282676 / 7218406 / 7677857；总计15178939，失败请求保留计费。
记录的GPU阶段墙钟：5841.9秒，含加载，非纯忙碌时间。
相同预算上限不代表实际等token或等FLOPs；优势估计更稳定不自动代表学习更好。

seed42工程出口：`NO_CLEAR_LEARNING_GAIN`；独立seed确认：`NOT_STARTED`。
最终决定：`NO_CLEAR_SEED42_LEARNING_GAIN`。

[实际状态](state.json) · [完整读数](summary/readout.json) · [决定](decision.json)
