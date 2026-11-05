# V2第三步：固定目标阶段MC条件性对照

状态：`COMPLETE`；阶段：`None`。

本轮是Train正控上的有限学习诊断；不宣称独立方法复现或测试集泛化。
| 模型/区域 | 成功/计划 | 指标正确 | 新生成token |
|---|---:|---:|---:|
| mc_target_train_panel | 40/64 | 40 | 1708376 |
| rloo_train_panel | 42/64 | 42 | 1709169 |

| 臂 | 完成批次 | 有效更新 | MC pair | 实际生成token |
|---|---:|---:|---:|---:|
| rloo | 3 | 12 | 0 | 2531290 |
| mc_target | 3 | 12 | 14 | 3023694 |

全轮新增生成：36770753 token；累计GPU阶段10121.1秒（含加载）。
普通RLOO和MC的辅助/父采样、有效更新不同，预算上限相同不等于等FLOPs。
新Test与旧市场Final OOS均未访问。

[状态](state.json) · [固定规则](../step3_protocol.json) · [完整读数](summary/readout.json)
