# 组合约束泛化与有效信息选择：执行与结果

状态：`COMPLETE`；当前阶段：`None`。
首版共同SFT作为original，本轮新教师SFT作为sft。所有结果来自真实已记录轨迹；未跑项不是零分。

所有候选已冻结，本表可包含冻结后运行的新保留测试。

## 固定终点结果

| 模型 | 区域 | 状态 | 成功/预定N | 成功率% | 业务正确/证据有效 | 条款完整/可核验N | 漏项仍成功 | 正确工具调用（N） | 正确成交费用（N） |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| original | dev | COMPLETE | 63/200 | 31.500 | 72/103 | 53/198 | 12 | 6.683（63） | 248.500（50） |
| sft | dev | COMPLETE | 194/200 | 97.000 | 194/195 | 194/199 | 0 | 7.345（194） | 295.006（174） |
| rloo_seed42 | dev | COMPLETE | 194/200 | 97.000 | 194/195 | 194/199 | 0 | 7.345（194） | 295.006（174） |
| sham_seed42 | dev | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| m3_seed42 | dev | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| rloo_seed43 | dev | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| sham_seed43 | dev | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| m3_seed43 | dev | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| no_key_document | dev | COMPLETE | 0/200 | 0.000 | 10/0 | NA/NA | NA | NA（NA） | NA（0） |
| privileged_full_information_oracle | dev | COMPLETE | 200/200 | 100.000 | 200/200 | NA/NA | NA | 0.000（200） | 298.276（180） |
| public_rule_retrieval | dev | COMPLETE | 200/200 | 100.000 | 200/200 | NA/NA | NA | 6.800（200） | 298.276（180） |
| original | test | COMPLETE | 55/200 | 27.500 | 64/110 | 51/199 | 5 | 6.509（55） | 178.371（39） |
| sft | test | COMPLETE | 191/200 | 95.500 | 191/192 | 191/196 | 0 | 7.330（191） | 281.201（171） |
| rloo_seed42 | test | COMPLETE | 191/200 | 95.500 | 191/192 | 191/196 | 0 | 7.330（191） | 281.201（171） |
| sham_seed42 | test | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| m3_seed42 | test | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| rloo_seed43 | test | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| sham_seed43 | test | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| m3_seed43 | test | NOT_RUN | NOT_RUN | NA | NA/NA | NA/NA | NA | NA（NA） | NA（NA） |
| no_key_document | test | COMPLETE | 0/200 | 0.000 | 10/0 | NA/NA | NA | NA（NA） | NA（0） |
| privileged_full_information_oracle | test | COMPLETE | 200/200 | 100.000 | 200/200 | NA/NA | NA | 0.000（200） | 283.690（180） |
| public_rule_retrieval | test | COMPLETE | 200/200 | 100.000 | 200/200 | NA/NA | NA | 6.800（200） | 283.690（180） |

条款完整度缺少真实提交约束时标NA；完整任务成功仍要求业务与证据共同通过。费用只对正确且有实际成交指标的任务取均值，正确不可行声明不擅自记成零成本成交。

## 业务敏感配对的实际表现

只在离线参考确认单条内容变化确实改变业务的配对上统计。两版本各自按原独立验证器判定，不要求交易向量复制唯一oracle。

| 模型 | 区域 | 敏感配对N | 全对/仅一对/都错 | 业务两题均正确 | 条款完整/可核验/预定轨迹 | 漏项仍业务正确 |
|---|---|---:|---:|---:|---:|---:|
| original | dev | 76 | 16/4/56 | 19 | 33/150/152 | 10 |
| original | test | 78 | 15/5/58 | 19 | 31/155/156 | 13 |
| rloo_seed42 | dev | 76 | 73/2/1 | 73 | 148/151/152 | 0 |
| rloo_seed42 | test | 78 | 73/1/4 | 73 | 147/152/156 | 0 |
| sft | dev | 76 | 73/2/1 | 73 | 148/151/152 | 0 |
| sft | test | 78 | 73/1/4 | 73 | 147/152/156 | 0 |

## 实际RL预算与有效更新

| 训练臂 | 状态 | 实际总token | 父/辅助token | 上限/尾余 | 有效更新 |
|---|---|---:|---:|---:|---:|
| rloo_seed42 | COMPLETE | 1275152 | 1275152/0 | 3000000/1724848 | 0 |
| sham_seed42 | NOT_RUN | NA | NA | NA | NA |
| m3_seed42 | NOT_RUN | NA | NA | NA | NA |
| rloo_seed43 | NOT_RUN | NA | NA | NA | NA |
| sham_seed43 | NOT_RUN | NA | NA | NA | NA |
| m3_seed43 | NOT_RUN | NA | NA | NA | NA |

已运行训练臂RL实际输入＋输出生成token：**1275152**。评估新增生成token：16374487；起点探测生成token：1221353。SFT监督使用单列，不能把RL预算称为项目总成本。
预算上限最多每臂每seed300万、六臂1800万；完整回合准入造成的实际尾余分别报告。相同上限不等于实际等token或等FLOPs；所有辅助续写均计入其臂预算，辅助结果不作为教师。

## 推进与出口

第三步模式：`FROZEN_SFT_BASELINE_ONLY`。第二seed决策：`SKIPPED`，触发：`NO_RL_SIGNAL`。

最终决定：`KEEP_COMPOSITIONAL_SFT_SKIP_ZERO_GRADIENT_RL_AND_M3`。


本轮普通RLOO首批无有效优势，0次更新，按计划结束；sham、M3与第二seed未训练。因此本轮无法比较局部credit机制，不是M3的负面训练结果。零步RLOO的确定性评估复用同权重新SFT轨迹，新增生成token为0，既不重复记采样成本，也不计作新的独立复验或RL学习收益。

[简短执行结论](CONCLUSION.md)

[真实同题失败→成功示例](examples/README.md) · [固定端点错误归类](summary/endpoint_error_analysis.json)

本轮保持模拟委托、真实历史已知报价和实际账本核验；不是未来收益或真实部署证明。局部业务credit与纯引用变化分开，不将删文档的全部损失称作金融决策贡献。配对版本、改写与跨seed重复不增加独立基础情景数。未以测试选择模型，未触及旧市场Final OOS，也未据此宣称算法首创。

[固定分母明细](summary/table.csv) · [分类结果](summary/per_type.csv) · [配对成功与共同成功成本](summary/paired_success.csv) · [业务敏感配对结果](summary/paired_business_sensitivity.csv) · [实际累计token曲线](summary/development_token_curve.csv)
