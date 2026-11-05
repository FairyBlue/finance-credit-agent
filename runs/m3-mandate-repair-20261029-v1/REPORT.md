# 多工具投资约束修复：执行与结果

**新任务通过SFT明显改善，当前M3配置未建立独立成功率增量。** [简要结论](CONCLUSION.md)。

控制器状态：`COMPLETE`。本报告只读取已记录结果，没有选择checkpoint、任务或窗口。冻结清单：COMPLETE；下表包括冻结后实际运行的test。

任务使用Train历史已知报价、模拟账户与约束文档，回答多工具约束修复是否正确，不把它写成未来交易收益。主采样原生生成；解析、工具、生成及系统失败保留。成功率以事先规定任务槽为固定分母，错误/缺失计失败；运行中读数应与完成状态一起看。

development与test各100个任务分别来自50个基础情景、每个情景2个改写；两个训练seed也复评同一批任务。改写与跨seed重复不增加独立任务数量。[原模型错误分析](baselines/original_error_analysis.json)、[真实交互示例](examples/README.md)。

## 原模型、共同SFT及各臂终点

| 模型 | 区域 | 成功/预定N | 成功率% | 业务可行/证据有效 | 正确费用均值（N） | 正确工具调用均值（N） | 系统/生成失败 | 缺失 | 评估输入/输出token | 复用来源 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m3_seed42 | development | 92/100 | 92.0000 | 74/100 | 221.1570（74） | 6.4565（92） | 0 | 0 | 0/0 | m3_seed42_curve000_dev |
| m3_seed42 | test | 84/100 | 84.0000 | 68/100 | 215.5044（68） | 6.4048（84） | 0 | 0 | 1673159/34549 | — |
| m3_seed43 | development | 92/100 | 92.0000 | 74/100 | 221.1570（74） | 6.4565（92） | 0 | 0 | 0/0 | m3_seed43_curve001_dev |
| m3_seed43 | test | 84/100 | 84.0000 | 68/100 | 215.5044（68） | 6.4048（84） | 0 | 0 | 1673152/34536 | — |
| original | development | 1/100 | 1.0000 | 1/1 | 375.2689（1） | 8.0000（1） | 0 | 0 | 2175393/92715 | — |
| original | test | 2/100 | 2.0000 | 4/2 | 405.2600（2） | 7.0000（2） | 0 | 0 | 2151591/88458 | — |
| rloo_seed42 | development | 92/100 | 92.0000 | 74/100 | 221.1570（74） | 6.4565（92） | 0 | 0 | 0/0 | rloo_seed42_curve000_dev |
| rloo_seed42 | test | 84/100 | 84.0000 | 68/100 | 215.5044（68） | 6.4048（84） | 0 | 0 | 0/0 | sft_test |
| rloo_seed43 | development | 92/100 | 92.0000 | 74/100 | 221.1570（74） | 6.4565（92） | 0 | 0 | 0/0 | rloo_seed42_curve000_dev |
| rloo_seed43 | test | 84/100 | 84.0000 | 68/100 | 215.5044（68） | 6.4048（84） | 0 | 0 | 0/0 | sft_test |
| sft | development | 92/100 | 92.0000 | 74/100 | 221.1570（74） | 6.4565（92） | 0 | 0 | 1672217/34691 | — |
| sft | test | 84/100 | 84.0000 | 68/100 | 215.5044（68） | 6.4048（84） | 0 | 0 | 1673152/34536 | — |
| sham_seed42 | development | 92/100 | 92.0000 | 74/100 | 221.1570（74） | 6.4565（92） | 0 | 0 | 0/0 | rloo_seed42_curve000_dev |
| sham_seed42 | test | 84/100 | 84.0000 | 68/100 | 215.5044（68） | 6.4048（84） | 0 | 0 | 0/0 | sft_test |
| sham_seed43 | development | 92/100 | 92.0000 | 74/100 | 221.1570（74） | 6.4565（92） | 0 | 0 | 0/0 | rloo_seed42_curve000_dev |
| sham_seed43 | test | 84/100 | 84.0000 | 68/100 | 215.5044（68） | 6.4048（84） | 0 | 0 | 0/0 | sft_test |
| privileged_full_information_oracle | development | 100/100 | 100.0000 | 82/100 | 233.9956（82） | 0.0000（100） | 0 | 0 | 0/0 | — |
| public_rule_retrieval | development | 100/100 | 100.0000 | 82/100 | 233.9956（82） | 6.5000（100） | 0 | 0 | 0/0 | — |
| no_key_document | development | 0/100 | 0.0000 | 18/0 | NA（0） | NA（0） | 0 | 0 | 0/0 | — |
| privileged_full_information_oracle | test | 100/100 | 100.0000 | 84/100 | 236.6434（84） | 0.0000（100） | 0 | 0 | 0/0 | — |
| public_rule_retrieval | test | 100/100 | 100.0000 | 84/100 | 236.6434（84） | 6.5000（100） | 0 | 0 | 0/0 | — |
| no_key_document | test | 0/100 | 0.0000 | 16/0 | NA（0） | NA（0） | 0 | 0 | 0/0 | — |

温度0下，同一权重、语义配置、数据及native/eager/seed配置的确定性评估可以复用；复用保持原100任务分母，新增请求与token按0计，原采样成本及来源单列。这不是新的重复实验，也不能把相同策略的缓存评分写成RL训练增益。

费用只在正确且实际有成本指标的任务上取均值，并明确N；正确不可行声明不被擅自当作一笔零成本成交。系统失败与业务错误分别计数，仍同时占据固定成功率分母。

CPU privileged_full_information_oracle使用隐藏完整信息，仅作上限参照；public_rule_retrieval通过公开文档与真实工具取信息，单独列为可运行规则基准。业务可行与证据有效分列，任务总成功要求两者（或正确不可行声明）同时满足。

冻结后的test CPU强基线已并入同表；[CPU汇总](baselines/test_summary.json)、[逐任务真实轨迹](baselines/test_trajectories.jsonl)。

development的no_key_document总成功0/100，其中业务可行18/100、证据有效0/100；总成功为零不代表业务可行数为零。

test的no_key_document总成功0/100，其中业务可行16/100、证据有效0/100；总成功为零不代表业务可行数为零。

## 共同SFT成功的含义

92/100表示最终业务可行与证据通过（包含正确不可行声明），不等于所有条款字段都抽取正确。mixed类16项中仅2项字段全对，另有6个最终成功任务仍有字段差异。本批8个失败都已读到正确证据，却在显式约束中把Financials写成Technology；求解器按传入的错误问题正常工作，问题不在求解器计算。[SFT逐项错误分析](baselines/sft_error_analysis.json)、[更新后的真实交互示例](examples/README.md)。

## 每类固定分母成功数

| 模型 | 区域 | 类别 | 成功/预定N |
|---|---|---|---:|
| m3_seed42 | development | cash | 16/16 |
| m3_seed42 | development | exposure | 16/16 |
| m3_seed42 | development | hold | 18/18 |
| m3_seed42 | development | infeasible | 18/18 |
| m3_seed42 | development | mixed | 8/16 |
| m3_seed42 | development | restriction | 16/16 |
| m3_seed42 | test | cash | 18/18 |
| m3_seed42 | test | exposure | 18/18 |
| m3_seed42 | test | hold | 16/16 |
| m3_seed42 | test | infeasible | 16/16 |
| m3_seed42 | test | mixed | 0/16 |
| m3_seed42 | test | restriction | 16/16 |
| m3_seed43 | development | cash | 16/16 |
| m3_seed43 | development | exposure | 16/16 |
| m3_seed43 | development | hold | 18/18 |
| m3_seed43 | development | infeasible | 18/18 |
| m3_seed43 | development | mixed | 8/16 |
| m3_seed43 | development | restriction | 16/16 |
| m3_seed43 | test | cash | 18/18 |
| m3_seed43 | test | exposure | 18/18 |
| m3_seed43 | test | hold | 16/16 |
| m3_seed43 | test | infeasible | 16/16 |
| m3_seed43 | test | mixed | 0/16 |
| m3_seed43 | test | restriction | 16/16 |
| original | development | cash | 0/16 |
| original | development | exposure | 0/16 |
| original | development | hold | 0/18 |
| original | development | infeasible | 0/18 |
| original | development | mixed | 1/16 |
| original | development | restriction | 0/16 |
| original | test | cash | 0/18 |
| original | test | exposure | 2/18 |
| original | test | hold | 0/16 |
| original | test | infeasible | 0/16 |
| original | test | mixed | 0/16 |
| original | test | restriction | 0/16 |
| rloo_seed42 | development | cash | 16/16 |
| rloo_seed42 | development | exposure | 16/16 |
| rloo_seed42 | development | hold | 18/18 |
| rloo_seed42 | development | infeasible | 18/18 |
| rloo_seed42 | development | mixed | 8/16 |
| rloo_seed42 | development | restriction | 16/16 |
| rloo_seed42 | test | cash | 18/18 |
| rloo_seed42 | test | exposure | 18/18 |
| rloo_seed42 | test | hold | 16/16 |
| rloo_seed42 | test | infeasible | 16/16 |
| rloo_seed42 | test | mixed | 0/16 |
| rloo_seed42 | test | restriction | 16/16 |
| rloo_seed43 | development | cash | 16/16 |
| rloo_seed43 | development | exposure | 16/16 |
| rloo_seed43 | development | hold | 18/18 |
| rloo_seed43 | development | infeasible | 18/18 |
| rloo_seed43 | development | mixed | 8/16 |
| rloo_seed43 | development | restriction | 16/16 |
| rloo_seed43 | test | cash | 18/18 |
| rloo_seed43 | test | exposure | 18/18 |
| rloo_seed43 | test | hold | 16/16 |
| rloo_seed43 | test | infeasible | 16/16 |
| rloo_seed43 | test | mixed | 0/16 |
| rloo_seed43 | test | restriction | 16/16 |
| sft | development | cash | 16/16 |
| sft | development | exposure | 16/16 |
| sft | development | hold | 18/18 |
| sft | development | infeasible | 18/18 |
| sft | development | mixed | 8/16 |
| sft | development | restriction | 16/16 |
| sft | test | cash | 18/18 |
| sft | test | exposure | 18/18 |
| sft | test | hold | 16/16 |
| sft | test | infeasible | 16/16 |
| sft | test | mixed | 0/16 |
| sft | test | restriction | 16/16 |
| sham_seed42 | development | cash | 16/16 |
| sham_seed42 | development | exposure | 16/16 |
| sham_seed42 | development | hold | 18/18 |
| sham_seed42 | development | infeasible | 18/18 |
| sham_seed42 | development | mixed | 8/16 |
| sham_seed42 | development | restriction | 16/16 |
| sham_seed42 | test | cash | 18/18 |
| sham_seed42 | test | exposure | 18/18 |
| sham_seed42 | test | hold | 16/16 |
| sham_seed42 | test | infeasible | 16/16 |
| sham_seed42 | test | mixed | 0/16 |
| sham_seed42 | test | restriction | 16/16 |
| sham_seed43 | development | cash | 16/16 |
| sham_seed43 | development | exposure | 16/16 |
| sham_seed43 | development | hold | 18/18 |
| sham_seed43 | development | infeasible | 18/18 |
| sham_seed43 | development | mixed | 8/16 |
| sham_seed43 | development | restriction | 16/16 |
| sham_seed43 | test | cash | 18/18 |
| sham_seed43 | test | exposure | 18/18 |
| sham_seed43 | test | hold | 16/16 |
| sham_seed43 | test | infeasible | 16/16 |
| sham_seed43 | test | mixed | 0/16 |
| sham_seed43 | test | restriction | 16/16 |
| privileged_full_information_oracle | development | cash | 16/16 |
| privileged_full_information_oracle | development | exposure | 16/16 |
| privileged_full_information_oracle | development | hold | 18/18 |
| privileged_full_information_oracle | development | infeasible | 18/18 |
| privileged_full_information_oracle | development | mixed | 16/16 |
| privileged_full_information_oracle | development | restriction | 16/16 |
| public_rule_retrieval | development | cash | 16/16 |
| public_rule_retrieval | development | exposure | 16/16 |
| public_rule_retrieval | development | hold | 18/18 |
| public_rule_retrieval | development | infeasible | 18/18 |
| public_rule_retrieval | development | mixed | 16/16 |
| public_rule_retrieval | development | restriction | 16/16 |
| no_key_document | development | cash | 0/16 |
| no_key_document | development | exposure | 0/16 |
| no_key_document | development | hold | 0/18 |
| no_key_document | development | infeasible | 0/18 |
| no_key_document | development | mixed | 0/16 |
| no_key_document | development | restriction | 0/16 |
| privileged_full_information_oracle | test | cash | 18/18 |
| privileged_full_information_oracle | test | exposure | 18/18 |
| privileged_full_information_oracle | test | hold | 16/16 |
| privileged_full_information_oracle | test | infeasible | 16/16 |
| privileged_full_information_oracle | test | mixed | 16/16 |
| privileged_full_information_oracle | test | restriction | 16/16 |
| public_rule_retrieval | test | cash | 18/18 |
| public_rule_retrieval | test | exposure | 18/18 |
| public_rule_retrieval | test | hold | 16/16 |
| public_rule_retrieval | test | infeasible | 16/16 |
| public_rule_retrieval | test | mixed | 16/16 |
| public_rule_retrieval | test | restriction | 16/16 |
| no_key_document | test | cash | 0/18 |
| no_key_document | test | exposure | 0/18 |
| no_key_document | test | hold | 0/16 |
| no_key_document | test | infeasible | 0/16 |
| no_key_document | test | mixed | 0/16 |
| no_key_document | test | restriction | 0/16 |

## 真实预算与更新

| 训练臂 | 预算上限 | 实际总token | 父/辅助token | 未使用尾余 | 有效更新 |
|---|---:|---:|---:|---:|---:|
| m3_seed42 | 3000000 | 2769082 | 1789252/979830 | 230918 | 16 |
| m3_seed43 | 3000000 | 2782394 | 1798659/983735 | 217606 | 26 |
| rloo_seed42 | 3000000 | 2747240 | 2747240/0 | 252760 | 0 |
| rloo_seed43 | 3000000 | 2808648 | 2808648/0 | 191352 | 0 |
| sham_seed42 | 3000000 | 2774190 | 1789252/984938 | 225810 | 0 |
| sham_seed43 | 3000000 | 2766946 | 1791400/975546 | 233054 | 0 |

截至已记录轮次，rloo_seed42、rloo_seed43、sham_seed42、sham_seed43累计0次有效RL更新；零优势批次保持原样跳过，复用终点评分不构成新增学习证据。

相同的是各臂生成token预算上限，实际消耗及尾余分别报告，不写成严格等算力。输入与输出token均计入；M3/sham辅助续写在预算内单列，辅助token不训练actor。开发曲线只画实际记录的累计token点，不插值制造相同token checkpoint。共同SFT监督输入13377204token，标签276708token，和RL生成预算分列。

共同SFT逻辑训练仍为一轮325次更新、5200条样本使用。一次执行优化重启废弃3次更新、48条样本使用，额外消耗123581输入、2213标签token。实际各attempt合计328次更新计算、5248条样本使用、13500785输入与278921标签token，和逻辑一轮分列；没有重写数据。[执行记录](protocol.json)。

采样器后续默认eager以减少每次启动的编译开销；初始original/SFT开发评估及RLOO seed42首批的graph执行记录保留。执行模式变化不改变BF16/native分布或生成token预算，各次engine_settings均保留；实际GPU阶段wall time包含不同执行开销，不能直接当作严格的方法算力比较。

## 信息credit的解释边界

Train信息干预诊断按arm、seed、轮次和当时策略区分丢失成功的原因。已有快照中，证据完整性占相当部分，另有业务或证明错误及未提交；因此局部credit同时包含引用与流程完成价值，不能全部称为金融决策价值。完整分组计数以[信息干预诊断](summary/information_credit_diagnostics.json)为准；这项机制解释不替换主成功指标，也不改变训练协议。

## 执行时点变化的副读数

主判仍是在固定已知报价下验证委托与证据，主表口径不变。CPU将原报价下82个可行计划放到下一个真实开盘执行，实际执行82个，其中52个仍满足原委托；另有2例跨时点委托版本变化，单独记录。这些数字只展示行情和生效时点改变带来的执行边界，不能将其比例当作主任务成功率，也没有替换主判报价或向模型提供未来报价。[副读数](baselines/dev_next_open_summary.json)、[逐任务记录](baselines/dev_next_open_diagnostic.jsonl)。

必要接线检查状态PASSED，14/14项通过。不可行声明须有实际约束证书；已有原模型输出用真实原动作CPU重放验证，原始生成与修正前记录保留，没有重新生成模型答案或修改教师数据。[检查记录](validation_checks.json)。

控制器阶段wall time合计8557.5713秒；已完成GPU模型阶段记录wall time8557.5713秒。模型阶段时间包含其加载、采样或训练阶段开销，不冒充GPU核函数计时。
[完整JSON](summary/readout.json)、[终点表](summary/table.csv)、[每类表](summary/per_type.csv)、[配对成功变化](summary/paired_success.csv)、[实际token曲线表](summary/development_token_curve.csv)。
