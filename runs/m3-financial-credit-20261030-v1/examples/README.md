# 强SFT的真实同前缀信息干预与恢复

仅整理本批fresh Train的已记录轨迹；没有重新采样、额外环境重放或读取Test。每个问题从首次读到指定财报前的同一真实前缀分叉，三条件各两次配对续写。masked只令这一次返回不可用；之后仍可重读原文或等价副本，工具上限仍为10次。placebo仅去掉返回中的可用/生效日期元数据，保留正文；它是格式控制，不是无关工具步骤。

本批gate：`FAIL / INSUFFICIENT_REPRODUCIBILITY_OR_PLACEBO_SEPARATION`。完整局部支持12个，非零均值差6个。两次都呈现超出placebo的关键业务损失共有2个基础问题，预设要求3个。未过门槛不等于差值全零；这个有限探测也不推论所有step credit无用。

| 条件 | 完整轨迹 | 业务正确 | 完整成功 | 指标正确 | 触发正确 | 执行正确 | 引用有效 |
|---|---:|---:|---:|---:|---:|---:|---:|
| factual | 24 | 18 | 18 | 18 | 21 | 22 | 24 |
| masked | 24 | 12 | 9 | 13 | 18 | 20 | 12 |
| placebo | 24 | 18 | 18 | 18 | 21 | 22 | 24 |

| 条件 | 终局调用均值（同24条分母） | 实际生成请求 | 输入＋输出token |
|---|---:|---:|---:|
| factual | 6.000 | 72 | 391,713 |
| masked | 7.250 | 102 | 535,291 |
| placebo | 6.000 | 72 | 387,177 |

masked的24条续写中，15条实际再次取得原文/等价副本，其中11条完成正确业务。业务与引用分开，不能把仅缺引用的失败解释成业务credit。
这里“原文重读0”指没有再次读取同一个doc_id；等价资料按实际FinQA source_id核对，不是说没有重读同源财报内容。每条恢复记录保留实际doc_id与source_id。

## financial-credit-base-10013

[已记录原始前缀、输出与工具返回](financial-credit-base-10013_recorded_probe.json)。

| 配对重复 | 条件 | 业务正确 | 提交指标 | 提交触发 | 调用数 | 引用有效 |
|---|---|---|---|---|---:|---|
| 0 | factual | 1.0 | 0.7600235271051857 | True | 6 | True |
| 0 | masked | 0.0 | 0.7884 | True | 7 | False |
| 0 | placebo | 1.0 | 0.7600235271051857 | True | 6 | True |
| 1 | factual | 1.0 | 0.7600235271051857 | True | 6 | True |
| 1 | masked | 0.0 | 0.7787878787878788 | True | 6 | False |
| 1 | placebo | 1.0 | 0.7600235271051857 | True | 6 | True |

该题两次factual与placebo均完成正确业务，两次masked均失败；失败包含指标错误，不能只归为引用损失。

masked `probe/financial-credit-10013-v0/rep1/masked/rep0`：原文重读0次、等价副本读取0次；业务正确=False，原10次预算内=True。

masked `probe/financial-credit-10013-v0/rep1/masked/rep1`：原文重读0次、等价副本读取0次；业务正确=False，原10次预算内=True。

## financial-credit-base-10110

[已记录原始前缀、输出与工具返回](financial-credit-base-10110_recorded_probe.json)。

| 配对重复 | 条件 | 业务正确 | 提交指标 | 提交触发 | 调用数 | 引用有效 |
|---|---|---|---|---|---:|---|
| 0 | factual | 0.0 | 3620800.0 | False | 6 | True |
| 0 | masked | 0.0 | 3621621.62 | True | 7 | False |
| 0 | placebo | 0.0 | 3620800.0 | False | 6 | True |
| 1 | factual | 0.0 | -821.6200000001118 | False | 6 | True |
| 1 | masked | 1.0 | 3621621.62 | False | 7 | False |
| 1 | placebo | 0.0 | -821.6200000001118 | False | 6 | True |

该题两次factual均失败，masked一次失败、一次终局业务正确；局部差值因此反向，但没有重复成立。它说明下游采样与恢复会改变局部符号，不能据此断言财报信息本来有害，也不证明每个中间计算步骤都正确。

masked `probe/financial-credit-10110-v1/rep3/masked/rep0`：原文重读0次、等价副本读取0次；业务正确=False，原10次预算内=True。

masked `probe/financial-credit-10110-v1/rep3/masked/rep1`：原文重读0次、等价副本读取0次；业务正确=True，原10次预算内=True。

[完整行为汇总](../summary/signal_behavior_analysis.json) · [信号门槛](../probe/paired/gate.json) · [固定分母读数](../REPORT.md)
