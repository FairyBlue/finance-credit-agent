# 主动订单监督 demo：已结束

固定八个既有Train窗口、seed43；未访问正式Valid/Final OOS。该实验检验主动反事实标签能否被策略利用，不是订单池化或等算力优势检验。

1. 第一阶段32个真实状态、128个不同订单、512条分支全部到原终点，得到11条两半均改善标签。确定性菜单偏向AAPL，标签为buy8/hold2/sell1。
2. 从冻结八窗RLOO复制模型，独立Adam完成16次监督更新：32条winner使用、32条正常请求回放。最终模型在 `stage2/current_actor/`，未新增完整FP32/Adam磁盘状态。
3. 两组新采样namespace5/6的六个评估cell全部结束：SFT与RLOO各64/64完整，新模型58/64完整。新模型5条生成长度失败、1条请求上限失败；固定分母收益保持NA。

| 模型 | namespace5净收益 | namespace6净收益 | 到原终点 |
|---|---:|---:|---:|
| SFT | 2.27805% | 2.24413% | 64/64 |
| 冻结RLOO | 3.08679% | 3.10666% | 64/64 |
| 标签监督16步 | NA | NA | 58/64 |

出口为 `STOP_MODEL_ROLLOUT_FAILURES`：保留RLOO作demo基线，结束本次蒸馏配置，不追加训练、不重采失败。新模型的标签NLL下降，但环境错误率升至21.37%/21.57%，并出现工具格式与参数错误，不能把局部拟合当成整段交易改善。

冻结RLOO模型：`runs/m2-rloo-eight-window-20261025-v1/seed43/current_actor/`。该模型的完整16步FP32/Adam恢复点仍保留在其同级 `training_state/`；本轮未修改它。共同SFT为 `runs/m2-order-recovery-sft-20261011-v1/merged_model/`。

从项目根读取已完成结果，不占GPU：

```bash
.venv/bin/python scripts/diagnostics/read_counterfactual_order_demo.py \
  --output runs/m2-counterfactual-order-demo-20261027-v1 --preview
```

完整流水线入口是 `scripts/diagnostics/run_counterfactual_order_demo.py`，采样和训练分别在 `src/agents/counterfactual_order_collect.py`、`src/agents/counterfactual_order_train.py`。控制器通过独立systemd服务和GPU租约运行，已完成；本目录不允许隐式重跑。训练初次启动的字段合并错误发生于0次更新前，修复后复用已完成采样接续，见 `controller_attempts/` 与 `logs/`。

结果入口：

- [固定分母读数](summary/readout.json)、[终点表](summary/summary.csv)、[比较图，明确标注NA](summary/comparison.png)
- [行为及环境错误](summary/behavior_readout.json)、[逐窗结果](summary/window_results.csv)、[配对记录](summary/paired_results.csv)
- [标签分布及选样偏向](stage1/label_analysis.json)、[训练读数](stage2/training_summary.json)、[流水线完成记录](completion.json)

总新增生成10305次请求、48189742输入token、249851输出token；监督更新另外使用298018输入token与1304标签token，模型阶段合计约36.31分钟。源父迹复用，不重复计采样；无失败补抽、自动hold、提高原上限或中间模型选择。
