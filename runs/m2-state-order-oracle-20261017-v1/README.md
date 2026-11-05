# 多状态具体订单 CPU 反事实回放

2026-10-17，SGT（UTC+8）。采集、读数、绘图均已完成，实际退出码均为 0。64 条历史精确复现、256 个原对角合法完整；1024 个移植 cell 中 878 合法、146 非法。256 个状态实例来自 64 条旧训练轨迹，不是新增模型轨迹；缓存合并后实际计算 624 个不同分支。零模型请求、零训练更新，无 GPU 或正式 Valid / Final OOS 访问。

[结果报告](../../docs/2026-10-17_M2_STATE_ORDER_ORACLE_REPORT.md) · [执行前方案](../../docs/2026-10-17_M2_STATE_ORDER_ORACLE_PLAN.md) · [后续安排](../../docs/2026-10-11_M2_NEXT_STEPS.md)。

## 数据与证据

- [contract.json](contract.json)：真实回放前封存的方案、源码与数据 SHA；`source_manifest.json` 保存选择规则、来源文件及原请求定位。合同是执行时身份，不随完成状态改写。
- [start_receipt.json](start_receipt.json)、[collection_completion.json](collection_completion.json)：启动及采集完成时间、实际计数、三份原始输出 SHA。`historical_replay.jsonl` 保存原轨迹复现，`branch_results.jsonl` 保存缓存去重后的真实分支，`state_results.jsonl` 保留全部 1024 个逻辑 cell 及失败。
- [readout.json](readout.json)：完整描述性分析、目标及限制、恒等检查、分层汇总与四份 CSV SHA。主比较只用四订单均合法完整的 anchor，缺失仍留在覆盖分母。
- [readout_metrics.csv](readout_metrics.csv)、[readout_coverage.csv](readout_coverage.csv)、[readout_algebra.csv](readout_algebra.csv)、[readout_matched_rows.csv](readout_matched_rows.csv)：主指标、支持覆盖、H/q/F 分解、支持行 tau；原四槽与不同订单排序分母分别保存。
- [independent_readout_audit.json](independent_readout_audit.json)：独立读取实际 `state_results.jsonl` 重算的计数、主结果、组件及 tau 核对，不来自重复模型采样。
- [integrity_audit.json](integrity_audit.json)：合同与输出 SHA 核验及合法 skip / 零成交计数。
- [execution_receipt.json](execution_receipt.json)、[final_verification.json](final_verification.json)：实际命令退出状态、最终完整性检查及当前文档 SHA。
- [PNG](state_order_oracle_summary.png)、[SVG](state_order_oracle_summary.svg)：读数生成的静态图，绘图脚本不属于回放前合同，单独记录源码 SHA。

金融目标为原窗口终局 Y，固定动作后 hold 至原终点，保留完整权益、风险与费用历史。非法移植为 NA，不替代成 hold。未来 hold baseline 是特权诊断；确定性同状态约零误差是代数恒等，不代表降噪、可部署 baseline 或 RL 收益。

## 在新目录复现

从仓库根目录执行。必须保留合同中原始输入和软件环境；以下命令创建**新的**目录，不覆盖本批证据，既有目录会被拒绝。

```bash
.venv/bin/python -B scripts/diagnostics/m2_state_order_oracle.py prepare \
  --output runs/m2-state-order-oracle-reproduction \
  --manifest runs/m2-state-order-oracle-20261017-v1/source_manifest.json
.venv/bin/python -u -B scripts/diagnostics/m2_state_order_oracle.py run \
  --output runs/m2-state-order-oracle-reproduction
.venv/bin/python -B scripts/diagnostics/m2_state_order_readout.py \
  --run runs/m2-state-order-oracle-reproduction
MPLCONFIGDIR=/tmp/m2-state-order-mpl-cache .venv/bin/python -B \
  scripts/diagnostics/plot_m2_state_order_oracle.py \
  --run runs/m2-state-order-oracle-reproduction
```

采集器会核验封存输入与生产源码身份，先完整重放历史，再计算订单移植及 hold 分支；没有自动补采、订单修复或更换窗口。正式复现使用实际环境数据门禁。读数脚本只读取本批已落盘结果，不需要 GPU、模型服务或再次市场回放；默认读数和图像输出拒绝覆盖。
