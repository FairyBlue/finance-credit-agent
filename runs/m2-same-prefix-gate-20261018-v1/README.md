# 同输入具体订单gate隔离验证

2026-10-18，12:53:22 SGT（UTC+8）完成。**工程与证据核验PASS**，prepare/run实际exit0；67项核心/集成测试、Ruff检查通过。只读旧128次首请求及fixed-hold标签，零模型、环境、GPU调用和更新，未接生产训练，未读取正式Valid/OOS。

[执行前方案](../../docs/2026-10-18_M2_SAME_PREFIX_GATE_PLAN.md) · [结果主题](../../docs/history/CREDIT_ORACLE_DIAGNOSTICS.md#gate单因素验证候选实现与信号覆盖通过) · [当前状态](../../docs/CURRENT_STATUS.md)。

## 实际读数与范围

保留32组/128槽。29个完整组的116槽支持主算子，3个partial组全部12槽对四算子均NA，包括其中8个已有合法标签的槽；不缩小组、不补0。原数据124合法/4拒单、k=0/1/2/3/4的0/0/1/2/29分母不变。

| 算子 | 非零组 / 29 | 非零槽 / 116 | NA组 / 槽 |
|---|---:|---:|---|
| zero | 0 | 0 | 3 / 12 |
| ungated_plain | 11 | 44 | 3 / 12 |
| family_gate_plain | 4 | 16 | 3 / 12 |
| order_gate_plain | 11 | 44 | 3 / 12 |

主输入是原金融Y，r=Y−fsum(Y)/4；pool关闭，η固定0.25仅派生ηS，非零阈值abs(S)>1e−12。family→order改变7组/28槽，原开启4组精确不变，没有丢失原有信号。输入排列等变；raw组和最大误差6.66e−16，q/Y共同H平移差最大4.44e−16。future H不传给候选，未计算GAE、whitening或actor advantage。

order_gate_plain与ungated_plain在本批确定性标签下逐值一致；18组同订单r为零。这是已知数据上的实现/覆盖验证，不是独立确认、pooling MSE、降噪、无偏性、排序或金融收益检验。样本是共同恢复SFT在M2旧Train状态上的条件结果，32状态及四槽不作独立市场样本。

## 身份、退出与独立审计

| 证据 | 文件 |
|---|---|
| 新方案/模块/runner/两测试共5项封存源码、旧source与输入身份 | [合同](contract.json)、[128槽快照](gate_inputs.jsonl)、[旧证据执行前基线](old_immutability_baseline.json) |
| 实际完成及退出0 | [完成收据](completion.json)、[实际执行收据](execution_receipt.json) |
| 实际四算子结果 | [读数](readout.json)、[逐槽JSON](slot_signals.jsonl)、[逐槽CSV](slot_signals.csv)、[逐组](group_signals.csv)、[方法汇总](method_summary.csv)、[检查](checks.json) |
| 67项synthetic夹具 | [预检收据](fixture_receipt.json)、[通过日志](fixture_tests.log)、[JUnit](fixture_tests.xml) |
| 独立实际算子复核，816项PASS | [审计](independent_operator_audit.json)、[标准库重算脚本](audit_scripts/independent_operator_audit.py) |
| 独立身份与旧证据不变PASS | [审计](independent_integrity_audit.json)、[脚本](audit_scripts/independent_integrity_audit.py) |

实际重新哈希确认三个旧run的85文件集合、目录及230项联合文件完全不变，其中包括156份封存源码/模型；新5源码、fixture、baseline、原行SHA和6份完成输出也一致。新runner直接按文件加载纯候选，实际backend模块列表为空。预封存曾发现旧包初始化会间接导入env，已修正隔离导入及干净子进程夹具；[当时的失败日志](preseal_fixture_failure.log)保留，属于预检，不是实际新run失败或模型尝试。

旧`compute_sibling_credit`仍要求四条完整20-decision轨迹；本轮不调用该接口伪造完整轨迹，只核对相同plain中心化公式。旧合同和旧run内说明也不回写；旧final_verification中documents_sha256仅记录当时可变文档快照，不约束后来CURRENT_STATUS的合理更新。

## 新目录复现

仅在与合同一致的源码、旧run和模型字节均可用时执行。先查看[预检收据](fixture_receipt.json)及对应测试，两轮脚本均CPU；新目录必须不存在，输出排他写入，不操作GPU holder。

```bash
.venv/bin/python -B scripts/diagnostics/m2_same_prefix_gate_probe.py prepare \
  --source runs/m2-same-prefix-sampling-20261018-v1 \
  --fixture runs/m2-same-prefix-gate-20261018-v1/fixture_receipt.json \
  --old-baseline runs/m2-same-prefix-gate-20261018-v1/old_immutability_baseline.json \
  --output runs/m2-same-prefix-gate-reproduction
.venv/bin/python -B scripts/diagnostics/m2_same_prefix_gate_probe.py run \
  --output runs/m2-same-prefix-gate-reproduction
```

新模型采样不在复现命令中。后续研究目标及预算建议见[方法评审](method_scope_review.md)：Q^π的随机后续、独立估计/参考seed与精度须另行固定；其6904请求可行性口径仍是草案，未封存为GPU执行方案、未启动，不作为本轮预算或已完成结果。
