# 独立估计/参考 credit 对照

2026-10-18（SGT，UTC+8）。21:58:37启动，22:17:43全部864路径COMPLETE；22:17:49服务实际exit0。GPU加载与CPU快准备并行，最终读数1.2秒；原holder已恢复76GiB、claim=None。

固定方案：[执行前原文](../../docs/2026-10-18_M2_CREDIT_COMPARISON_PLAN.md)。共同恢复SFT为π，32个旧Train前缀、原自由工具接口、T1/p1、原21-session终点；不训练，不读取正式Valid/Final OOS。124个合法原slot各2条fresh EST，77个订单/hold条件各8条fresh REF，共864条；4个非法原head保留NA。EST与REF及原pilot种子分开，重复订单的EST按原slot独立续写。

每session最多4个模型请求，包括信息工具、拒单和恢复；预列39024个潜在请求种子，无补抽、无代选hold。所有864条路径都保留。主分母64组rep/256槽，原partial的6组rep/24槽连zero参照全NA；其余需要四EST完整且各所需REF cell的8次完整。

REF先求具体cell的8次Y均值，再按原四slot重复权重中心化τhat；这是有限样本参考，不是精确Q或A^π。六方法共用支持，pooling主对照共用order gate。报告原金融Y平方单位的MSE、成对loss差及预分REF4+4敏感度；η=.25只作两侧共同尺度派生。

exact pool对同订单同值目标是投影，因此其相对order plain的MSE非增是逐样本结构恒等；方向稳定不能证明REF精度或策略收益。all-same-order时order gate置零也与τ=0结构一致。参考依赖的family-vs-exact和gate比较按订单分层报告，不称正式显著或金融收益。

## 实际结果

全部248 EST与616 REF完成，实际10788请求=3149 EST+7639 REF，低于39024硬上限；9756推进、1032拒单均继续恢复，最多3请求/session。无截断、补抽或自动代选hold。64组rep/256槽主分母中，58组rep/232槽六方法共支持，原partial的6组rep/24槽全部NA；77个REF cell各8次完整。

| 方法 | REF8 MSE | REF前4 | REF后4 |
|---|---:|---:|---:|
| zero，S=0 | 0.03608 | 0.05261 | 0.04336 |
| ungated plain | 0.59301 | 0.60738 | 0.60246 |
| family gate plain | 0.12934 | 0.15857 | 0.12391 |
| order gate plain | 0.26920 | 0.28356 | 0.27864 |
| family pool，同order gate | 0.05849 | 0.08520 | 0.05560 |
| exact pool，同order gate | 0.11992 | 0.13428 | 0.12937 |

原金融Y平方单位；zero仅表示附加S置零，所有路径的交易均来自共同π。本轮没有训练这些信号。exact−family pool的成对loss差为+0.06143，两半+0.04909/+0.07377；order−family gate plain为+0.13986，两半均正。exact相对family gate plain全8差−0.00942，但两半−0.02429/+0.00545翻号，记不可判。

144槽四单相同之外，同族异订单56槽的zero/family pool MSE为0.08108，exact 0.16292；跨族32槽zero 0.11968、family 0.28217、exact 0.58429。family pool在前两层共200/232槽结构性为零，较低误差不等于恢复有效动作credit。当前没有直接升级gate/exact pool或扩大RL的依据；8REF非精确Q、分半同向非显著性或收益证据。

CPU快准备21:59:42已就绪；模型预热22:01:15完成，最后生成22:17:43。服务约19分13秒，模型就绪后约16分28秒；整卡采样峰值34.88GiB。轻量终验没有重复权重/行情全量哈希。

结果：[credit_readout.json](credit_readout.json)、[全方法](credit_methods.csv)、[成对比较](credit_comparisons.csv)、[分层比较](credit_strata_comparisons.csv)、[逐槽及NA](credit_slots.csv)。完整解释并入[机制主题](../../docs/history/CREDIT_ORACLE_DIAGNOSTICS.md#独立估计与参考细分pool未优于零credit)。

## 运行身份与收据

- [contract.json](contract.json)：新worker/controller/方案及轻量模型身份；复用此前成功完整哈希收据，只核当前文件stat和小merge收据，不重扫8GB模型。
- [continuation_manifest.json](continuation_manifest.json)、[request_seed_table.jsonl](request_seed_table.jsonl)：所有head、cell、trajectory与固定独立种子。
- [driver_start_seal.json](driver_start_seal.json)：首请求前封存的新CPU驱动；源码`m2_credit_comparison_driver.py`。不改旧source/run。
- [fast_preprocessing_receipt.json](fast_preprocessing_receipt.json)：生产loader按symbol并行，沿用已验收checksum；实际prefix账本、完整观察和prompt token仍逐轨迹核对。
- [launch_spec.json](launch_spec.json)、[launch_receipt.json](launch_receipt.json)、[controller_start.json](controller_start.json)：独立systemd服务、启动接受及租约owner。启动接受不替代实际完成。
- [数学夹具](m2_pooling_reference_math_checker_20261018_result.json)、[驱动最小IPC](driver_minimal_ipc_completion.json)：小型CPU验证与源码副本保存在本目录，在GPU加载/采样期间保存。
- [readout_start_seal.json](readout_start_seal.json)、[21项小夹具](m2-credit-comparison-readout-fixture-20261018-v1.json)、[独立生产公式核对](independent_formula_review.json)：读取真实结果前冻结读数；[独立实际小表重算](m2_credit_actual_readout_review_20261018.json)由REF8/EST4重算全部算子、MSE及分层，通过。
- [driver_completion.json](driver_completion.json)、[worker_completion.json](worker_completion.json)、[process_exit_receipt.json](process_exit_receipt.json)、[service_exit_receipt.json](service_exit_receipt.json)、[readout_exit_receipt.json](readout_exit_receipt.json)：真实CPU/GPU/service/readout退出均0，无重试。
- [final_execution_checks.json](final_execution_checks.json)、[holder_return_verified.json](holder_return_verified.json)：来源/轻量模型stat不变、固定种子与预算、1032拒单及恢复、原占位76GiB归还均通过。

## 复现入口

在新的空输出目录执行，勿复用本次目录补跑：

```bash
.venv/bin/python -B scripts/diagnostics/m2_credit_comparison_controller.py prepare --output NEW_OUTPUT
```

启动方式见`launch_spec.json`，使用相同GPU租约和隔离cgroup。CPU driver在加载期间按SHA写入`driver_ready.json`，controller封存后启动；读数器在采样期间准备，执行前单独封存。读取完成结果的命令：

```bash
.venv/bin/python -B scripts/diagnostics/m2_credit_comparison_readout.py --run NEW_OUTPUT
```

仅在全部路径已分类和CPU/GPU/service真实退出后运行最终读数。原可行性pilot、两次零请求启动失败及所有旧结果保留。
