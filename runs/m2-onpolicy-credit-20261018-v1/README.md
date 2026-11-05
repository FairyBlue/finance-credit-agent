# 当前π访问状态的credit与固定收缩验证

2026-10-18（SGT，UTC+8）。22:40:58服务启动；22:51:20采样完成，22:51:25服务实际exit0。共同恢复SFT π，8条fresh父轨迹、16个首次进入j5/15的实际状态、64个单次head及300条后续全部分类，3403次请求。未训练，未读取正式Valid/Final OOS。

固定方案：[取样前原文](../../docs/2026-10-18_M2_ONPOLICY_CREDIT_PLAN.md)。父轨迹从reset开始，八个窗口均为此前已使用Train日期：0001/0009/0017/0025/0065/0073/0149/0157，不含旧M2交互前缀。父/head/EST/REF四namespace预列23744个潜在请求种子，与旧轮分离；未用种子不回收补采。

## 实际执行与支持

8父轨迹全部COMPLETE、16状态全CAPTURED；64head中54合法、10拒单保留，head不repair/retry。24个同条件具体订单cell，108 EST+192 REF后续全COMPLETE，每REF cell八次完整，同订单共享REF前首步后账本与完整消息核同。实际各阶段172/64/1059/2108请求，总3403（上限23744）。3054推进、349拒单；父/后续339拒单均继续恢复，10head拒单按方案不恢复。最多3请求/session，预设上限4，无截断或自动hold。

主分母32组rep/128槽rep，11个完整合法head状态各2EST形成22组rep/88共支持槽；5partial状态的10组rep/40槽所有九方法全NA。没有因为父最终收益选状态，全部16状态保留；此主比较适用于完整合法四首动作条件子集。

## 九方法结果

原金融Y平方单位；REF按cell求八次均值，再按原四slot重复权重中心化τ。α收缩raw exact residual，η=.25只作共同派生尺度，不混淆两者。本轮没有GAE/whitening或更新模型。

| 方法 | REF8 MSE | REF前4 | REF后4 |
|---|---:|---:|---:|
| zero，S=0 | 0.11821 | 0.15942 | 0.17912 |
| ungated plain | 0.42580 | 0.42697 | 0.52675 |
| family gate plain | 0.20301 | 0.22787 | 0.28026 |
| order gate plain | 0.22335 | 0.22451 | 0.32429 |
| family pool/order gate | 0.14349 | 0.19341 | 0.19568 |
| exact pool/order gate | 0.15659 | 0.15775 | 0.25753 |
| α=.25 exact | 0.08905 | 0.12025 | 0.15997 |
| α=.5 exact | 0.08573 | 0.10691 | 0.16665 |
| α=.75 exact | 0.10824 | 0.11941 | 0.19917 |

α=.25/.5对zero的loss差分别−0.02916/−0.03248，两份参考半组均负，出现值得独立复核的非零credit线索。两α之间的full8差−0.00332，但半组−0.01333/+0.00669翻转，不能选.5最佳；α=.75、未缩exact对zero也翻号。family pool比zero差0.02528，两半均正。

7个同订单状态的56槽为结构性零目标；实际有区分的仅4个状态/32槽。三同族异订单状态24槽上，zero=.35255，α=.25/.5=.20915/.14947；唯一跨族状态8槽上zero=.24272，两收缩=.35213/.49462，均更差。次要逐anchor删除敏感度中，full8改善始终保留，但删除p007/j15后第二参考半翻正，参考稳定性有限。所有α并列报告，不据分层事后选α或改gate。

zero仅表示附加S置零，交易均由共同π生成。REF8非精确Q、半组同向非显著性，结果不证明训练/金融收益。新鲜状态仍来自已接触Train窗；与旧状态轮不同的时点/head/seed不能拼成分布改变的因果对照。下一步应保留原九方法在独立新父轨迹/状态批次复核，先不扩大RL。

## 资源与完成收据

GPU预热22:43:43完成，新三阶段驱动22:45:32封存启动，CPU快准备22:46:09就绪；预热后约2分26秒等待接线/准备，不记作有效推理。随后父/head/branch使用同一worker至22:51:20。服务约10分26秒，读数0.18秒；整卡峰值35.01GiB，原holder已恢复76GiB、claim=None。未重复8GB模型或行情全量哈希，真实prefix/观察/token/账本对应仍核验；下次可直接复用本轮驱动与GPU加载并行。

- [contract.json](contract.json)、[continuation_manifest.json](continuation_manifest.json)、[request_seed_table.jsonl](request_seed_table.jsonl)：方案、模型轻量身份、父状态/种子/硬预算。
- [driver_start_seal.json](driver_start_seal.json)、[readout_start_seal.json](readout_start_seal.json)：驱动与读数首用前独立封存；旧source/run不改写。
- [parent_results.jsonl](parent_results.jsonl)、[anchor_results.jsonl](anchor_results.jsonl)、[head_results.jsonl](head_results.jsonl)、[branch_manifest.json](branch_manifest.json)、[trajectory_results.jsonl](trajectory_results.jsonl)：全部planned分母、实际状态、非法head、动态合法分支和完成结果。
- [onpolicy_readout.json](onpolicy_readout.json)、[逐槽与NA](onpolicy_slots.csv)、[九方法](onpolicy_methods.csv)、[分层比较](onpolicy_strata_comparisons.csv)：固定全表与有限参考结果。
- [driver_completion.json](driver_completion.json)、[worker_completion.json](worker_completion.json)、[process_exit_receipt.json](process_exit_receipt.json)、[service_exit_receipt.json](service_exit_receipt.json)、[readout_exit_receipt.json](readout_exit_receipt.json)：CPU/GPU/service/readout实际exit0。
- [最小全流程IPC](driver_minimal_ipc_completion.json)、[代数夹具](m2_onpolicy_credit_algebra_fixture_20261018.json)、[reader核心](m2-onpolicy-credit-readout-fixture-20261018-v1.json)、[schema夹具](m2-onpolicy-credit-adapter-fixture-20261018-v1.json)、[独立公式核对](m2_onpolicy_credit_reader_math_review_20261018.json)：小型验证均PASS，未阻GPU采样。
- [独立实际小表重算](m2_onpolicy_credit_actual_review_20261018.json)、[轻量终验](final_execution_checks.json)、[holder归还](holder_return_verified.json)：全部九公式/REF/slot权重/分层、固定seed预算/模型stat/真实退出及资源归还通过，无大模型/行情重复扫描。

## 复现入口

新空目录执行prepare，启动参数照[launch_spec.json](launch_spec.json)与[launch_receipt.json](launch_receipt.json)，通过既有租约、隔离cgroup和独立source seal启动已完成driver；不在本目录重试/补跑。

```bash
.venv/bin/python -B scripts/diagnostics/m2_onpolicy_credit_controller.py prepare --output NEW_OUTPUT
.venv/bin/python -B scripts/diagnostics/m2_onpolicy_credit_readout.py --run NEW_OUTPUT
```

第二条只在全部阶段分类与真实进程退出后执行；driver在GPU加载期间ready、reader在实际Y读取前seal，具体格式见本次收据。完整解释并入[机制主题](../../docs/history/CREDIT_ORACLE_DIAGNOSTICS.md#当前π状态固定收缩出现有限参考改善)。
