# 当前π状态：同窗口独立种子复验

2026-10-19，SGT（UTC+8）。14:50:24 GPU服务启动；CPU快准备14:51:47就绪，早于GPU预热14:53:08完成。父/head/branch连续共用同一worker。14:58:01全部采样完成，14:58:06服务实际exit0；3335请求，314后续全部COMPLETE，联合方向判据通过。

[取样前方案](../../docs/2026-10-19_M2_ONPOLICY_CREDIT_REPLICATION_PLAN.md)。九方法、固定α=.25/.5/.75、共同恢复SFT π、j5/15、原21-session终点及单次head规则保持。没有训练，正式Valid/Final OOS未读取。

## 日期范围与独立性

Train有461个session、441个21-session滑窗。相对旧8窗及既有12个Train开发评估窗，完整窗口之间至少留20个空交易session时，没有符合条件的新窗口；见[元数据选择收据](../m2-onpolicy-credit-replication-selection-20261019-v1.json)。因此沿用原8个Train窗口，每窗一条全新父轨迹，只验证同窗随机状态/Monte Carlo重复性，不称日期泛化或未接触数据。

replication_id与seed namespace为`m2-onpolicy-credit-replication-20261019-v1`；PARENT/HEAD/EST/REF各自独立，全部23744潜在seed与旧pilot、旧状态EST/REF及第一批当前π状态的完整潜在表无交集。未花种子不回收补采，不复制旧路径、标签或筛两批共同支持。

## 预算、支持与预设判读

8父轨迹、16 planned anchor、64个单次head。合法slot各2EST，同anchor/economic-order且首步后完整金融状态/消息核同后共享8REF；条件不一致整anchor NA，无额外hold。硬预算640父+64head+4608EST+18432REF=23744；父/后续每session最多4请求，信息工具/拒单/恢复都计入，head不repair/retry。

固定32组rep/128槽rep主分母，四个合法head、四EST完整且各所需REF八条完整才支持。任一缺失整组九方法含zero全部NA；未到达父状态不补跑，已达到的状态不因父后来失败删除。

预先关注α=.25/.5各自相对zero的candidate−comparator loss差：两者在REF8、REF前4、REF后4六个差均小于−1e−12，才记本批描述性方向共同复现；否则不挑另一个α为赢家。所有九方法/三个α和.5−.25仍并列报告。阈值只是符号/数值容差，不是显著性标准；没有支持或动作不分歧不加样。

α乘raw exact residual，η=.25只作两侧共同尺度派生。目标是原四slot重复权重中心化的有限Q代理，不是A^π/V^π或策略收益。zero只是附加S置零，交易都来自共同π。REF8/分半同向不代表精确Q或显著性。

## 实际完成与结果

8父轨迹完整、16状态全CAPTURED，64head中57合法/7拒单；25cell，114 EST+200 REF后续全部314 COMPLETE，25cell各8REF。父/head/EST/REF实际174/64/1131/1966请求，总3335，低于23744上限。2993推进、342拒单；父/后续335拒单继续恢复，7head按规则不repair。最多3请求/session，预设4次上限不变，无截断或自动hold。

主32组rep/128槽rep中，28组rep/112槽共同支持，原4partial组rep/16槽九方法全NA。6同订单状态48槽为零目标控制；5同族异订单状态40槽、3跨族状态24槽，共8个有动作分歧状态，仍非独立市场样本。

| 九个固定方法 | REF8 MSE | REF前4 | REF后4 |
|---|---:|---:|---:|
| zero，S=0 | 0.13978 | 0.14654 | 0.17334 |
| ungated plain | 0.37161 | 0.39113 | 0.39242 |
| family gate plain | 0.21768 | 0.23905 | 0.23663 |
| order gate plain | 0.28615 | 0.30567 | 0.30696 |
| family pool/order gate | 0.15306 | 0.17187 | 0.17458 |
| exact pool/order gate | 0.12485 | 0.14437 | 0.14565 |
| α=.25 exact | 0.08510 | 0.09505 | 0.11547 |
| α=.5 exact | 0.06438 | 0.07753 | 0.09157 |
| α=.75 exact | 0.07763 | 0.09396 | 0.10163 |

原金融Y平方单位。α=.25−zero的full8/半A/半B为−0.05468/−0.05149/−0.05787，α=.5为−0.07539/−0.06902/−0.08177，六方向均满足预设规则，记`PASS_DESCRIPTIVE_DIRECTION_ONLY`。本批两个异质分层均值和两半也均改善；逐一删除14个支持anchor的次要分析中，两个α三个panel仍全负。没有信号精度、显著性、跨日期或收益结论。

本批.5比.25更低，但首批两者排序在REF半间翻转，仍不选最佳α。未缩exact和.75本批也低于zero，完整报告而未改变主判据；family pool仍高于zero。两批各用自己的完整支持，不合并均值或筛共同状态。

本实验共享完整模型可见prefix及state，原生产M2后期siblings常为不同状态，不能直接替换原collector；REF8仅离线评价。下一步先固定新的同前缀训练接口，并保相同α的plain/exact对照，分清强度与池化作用，计入全部parent/branch成本。

服务约7分43秒；预热到末次生成约4分53秒，CPU已提前就绪、没有GPU等待CPU接线。读数0.59秒，CPU/GPU/service/readout实际exit0，峰值35.11GiB，holder恢复76GiB、claim清除，未重复模型/行情全量哈希。

[联合方向读数](replication_readout.json) · [九方法原表](onpolicy_readout.json) · [逐槽与NA](onpolicy_slots.csv) · [分层比较](onpolicy_strata_comparisons.csv) · [独立小表重算/次要敏感度](m2_onpolicy_replication_actual_review_20261019.json) · [轻量终验](final_execution_checks.json) · [资源归还](holder_return_verified.json) · [实际退出](readout_exit_receipt.json)。详细解释见[主题结果](../../docs/history/CREDIT_ORACLE_DIAGNOSTICS.md#独立种子复验同窗口的收缩信号方向复现)。

## 实现、收据与复现

新薄driver复用冻结`m2_onpolicy_credit_driver.py`，仅更换seed namespace和身份收据；controller复用已成功租约/隔离cgroup与原worker，旧源码/run不改写。新reader复用冻结九公式，单独生成`replication_readout.json`判读联合方向，不改写原`onpolicy_readout.json`或合并两批均值。

- [contract.json](contract.json)、[continuation_manifest.json](continuation_manifest.json)、[request_seed_table.jsonl](request_seed_table.jsonl)：方法、模型stat、预算、namespace及来源绑定。
- [driver_start_seal.json](driver_start_seal.json)、[最小全流程IPC](driver_minimal_ipc_completion.json)：新namespace的2528次fake RPC通过，测试没有真实环境/model/GPU调用。
- [readout_start_seal_v2.json](readout_start_seal_v2.json)、[v2小型检查](m2-onpolicy-replication-reader-v2-fixture-20261019-v1.json)：读取真实Y前封存。初版严格metadata接口缺少driver附带的两个身份字段，预检时修复为v2；原reader/seal保留，公式及联合标准逐字不变，未因此重跑模型。
- [launch_spec.json](launch_spec.json)、[launch_receipt.json](launch_receipt.json)、[fast_preprocessing_receipt.json](fast_preprocessing_receipt.json)：独立服务启动与CPU/GPU准备时序。启动接受不替代真实完成。

新空目录prepare，启动参数照本次launch_spec，driver_ready应包含薄入口与冻结base的SHA；CPU在GPU加载期间启动。全部路径分类和真实CPU/GPU/service退出后，用封存v2读数：

```bash
.venv/bin/python -B scripts/diagnostics/m2_onpolicy_credit_replication_controller.py prepare --output NEW_OUTPUT
.venv/bin/python -B scripts/diagnostics/m2_onpolicy_credit_replication_readout_v2.py --run NEW_OUTPUT
```

本目录不可重试/补跑。模型身份复用已验完整收据并核当前stat/小merge收据，未重复8GB权重/行情全量哈希；实际prefix、token、账本对应仍核验。
