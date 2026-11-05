# 原生同输入首请求采样

执行日期：2026-10-18，SGT（UTC+8）。固定32个旧Train前缀，每个4个相同输入、不同seed的原生首请求，共128次；使用共同恢复SFT，保持原自由工具协议。实际进度见`live_status.json`，生成与CPU标签分别记收据；不发起训练、恢复交互或补抽。

**已完成，工程及证据核验PASS。** 11:21:38启动，11:25:35生成完成，11:32:05 CPU标签完成；GPU子进程/控制器/服务、CPU评估及独立读数实际exit0。128次生成完整，124合法推进、4拒单保留Y/q为NA；没有补采、优化器更新或正式Valid/OOS访问。

29/32组支持完整四槽，k=0/1/2/3/4分别0/0/1/2/29；其余3组主n4指标保持NA。完整组18个[4]、7个[3,1]、4个[2,1,1]。11组有具体订单分歧，原family gate仅打开4组；漏掉的7组固定hold贡献也均不同。全合法子集108/124个同订单peer槽、16个singleton；118个交易槽中104有peer、14个singleton。重复包含18组全同订单，不能将高重复率当作广泛比较支持。

合法交易只涉及GS/MSFT/HD，目标0或0.2，均在旧有限菜单内；10个零成交=6个hold+4个合法next-open skip。固定hold标签只作条件诊断，不证明降噪、原策略Q或收益优势。结果解释和下一步统一写入下方机制主题。

[执行前方案](../../docs/2026-10-18_M2_SAME_PREFIX_SAMPLING_PLAN.md) · [当前状态](../../docs/CURRENT_STATUS.md) · [机制主题](../../docs/history/CREDIT_ORACLE_DIAGNOSTICS.md)。

## 实际完成与独立证据

| 范围 | 凭据 |
|---|---|
| 固定128次生成、原生返回与实际退出 | [生成完成](generation_completion.json)、[worker退出](worker_exit_receipt.json)、[controller完成](controller_completion.json)、[systemd实际退出](service_exit_receipt.json) |
| 128槽CPU分类、124固定hold标签及实际exit0 | [标签完成](evaluation_completion.json)、[CPU退出收据](cpu_exit_receipts.json)、[逐槽原始证据](evaluations.jsonl) |
| 描述统计及主分母 | [读数](readout.json)、[逐槽](slot_metrics.csv)、[逐组](group_metrics.csv)、[分日期](day_summary.csv)、[检查](checks.json) |
| 独立实际审计 | [生成838项](generation_independent_audit.json)、[CPU账务与状态3042项](evaluation_independent_audit.json)、[统计重算](statistics_independent_audit.json)；源码在`audit_scripts/` |
| 显存归还与封存身份 | [实际归还审计](gpu_return_independent_audit.json)：156份source/model哈希一致，174次整卡观测峰值34.208GiB；11:28:40 SGT确认holder为76GiB、paused=false、claim=None |

四条拒单：a064/s0的HSI越出股票池，a072/s3买GS→0.2方向不合法，a104/s1、s2卖HD→0方向不合法。未发起后续恢复，不能将本轮首请求结果当作恢复验收。全部124首步账务与冻结菜单同订单记录一致；其中97个终局可用既有同订单终局全字段复核。余下终局仍完整执行并核对原评分、q=Y−H及固定hold无新费用/换手。

## 执行身份

- `proposal.json`是上一轮准备清单的逐字节副本。`inputs.jsonl`包含128个固定槽、原始prompt IDs、seed及来源；四槽不改变原观察，不注入菜单或未来标签。
- proposal内的PREPARED_NOT_STARTED是原清单的历史状态，保留原字节；本轮实际执行状态由本目录完成/退出收据记录，不回写旧run或冻结合同。
- `contract.json`在生成前封存模型全部权重、输入、方案、源码、关键vLLM源码、版本及资源配置。模型是共同恢复SFT，面对M2历史状态，属于条件行为诊断，不是原各轮M2策略复现。
- `model_preflight_receipt.json`记录全部导出文件、权重分片与训练来源的CPU预检；`worker_fixture_receipt.json`和`readout_fixture_receipt.json`是明确标记的离线测试，不计为真实模型输出。控制夹具源码在`audit_scripts/`。

## 原生生成与资源

- `launch_spec.json`、`launch_receipt.json`保存专用systemd单元及真实启动返回码；`controller_start.json`绑定控制器身份。
- `gpu_handoff/config.json`与`gpu_admission.json`保存既有persistent holder的租约、owner/UUID和paused-zero准入。运行期间保留95GiB外层上限和2GiB实际物理余量；硬件约93.6GiB，原holder自身76/80GiB设置不变。
- `worker_config.json`、`worker.log`保存真实推理配置和日志；`gpu_observations.jsonl`逐秒记录整卡占用，不将不可归属的显存抵扣。
- `submissions.jsonl`先保存16批实际请求及SamplingParams；`raw_generations.jsonl`先保存完整原生返回；`generations.jsonl`再按request ID和prompt核对，保留tokens、EOS、finish/stop及原文，截断和非法不补抽。
- `generation_completion.json`只说明128次原请求全部留存；各槽生成是否完整仍看generation_status。`worker_exit_receipt.json`、`controller_completion.json`及`service_exit_receipt.json`分别记录实际子进程、控制器和systemd退出状态，不以“已启动”代替实际exit0。

## CPU分类与判读

`evaluations.jsonl`逐槽独立重放全部原prefix，核对金融状态和实际观察，再将完整响应的原文本交真实环境。保存原调用、第一步info/trace/返回观察和经济统计键。INCOMPLETE不执行；INVALID和合法信息工具没有金融q；合法推进才hold至原窗口终点。标签只在离线侧，不输入模型。

`evaluation_completion.json`记录全体128槽已分类及应做标签完整；`readout.json`、`slot_metrics.csv`、`group_metrics.csv`、`day_summary.csv`、`checks.json`由独立读数生成。报告全体请求和k=0…4直方图；完整四槽主口径与partial合法子集分开，partial不中心化。网格外合法订单仍合法，不映射到菜单网格。CSV中的`legal_subset_non_hold_orders`专指buy/sell交易订单，不包括abstain。

本轮先回答真实同输入下的推进、动作分歧、具体订单重复和singleton覆盖，不计算新的pooling收益、降噪MSE或动作排序。固定hold标签不等于原策略继续行动的Q；共享行情、同窗日期和四槽不能当作独立样本。

## 新目录复现

先核对本目录合同对应的软件与数据。新目录必须不存在，输出排他创建；原模型权重和输入不改写。

```bash
.venv/bin/python -B scripts/diagnostics/m2_same_prefix_sampling.py prepare \
  --output runs/m2-same-prefix-sampling-reproduction \
  --model-receipt runs/m2-same-prefix-sampling-20261018-v1/model_preflight_receipt.json \
  --holder runs/gpu-reservation-persistent-20261007
```

GPU阶段须按`launch_spec.json`中的专用systemd服务、KillMode、资源与ExecStopPost配置运行`controller --output 新目录`，由控制器执行租约准入和子进程；不能跳过handoff直接运行worker。GPU子进程及后代退出后回填holder，service结束再核对claim清除。

生成完成后，CPU阶段和读数从仓库根目录执行：

```bash
.venv/bin/python -u -B scripts/diagnostics/m2_same_prefix_sampling.py evaluate \
  --output runs/m2-same-prefix-sampling-reproduction
.venv/bin/python -B scripts/diagnostics/m2_same_prefix_sampling_readout.py \
  --run runs/m2-same-prefix-sampling-reproduction
```

原始生成和CPU结果均不覆盖；任一进程或证据不一致保留失败及未启动槽。正式Valid/Final OOS及新训练保持未开启。
