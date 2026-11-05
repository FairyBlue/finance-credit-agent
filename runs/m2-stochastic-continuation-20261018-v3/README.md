# 随机策略后续可行性pilot：已完成

2026-10-18，21:37:21 SGT启动v3，21:43:14全部154路径完成，21:43:18服务成功退出。77个状态—订单/hold条件、各2条独立后续，实际1903次请求，低于6904上限。共同恢复SFT、T1/p1、原自由工具协议，全部到原session20终点；CPU/GPU/服务及轻量读数实际exit0。不是训练或正式Valid/OOS，也不是pooling胜负试验。

[执行方案](../../docs/2026-10-18_M2_STOCHASTIC_CONTINUATION_PLAN.md) · [当前状态](../../docs/CURRENT_STATUS.md) · [driver日志](driver.log) · [GPU日志](worker.log) · [固定manifest](continuation_manifest.json) · [合同](contract.json)。

## 实际结果

| 起始j | 条件 / 路径 | 完整路径 | 模型请求 | 两次Y不同 / 相同 |
|---|---|---:|---:|---|
| 0 | 18 / 36 | 36 | 753 | 18 / 0 |
| 5 | 19 / 38 | 38 | 578 | 19 / 0 |
| 10 | 19 / 38 | 38 | 380 | 18 / 1 |
| 15 | 21 / 42 | 42 | 192 | 13 / 8 |
| 合计 | 77 / 154 | 154 | 1903 | 68 / 9 |

1903条生成全部完整，无截断；1726次推进、177次拒单（买入方向136、卖出方向38、池外股票3），无信息工具。96条路径有额外非推进请求，全部由模型继续恢复；最多3请求/session，未用完4次上限，无自动hold。3371次环境调用=1314原prefix+154固定首动作+1903模型响应。输入8,870,710tokens、输出39,927tokens。

全部77对完整，无配对NA；绝对Y差中位0.60809666、均值0.73265425、最大3.83520386，单位是原金融Y，不是收益百分点或标准差。8对原文序列也完全相同，另1对Y相等但原文序列不同。两draw描述支持进一步设计独立估计/参考，不能推断精确Q、零方差、pooling MSE或方法优越性。

GPU模型就绪到最后返回约3分17秒，v3服务含初始化约5分57秒；采样时实测利用率100%，整卡峰值35.06GiB。holder已回到76GiB、paused=false、claim=None。两次前置失败/修复耗时单独保留，不混入有效推理时间。

## 完成与轻量检查

| 范围 | 收据 |
|---|---|
| 完整154路径与1903请求 | [driver完成](driver_completion.json)、[worker完成](worker_completion.json) |
| 实际进程与服务退出0 | [CPU/GPU退出](process_exit_receipt.json)、[systemd退出](service_exit_receipt.json) |
| 轻量读数及实际exit0 | [完整读数](continuation_readout.json)、[逐路径](continuation_paths.csv)、[配对](continuation_cells.csv)、[分日期](continuation_days.csv)、[读数退出](readout_exit_receipt.json) |
| 必要源码/模型stat/预算/失败记录/资源交还核验 | [执行检查](final_execution_checks.json)、[最终holder状态](holder_return_verified.json) |
| CPU快路径与首笔真实成交 | [快加载](fast_preprocessing_receipt.json)、[真实买单回归](first_buy_smoke_receipt.json) |
| 启动/独立driver/读数身份 | [启动合同](contract.json)、[driver seal](driver_start_seal.json)、[读数seal](readout_source_seal.json)、[12项synthetic读数夹具](readout_fixture_receipt.json) |

沿用户要求没有重复8GB权重、9220行情文件或230项历史库存全量审计。启动小文件/大小/mtime及执行后stat一致，实际请求读数验证原生ID0..1902、prompt/seed/attempt/EOS与预算、原终点和完整失败分母。早期holder_return_receipt.json记录过正常GROWING过程，最终已恢复状态以holder_return_verified为准。

## 执行定义与边界

GPU加载和CPU环境准备并行。启动复用刚完成完整校验的模型身份收据，只查merge、文件大小与mtime/stat；行情通过原生产loader按symbol并行读取，不再重复9220文件checksum扫描。实际每条完整prefix的金融状态、真实观察与tokens仍匹配历史，固定首动作合法推进和账本一致后才请求模型。CPU与GPU分进程，CPU的USE_TORCH=0不影响推理。

当前session最多4次模型请求，信息工具、拒单和恢复均计入；截断、不完整结束、输入上限和请求上限保持NA，不补抽、不自动hold。所有154路径保留，不按收益或gate选状态。两个rep只描述可行性、请求成本与后续差异，不作为独立估计/参考或市场样本。

原首动作不计新增模型请求。hold首动作之后同样由π继续，因此该条件是Q^π(h,hold)样本，不是全程hold H，也不是V^π。正式科学比较须在本pilot之后另定参考估计精度与预算。

v1缺CPU命令run参数、v2比较原Fill.datetime与JSON字符串，在0模型请求前失败，完整保留；[前次记录](previous_attempts.json)。v3新增controller_v3、driver_v2处理这两处，原版本不改写，同154路径与seed表；[真实买单回归](first_buy_smoke_receipt.json)确认日期归一后info/账本与旧结果完全一致。

固定输入与源身份见contract、request_seed_table.jsonl；CPUdriver在首次启动前另存driver_start_seal.json。worker先保存submissions/raw_generations再分类，generations保留原生ID、prompt、seed、EOS及原文。driver_requests/driver_evaluations保留实际观察对应请求与原环境反馈，trajectory_results仅完整终点有Y，其余NA。

运行结束以driver_completion、worker_completion、process_exit_receipt和service_exit_receipt为准；launch_receipt只证明服务已启动。GPU沿原holder租约交接，整卡上限95GiB且保留2GiB物理余量。结果已整合[机制主题](../../docs/history/CREDIT_ORACLE_DIAGNOSTICS.md#随机策略后续pilot154路径全部完成)。

新的复现目录使用`m2_continuation_controller_v3.py prepare --output 新目录`，按本目录driver_ready.json写相同driver_v2源码seal，再依据launch_spec.json替换unit/output启动专用服务；不能绕过租约直接启动GPUworker。结束后运行`m2_continuation_readout.py --run 新目录`。原目录排他写，旧失败不覆盖。

后续独立估计/参考建议另固定fresh seed、124合法槽各2条估计和77cell各8条参考，最多39024次新请求；本轮两rep不重划到两侧。该预算尚未执行，不以本轮可行性PASS宣布pooling有效。
