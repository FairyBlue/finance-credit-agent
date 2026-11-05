# M2 同输入首请求采样执行方案

2026-10-18，SGT（UTC+8）。用户已批准继续执行[同前缀菜单验证](history/CREDIT_ORACLE_DIAGNOSTICS.md)后准备的[128次采样清单](../runs/m2-same-prefix-menu-20261018-v1/sampling_proposal.json)。本方案在模型生成前封存，回答同一真实输入下的合法推进、动作差异、具体订单重复与singleton覆盖，不训练模型，不检验收益优越性。

## 固定输入与模型

原清单保持不变：旧M2、source seed42、原兄弟0、八个固定Train窗口、零起算session 0/5/10/15，共32个anchor。每个anchor四槽共用已逐token核对的同一prompt IDs，分别使用清单中128个无碰撞请求seed。source seed与本轮RNG seed分开记录，不根据输出、收益或支持情况换状态或补抽。

模型是10月11日共同恢复SFT的本地merged_model，不是旧M2各轮checkpoint。执行前核对merge receipt、全部导出文件和权重分片SHA。本轮属于SFT在M2历史状态上的条件行为诊断，32个决策点不是独立的32个新episode。

只把原prompt IDs交给模型，保持single_tool_portfolio_v1原自由工具接口和enable_thinking=False；菜单与未来hold标签不注入输入，不用约束解码。prompt上限8192、输出4096、context12288；固定temperature1、top_p1、top_k−1、repetition_penalty1、n=1，并逐请求绑定seed。共128次原生首请求，最多524288输出tokens；没有后续模型调用、恢复交互或自动重试。

## GPU执行与原始证据

使用已验收的vLLM 0.9.2原生offline路径、BF16、TP1、max_num_seqs8、batched_tokens8192、chunked prefill及prefix cache，gpu_memory_utilization0.35。按固定顺序每批8请求；所有SamplingParams及返回的request ID、prompt IDs、输出tokens、engine text、finish/stop状态和EOS位置完整落盘。先保存整批raw返回，再映射和判读；返回ID必须全局唯一并按原生插入顺序对应输入，同prompt不能掩盖错配。

CPU controller运行在新的专用systemd服务内，沿用现有persistent holder租约。先核对owner/UUID，再claim、暂停并仅释放本holder payload，fresh free至少40GiB后启动独立GPU子进程。保留原holder的76GiB idle target及80GiB自身上限。新监控按设备total−free记录总占用，外层上限95GiB，同时要求至少2GiB物理余量；真实物理总量约93.6GiB，不按95GiB分配。worker退出且全部后代清空后，沿原controller_idle流程回填，服务结束后核对claim清除与占位恢复。失败只终止本服务的子进程/后代，不操作其他GPU任务。

## 完整性与环境分类

记录原生生成完整性、环境状态、终点标签三层。自然EOS正常结束且非空、tokens少于4096、returned.finished为真、finish_reason为stop才进入完整候选；原配置允许的EOS为151645/151643，输出应以其一结束、内部不含更早EOS。未知结束、abort、空输出、token上限或不一致停止均保留为INCOMPLETE，不执行、不补抽。保存原stop_reason（None可合法表示主EOS）。

完整响应按原tokenizer skip_special_tokens=True解码，**原文本**直接交给真实环境。统计时再规范化symbol及数值，不能先修调用再执行。原生Hermes解析、原参数、实际info/trace与返回观察留存：非法为INVALID；合法retrieve/compute为NONADVANCING；只推进一个session且无非法标记为ADVANCING。信息工具、非法、截断不计hold、不计q=0。

CPU标签阶段逐槽独立reset并重放完整旧prefix，核对金融状态和真实Observation.messages，保留同session拒单、工具记忆与风险历史。合法推进订单继续hold至原窗口终点，得到金融Y和相对同状态旧H的q；它是固定hold后续标签，不是原策略后续Q。非法/非推进/不完整的Y/q为NA。标签完全离线，不进入模型输入。

具体经济订单键仅对合法推进定义：buy/sell按工具名、规范symbol、有限数值target，0/0.0/−0.0同值；hold与abstain保留为不同动作族，abstain reason只留API证据，不人为增加经济订单种类。网格外合法订单仍合法，单列off-grid及该anchor菜单成员关系，不映射到邻近目标。

## 分母、完成标准与边界

128槽全部保留并逐组报告合法完整推进数k=0…4。主n=4分歧gate、重复partition、peer/singleton覆盖仅在四槽均完整生成、合法推进且固定hold标签完整的组上定义；partial组主指标为NA，不中心化剩余成员、不跨anchor凑同伴。可另列合法子集覆盖，但明确条件与k。报告原family gate和具体订单分歧分别覆盖多少组，family pooling机会还需gate打开且同族至少两人。

执行完成指128次实际请求身份与返回全部留存、全体槽已分类、所有应做的CPU标签完成、真实退出状态及GPU归还核对通过；即使包含INCOMPLETE或INVALID，也不能悄悄删除它们来宣称覆盖通过。若发生进程/设备或证据不一致，保留失败及未启动槽，不自动重新生成。

读数不计算新的pooling收益、降噪MSE或动作排序。固定hold确定性结果中，同状态具体订单重复的Y相同是环境结构；本pilot首先检验真实模型是否提供足够的可比较动作与重复支持。正式Valid/Final OOS、RL和学习baseline保持未启动。
