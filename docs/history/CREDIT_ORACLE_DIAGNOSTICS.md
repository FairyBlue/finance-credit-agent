# 功劳机制与具体订单 Oracle（10 月 14 日—19 日）

整理日期：2026-10-19。合并池化存档、合成CPU oracle、多状态回放、同前缀菜单/采样、gate隔离、随机后续、当前π状态固定收缩及独立种子复验；详细原稿和冻结方案见[历史归档](../archive/README.md)，当前安排见[当前状态](../CURRENT_STATUS.md)。`runs/`完整产物主要在服务器。

**预设α=.25/.5的收缩credit，在同八窗口的新种子批次中方向共同复现。** 10/19的8条新父轨迹、16个状态及314条后续全部完成，3335次请求；112共支持槽的zero MSE为0.13978，α=.25/.5为0.08510/0.06438，两个REF分半均更低，满足取样前固定的六方向联合规则。证据仍局限于同窗Monte Carlo/状态重复性，不能称日期泛化、选择最优α或推出训练收益；原生产M2异状态siblings尚不能直接替换。本轮与首批、旧状态864路径均分别保留。

## 存档分析：池化生效，不等于 credit 更准确

使用[恢复与池化对照](RECOVERY_AND_ACTION_POOLING.md)的 256 条训练、1,080 条评估和 32 次更新存档重算，不启动模型。实际 actor 信号为 `whiten(U_GAE + ηS)`、η0.25；critic target 不加 S。Plain 与 pool 都按推进日比较四兄弟，使用同一动作族分歧 gate。

| 检查 | 结果 |
|---|---:|
| 排除初态后，同日兄弟前置组合不同 | 276/304 组，90.8% |
| 有分歧且同族至少两人的可池化组，具体订单不同 | 126/209，60.3% |
| 上述可池化组，前置组合不同 | 191/209，91.4% |
| 原 family gate 关闭，具体订单仍不同 | 118/163，72.4% |
| 同批 pool 剩余残差平方能量 / plain，seed42/43 | 44.96% / 33.12% |
| 实际 std(ηS)/std(U)，逐批范围 | seed42 4.13%–8.13%；seed43 3.60%–6.81% |
| Plain / pool 最终白化优势符号不同 | seed42 4/699；seed43 10/711 |

同族均值压缩约 55%–67% 的残差能量，但平均掉的差异可能含真实订单效果，不能当作已去除的噪声。GAE 仍占主要尺度；白化分母仅约增大 0.3%–1.5%，不足以解释为白化抹掉全部信号。

第 4 轮 sampled 的 M2−plain 平均差为 −0.06555 个百分点；12 窗 5 正、7 负，逐一移除任一窗均仍为负。两臂回合全部完整、最长连续无效请求均为 2；将费用按固定路径算术加回后，差仍约 −0.06116 个百分点，费用不足以解释方向。没有显著性或因果归因结论。

共同 benchmark 或加性常数会在 plain 和 pool 的中心化中同时消去，不是池化独有优势；不同持仓与同市场的暴露交互仍保留。32 次 critic 更新中 8 次同批 RMSE 上升，更新后 explained variance 为 −0.0532 至 0.1393，也是共同训练限制。

## 合成 Oracle：误差下降与订单排序必须同时报告

10 月 14 日完成 1,105,920 个合成 group、10,321,920 个成员、2,268 行指标；均不是模型轨迹，模型请求和优化器更新为零。主分析 n=4，n=8/16 仅作敏感度，不改变生产 rollout_n。

用真实 PortfolioLedger 定义相对同状态 hold 的下一收盘净资产增量，含费用；buy A/B 到 40% 的效果分别为 +0.76953/−0.62982 个百分点。下表固定独立扰动标准差 1 个百分点、共同市场扰动为零，MSE 单位为百分点平方。

| n=4 场景 | Plain MSE / 排序 | Family pool MSE / 排序 | 具体订单 pool，原 gate |
|---|---|---|---|
| buy A、buy A、hold、hold | 0.74502 / 71.06% | 0.24654 / 78.47% | 重复具体动作平均可降独立扰动 |
| buy A、buy B、hold、hold | 0.73922 / 72.12% | 0.49708 / 50.00% | 0.48994 / 74.27% |

无扰动的同族异质场景中，plain 和具体订单 pool 的 MSE 为零、排序 100%；family pool 已有 MSE 0.24477、排序 50%。它可以降低总 MSE，却抹掉方向相反订单的区分。具体订单全为 singleton 时精确退化为 plain；n=4 的改善部分来自两个 hold 的平均。

buy A、buy A、buy B、buy B 全属 buy，原 gate 将所有方法置零；只细化 pool 键无效，必须单独检验具体订单 gate。不同继承状态下，订单键相同也不能消除 hold 基线误差。共同加性项两者均抵消；市场暴露改变动作实现效果，不能一概称为噪声。

真实同状态探索回放另显示目标期限可翻转排序：0378 窗口中，历史 M2 订单下一日比 plain 少 0.04177 个百分点；之后共同 hold 到终点却多 1.90071 个百分点。该案例由旧差异选出，不是独立优势证据，但说明一日净收益不能替代原终局金融 Y。

## 多状态 Oracle：保留完整 prefix 与非法菜单分母

10 月 17 日按预先冻结方案回放两个 seed、八个原训练窗、session j=0/5/10/15：64 个 source group、256 个 anchor、1,024 个订单移植 cell，另有同状态 hold 参考。64 条原历史精确复现，256 个原对角订单全部合法完整；零模型请求、零更新、不用 GPU、不读 Valid / Final OOS。

从状态 i 执行本组订单 j，再 hold 到原 21-session 终点得到金融 `Q_ij`；同状态直接 hold 的终局为 `H_i`，贡献 `q_ij=Q_ij−H_i`。风险、权益、换手和回撤历史随完整 prefix 保留。该目标是固定后续贡献，不是原策略继续行动时的 Q/advantage。

| 决策 j，每行 64 个 anchor | 合法 cell / 256 | 四订单完整 anchor | 完整 4×4 组 / 16 | 同状态同伴 | 同状态同订单同伴 |
|---|---:|---:|---:|---:|---:|
| 0，共同初态 | 256 | 64 | 16 | 64 | 57 |
| 5 | 219 | 36 | 5 | 21 | 10 |
| 10 | 211 | 29 | 3 | 0 | 0 |
| 15 | 192 | 27 | 2 | 0 | 0 |

后期仅 21/192（10.9%）有同状态同伴、10/192（5.2%）有同状态同订单同伴；更细键无法凭空增加同状态样本。146 个非法移植均因目标方向不适用于兄弟组合，保留 NA 和总分母，不修订单或补 hold；这是菜单支持问题，不是恢复验收失败。878 个合法 cell 含 32 个合法 next-open skip，均保留。

156 个四订单完整 anchor 中，104 个存在真实效果差异，原 family gate 将其中 67 个整行置零。下表评价同 anchor 四槽中心化 q，MSE 单位为金融 Y 平方，排序在 418 个非平局槽对上计算，预测 tie 计 0.5。

| 算子 | MSE | 排序一致率 |
|---|---:|---:|
| 全零参照 | 0.38443 | 50.00% |
| Plain + 原 family gate | 0.26147 | 70.22% |
| Family pool + 原 gate | 0.27165 | 65.43% |
| 具体订单 pool + 原 gate | 0.26147 | 70.22% |
| 具体订单 pool + 具体订单 gate / ungated plain | 约 0 | 100% |

末行来自确定性同状态同订单的代数恒等，不证明降噪或 RL 收益。观察到的排序损失为预测 tie 增多，未见反序。只细化 pool 键保留原 gate，结果恰等于 gated plain。

原终分可精确写作 `Y_i=H_i+q_ii+F_i`，F 是原后续行为与固定 hold 的差异，不是独立噪声。后期 H、q、F 都有可观尺度和交叉协方差；不能把分量平方能量直接加作误差归因。未来 hold H 只能作特权参照，本轮没有可部署状态 baseline；原 Y−H 仍含 F，未保证改善。

菜单相对贡献 `tau_i=Q_ii−mean_j(Q_ij)` 又不同于相对 hold 的 q。后期仅 10/48 个组支持完整矩阵；不同状态的 tau 未必零和，强制零和残差存在表示误差下界（该 10 组均值 MSE 0.00513）。这些是目标、表示与支持边界，不能直接推断原策略梯度有偏。代数检查最大误差 3.33e−15，独立重算通过。

## 同前缀菜单验证：真实输入与分支账务通过

10 月 18 日 00:39:32 SGT 完成，采集耗时16分43秒，采集及独立读数实际退出码均为0，工程结果PASS。[执行前方案](../2026-10-18_M2_SAME_PREFIX_MENU_PLAN.md)固定沿用全部256个 Train anchor，不按收益或分支结果换状态；[合同](../../runs/m2-same-prefix-menu-20261018-v1/contract.json)绑定135份源码与输入身份。

先独立 reset 并重放每个 anchor 前的全部原请求，包括同 session 的拒单和信息工具；保存实际 reset / step 返回的 Observation.messages，保留临时 last_action、error、order_rejection 及工具记忆。全部256份输入逐 token 匹配原始 request，长度3276–5070 tokens；后期16个带拒单反馈的状态原样保留。金融状态相同并不保证输入相同，金融风险状态与真实消息、工具 schemas、协议和 token IDs 分别核验，未使用 `_state_text` 重建替代观察。

菜单只读当前可见仓位与股票全集：20股、固定目标网格0%、10%、20%，按可见权重差大于1e−6决定买卖，另含 hold。全部菜单及输入包在第一笔新分支执行前[统一封存](../../runs/m2-same-prefix-menu-20261018-v1/snapshot_completion.json)，没有按未来成交或收益筛选，也没有把菜单注入原模型输入。

| 决策 session j | 状态数 | 菜单订单 | buy / sell / hold | 合法 next-open skip | 零成交，含 hold |
|---|---:|---:|---|---:|---:|
| 0 | 64 | 2624 | 2560 / 0 / 64 | 0 | 64 |
| 5 | 64 | 2769 | 2314 / 391 / 64 | 34 | 98 |
| 10 | 64 | 2738 | 2398 / 276 / 64 | 12 | 76 |
| 15 | 64 | 2761 | 2338 / 359 / 64 | 28 | 92 |
| 合计 | 256 | 10892 | 9610 / 1026 / 256 | 74 | 330 |

每状态菜单41–44项，全部10,892个请求合法推进，没有丢失候选或替换拒单。330项零成交正好包括256个 hold 和74个合法交易 skip，其余10,562项实际成交并有正费用。初态无卖出持仓，后期192个状态中185个有 sell 菜单；保留无 sell 状态，不强行补卖单。网格外的唯一原订单是 a173 的 `buy MSFT→0.3`，合法且原样进入控制分支，未映射成0.2。这是有限诊断菜单，不是原连续动作空间的全部支持。

实际执行10,892个菜单 first-step，加256个独立重复 buy 和256个原订单控制，共**11,404个独立构造的分支执行**；总计122,999次环境请求、11,660次环境构造，后两项包含原 prefix 重放。1,280个分支到达原21-session终点：768个预定不同菜单分支、256个重复 buy、256个原订单控制；其余10,124条记录只验证首步，不标为完整episode。四槽脚本探针仅有1,024个逻辑引用，不能把它或分支执行数当作新增模型轨迹或独立统计样本。

全部11,404次起始金融状态、真实消息与工具身份一致，费用、现金、股份和历史权益/收益/换手核对通过；256组重复 buy 的首步及终局结果精确一致。256个原订单首步 info 与历史原请求一致，固定 hold 终点匹配上一轮对角 Q；256个 hold 终点匹配上一轮 H。最大仓位与风险历史继续沿用原 prefix，没有重置。正反例夹具46项和独立读数的9类损坏负例通过；实际全量结果另有[检查收据](../../runs/m2-same-prefix-menu-20261018-v1/checks.json)。

本轮零模型请求、零更新、不用GPU、不读正式 Valid / Final OOS。它证明同状态分支、合法菜单与输入包具备后续取样条件，不证明模型能自主选出这些订单；确定性重复一致也不作降噪或收益结论。全量指标见[读数](../../runs/m2-same-prefix-menu-20261018-v1/readout.json)、[菜单覆盖](../../runs/m2-same-prefix-menu-20261018-v1/menu_coverage.csv)、[角色分层](../../runs/m2-same-prefix-menu-20261018-v1/branch_summary.csv)；费用跨互斥反事实分支的合计仅为工程计数，不是一个策略的成本。

## 原生同输入采样：重复支持存在，但原gate漏掉差异

10月18日11:21:38 SGT启动，11:25:35生成完成，11:32:05 CPU标签完成；GPU服务、CPU评估和独立读数实际退出码均为0。按[执行前方案](../2026-10-18_M2_SAME_PREFIX_SAMPLING_PLAN.md)及原[128槽清单](../../runs/m2-same-prefix-menu-20261018-v1/sampling_proposal.json)，固定source seed42、原兄弟0、八个Train窗口和session0/5/10/15，共32个前缀各4槽。模型是共同恢复SFT，不是原各轮M2策略；相同输入分别使用128个无碰撞seed，T1/p1、原自由工具输出，无菜单注入、约束解码、修复或补抽。

全部128请求原生ID0–127、输入token及返回映射一致，均自然EOS结束，无截断或空输出；总输出2667tokens，每次12–22。先保存16批完整raw再判读，独立tokenizer解码128/128一致。124条合法推进，含buy74、sell44、hold6；4条拒单保留原文、错误反馈及Y/q为NA，没有信息工具或其他非推进请求。每槽独立重放完整prefix，合法推进后hold至原21-session终点，共128次环境构造、2606次环境请求。

| 决策session j | 合法推进 / 32请求 | 完整四槽 / 8组 | 原family gate打开组 | 具体订单分歧组 |
|---|---:|---:|---:|---:|
| 0 | 31 | 7 | 0 | 3 |
| 5 | 32 | 8 | 2 | 2 |
| 10 | 29 | 6 | 2 | 2 |
| 15 | 32 | 8 | 0 | 4 |
| 合计 | 124 | 29 | 4 | 11 |

全32组的推进数k=0/1/2/3/4分别为0/0/1/2/29。a064/s0买HSI越出冻结股票池；a072/s3买GS到0.2不满足买入方向；a104/s1、s2卖HD到0不满足卖出方向。这3个partial组的主n4指标全部NA，不把剩余槽重心化。本轮只取首请求，没有测试这些拒单之后的恢复，不能由此推翻或替代此前恢复验收。

29个完整组的具体订单重复partition为18组[4]、7组[3,1]、4组[2,1,1]，没有[2,2]或四单全异。11个有订单分歧的完整组也都存在具体订单重复；但18组全同订单没有组内效果对比。全体合法子集有108/124个槽存在同anchor同订单同伴、16个singleton；同伴槽占全部128预定请求的108/128。单列118个交易订单：104有同伴、14个singleton；不能用全部重复率掩盖有效比较组有限，也不能跨anchor凑同伴。

**原family gate在7/11个具体订单分歧组中关闭，7组的固定hold贡献q也都不同**，分布为j0三组、j15四组。原gate打开的4组中，另有3个可池化同族cell包含不同具体订单且q不同。例如a024同为buy的HD→0.2与MSFT→0.2，固定hold贡献分别约−1.60676、+2.05602（单位为原金融Y），按buy合并会失去区分。这是目标条件下的订单差异证据，不是新的pooling MSE、排序或策略收益检验。

124条合法订单全部落在旧有限菜单内，118个buy/sell请求槽仅涉及GS/MSFT/HD，目标仅0或0.2；没有合法网格外输出。这只描述本批窄支持，不能说原自由动作空间已被菜单覆盖，也不能据此确定是何种训练先验导致。10次零成交包含6个hold和a044四槽合法next-open skip；后者buy MSFT到0.2时次开实际权重约0.20098998，保留合法标签q=0，不作为拒单或删样本。实际有成交的是114个请求槽。

输入、金融状态和首步返回观察独立核对通过；124个首步账务与上一轮封存同订单菜单记录完全一致，97个终局可用已有同订单终局全字段复核。相同状态同订单的first_info、Y/q及终局指标一致，属于确定性环境检查，不证明降噪。全部156份封存源码/模型文件哈希一致；整卡174次采样峰值34.208GiB，未越过min(95GiB,物理总量−2GiB)；既有holder恢复76GiB、paused=false、claim=None。生成审计838项、CPU证据审计3042项及独立统计审计通过，见[运行说明与收据](../../runs/m2-same-prefix-sampling-20261018-v1/README.md)。

## Gate单因素验证：候选实现与信号覆盖通过

10月18日12:53:22 SGT按[封存方案](../2026-10-18_M2_SAME_PREFIX_GATE_PLAN.md)完成，prepare/run实际exit0；只读取上一轮数据，不新增模型、环境或GPU调用，零优化器更新。设计已知道7个遗漏组，因此本轮是实现及覆盖验证，不是独立方法确认。新模块[same_prefix_gate.py](../../src/rewards/same_prefix_gate.py)与新[CPU runner](../../scripts/diagnostics/m2_same_prefix_gate_probe.py)隔离存放，旧生产训练和历史源码保留。

每组主输入仍为原fixed-hold金融Y，计算r_i=Y_i−fsum(Y)/4。四个算子只改gate，不做pool或学习baseline；η固定0.25，逐槽另存ηS，没有接入GAE、whitening或actor。原hold H只作center(Y)=center(Y−H)的检查，不传入候选。经济订单按原合法调用规范symbol和数值target，hold与abstain分开；不修复原拒单。

| 算子 | 有非零raw S的组 / 29完整组 | 非零槽 / 116主支持槽 | NA组 / NA槽 |
|---|---:|---:|---|
| zero | 0 | 0 | 3 / 12 |
| ungated plain | 11 | 44 | 3 / 12 |
| family-gated plain | 4 | 16 | 3 / 12 |
| order-gated plain | 11 | 44 | 3 / 12 |

非零按预先固定的abs(S)>1e−12定义。全32组/128槽仍保留，原124合法标签中仅116槽具有完整n4支持；3个partial组内另8个合法槽也不单独中心化。该12槽连zero参照均NA，与完整组的真零分开。

family→order只改变7组/28槽，没有丢失原有信号；原gate打开4组逐值不变。逐anchor开启状态匹配原独立读数，旧family plain中心化公式精确保持；实际输入排列等变，raw S组内和最大绝对误差6.66e−16，q/Y平移差最大4.44e−16。同订单固定hold标签完全相同，故本批order-gated与ungated逐值相等；18组全同订单仍为零。这些恒等检查和4→11覆盖已由已知标签及定义决定，不能作为降噪、无偏、pooling或收益优势。

67项核心和集成测试通过，含prefix/模型快照混用、非有限值、完整/partial标签、源码/输入SHA损坏、逐组gate错配及禁止覆盖；夹具均明确synthetic，不计新模型结果。预检发现旧rewards包初始化会间接导入环境，runner改为按文件隔离加载纯候选，实际进程模型/环境后端模块列表为空；旧包代码不改。原预检失败日志与修正后的通过日志均保留。冻结数据、新旧源码及独立重算的证据见[运行收据](../../runs/m2-same-prefix-gate-20261018-v1/README.md)。

## 随机策略后续pilot：154路径全部完成

10月18日21:37:21 SGT启动v3，21:43:14最后路径完成，21:43:18服务实际成功退出。按[新方案](../2026-10-18_M2_STOCHASTIC_CONTINUATION_PLAN.md)，固定共同恢复SFT为π、T1/p1、原自由工具协议和原21-session终点；全部32个旧Train前缀，48个合法唯一订单条件加29个未出现的hold条件，共77cell，每cell两个独立后续。154条原首动作按冻结原文执行，不计新增模型请求；之后均由π继续，hold条件因此是Q^π(h,hold)样本，不是全程hold H或V^π。

按用户要求优先GPU有效采样，模型加载与CPU快准备并行：复用已通过全哈希的SFT身份，启动只查merge、小配置和大小/mtime/stat；生产行情loader按20个symbol并行，省去重复9220文件raw/Pandas checksum。实际每条完整prefix的金融状态、真实观察、tokens、固定首动作账本及原终点仍匹配。v3总服务约5分57秒，模型就绪到最后返回约3分17秒，推理时GPU利用率实测100%，整卡峰值35.06GiB；CPU/GPU/服务/读数均exit0，holder恢复76GiB且claim清除。没有为此重复大规模CPU审计或正式Valid/OOS读取。

| 起始session j | cell / 路径 | 完整路径 | 模型请求 | 两次Y不同 / cell | 两次Y相等 |
|---|---|---:|---:|---:|---:|
| 0 | 18 / 36 | 36 | 753 | 18 / 18 | 0 |
| 5 | 19 / 38 | 38 | 578 | 19 / 19 | 0 |
| 10 | 19 / 38 | 38 | 380 | 18 / 19 | 1 |
| 15 | 21 / 42 | 42 | 192 | 13 / 21 | 8 |
| 合计 | 77 / 154 | 154 | 1903 | 68 / 77 | 9 |

1903次请求全部自然完整结束、无截断；1726次推进、177次拒单，无信息工具。96/154条路径出现额外非推进请求；136次买入方向、38次卖出方向、3次股票池外错误均由π继续处理，最多3次请求完成一个session，未触发4次上限，没有代选hold。154条全部到达原session20终点，有限Y完整；总环境请求3371包括1314次原prefix重放、154个固定首动作和1903个模型响应。输入8,870,710tokens、输出39,927tokens，每次输出12–22tokens。

77对均有标签，68对两次Y不同，绝对差中位0.60809666、均值0.73265425、最大3.83520386，单位是原金融Y；68个非零差最小约0.00095079，并非最后几位roundoff。9对Y相等中8对原文序列也相同，另1对请求序列不同但金融Y相等。这里只是本批两个draw的描述；不能把相等对称作零方差，或将配对差当标准差、收益百分点、Q精度、MSE/排序或方法胜负。32个状态来自8个旧M2窗口，77cell及两rep不独立于共享日期；π仍是SFT，不能外推为旧M2各轮后续。

原4个拒单首动作仍保留旧NA，不纳入本次合法/hold条件；本pilot不是原128槽的估计/参考实验，也不补齐3个partial组的主n4支持。两rep均属可行性重复，没有分别充当估计和独立参考。

启动过程的v1缺CPU命令run参数、v2直接比较Fill.datetime与旧JSON字符串，在0模型采样请求前失败；两份失败目录和原源码保留。v3只增加正确CLI参数和日期JSON归一，真实首买单验证旧info/账本完全一致，固定154路径与6904种子表不变。失败的GPU加载耗时不混入有效推理时间。实际证据见[完成收据与轻量检查](../../runs/m2-stochastic-continuation-20261018-v3/README.md)；未重新执行8GB权重或230文件历史全量哈希。

## 独立估计与参考：细分pool未优于零credit

按[预先固定方案](../2026-10-18_M2_CREDIT_COMPARISON_PLAN.md)，22:17:43 SGT完成864条全新后续：124个合法原slot各2条EST，共248条；77个订单/hold条件各8条REF，共616条。两侧seed namespace分离、39024个潜在请求种子无碰撞，且与pilot不重用。重复订单的EST保留原slot独立续写；REF共享前，76次同条件跨slot的首步后观察、账本与info均相同，运行时继续核验。共同恢复SFT为π，仍是32个旧M2 Train前缀、原自由工具接口、T1/p1和原21-session终点。

全部864路径COMPLETE，实际10788请求=3149 EST+7639 REF，低于39024硬上限；9756推进、1032拒单，最多3请求/session，仍用预定4次上限。所有生成自然结束、无截断，无补抽、代选hold或训练。64组rep/256槽主分母保留；29个原完整组各2次EST形成58组rep/232槽共同支持，3个原partial组的6组rep/24槽全部NA，zero也不补零。77个REF cell各8条全部完整。

参考先按具体cell求8次Y均值，再按原四slot的重复权重中心化τhat。以下均为原金融Y平方单位的逐槽MSE，不是收益百分点平方；η=.25仅作共同尺度派生，未混入GAE或whitening。

| 方法 | REF8 | REF前4 | REF后4 |
|---|---:|---:|---:|
| zero，附加S=0 | 0.03608 | 0.05261 | 0.04336 |
| ungated plain | 0.59301 | 0.60738 | 0.60246 |
| family gate plain | 0.12934 | 0.15857 | 0.12391 |
| order gate plain | 0.26920 | 0.28356 | 0.27864 |
| family pool，固定order gate | 0.05849 | 0.08520 | 0.05560 |
| exact order pool，同gate | 0.11992 | 0.13428 | 0.12937 |

zero只表示读数时附加credit S置零；这些路径的交易均由共同π生成。本轮没有用这些信号训练。family pool的本批公式与原family-gate pooled公式相等，但不代表复现了原M2完整训练。

| 分层，共同支持 | 槽数 | zero | order plain | family pool | exact pool |
|---|---:|---:|---:|---:|---:|
| 四槽同订单 | 144 | 0 | 0 | 0 | 0 |
| 同动作族、不同订单 | 56 | 0.08108 | 0.66051 | 0.08108 | 0.16292 |
| 不同动作族 | 32 | 0.11968 | 0.79579 | 0.28217 | 0.58429 |

因此zero的较低误差不只来自144个结构性零目标槽。family pool在上述前两层共200/232槽本来就输出零，较低误差不能直接解释成恢复了有效动作credit。粗池化在本批比细池化更接近有限参考，也不能反证粗key已正确描述动作效果。

成对loss差定义为候选减参照：exact−family pool为+0.06143，REF两半分别+0.04909/+0.07377；order−family gate plain为+0.13986，两半+0.12499/+0.15473。gate恢复覆盖并没有在本批转成更准信号。exact−family gate plain虽全8为−0.00942，两半−0.02429/+0.00545翻号，按方案记不可判。

exact相对order plain的−0.14928在两半完全相同，是同订单投影的逐样本恒等：τhat在同订单槽内相同，池化去掉的组内波动平方能量必然降低该目标的loss。四单相同时order gate的置零也符合τ=0结构；这些方向不能作独立降噪成功或REF精度证据。REF8仍非精确Q，分半同向不表示显著或可推广；本批也不证明真实动作价值全为零。目标仍是四动作中心化的Q代理，不是A^π/V^π、动作排序或策略收益。

GPU服务21:58:37—22:17:49 SGT，约19分13秒；模型就绪到最后生成约16分28秒。CPU快加载在21:59:42就绪，与GPU加载并行；沿用已有完整身份/行情收据，只核必要stat、真实prefix/观察/tokens，不重复8GB模型和全部行情checksum。最终读数1.2秒，CPU/GPU/service/readout实际exit0，整卡采样峰值34.88GiB；holder恢复76GiB、claim清除。[轻量终验](../../runs/m2-stochastic-credit-comparison-20261018-v1/final_execution_checks.json)及[独立小表重算](../../runs/m2-stochastic-credit-comparison-20261018-v1/m2_credit_actual_readout_review_20261018.json)通过，旧失败与旧run不改写。

## 当前π状态：固定收缩出现有限参考改善

按[取样前方案](../2026-10-18_M2_ONPOLICY_CREDIT_PLAN.md)，共同恢复SFT在全部8个已用Train窗口各生成一条全新父轨迹，从reset运行至原终点；不重放旧M2动作。固定j5/15首次进入且该session任何请求之前的真实完整观察作为16个anchor，全部取得；父后来失败不删已取得状态，未到达才NA。本批8条父轨迹均完整，共172请求=160推进+12拒单。新鲜的是策略轨迹与访问状态，窗口仍是已接触Train日期。

每状态四个独立单次原生head，64次中54合法、10拒单保留，不给head恢复或补抽；父轨迹和后续则仍允许原4请求/session内的实际恢复。54个合法slot各2次独立EST，共108条；同anchor、具体订单且首步后金融状态/完整消息核同后共享8次REF，24个cell共192条，无额外hold。300条后续全部COMPLETE；父/head/EST/REF各172/64/1059/2108请求，总3403，低于23744硬上限，四namespace种子与旧轮分离。全部3403生成自然完整、无截断；3054推进、349拒单，父/后续339次拒单均继续恢复，10个head拒单按方案不repair；最多3请求/session，无代选hold。

32组rep/128槽rep固定保留。11个四head合法状态×2EST为22组rep/88槽共支持；5个partial状态的10组rep/40槽，所有九方法含zero均NA。16状态均CAPTURED，无条件错配；24个cell各8REF完整。该主结果仅适用于四首动作均合法且续轨完整的条件子集，不能当成16状态无条件结果或88个独立市场样本。

REF8按具体cell求均值，再依原四slot重复权重中心化τ。下表单位为原金融Y平方；原六方法加预先固定α=.25/.5/.75乘raw exact residual，全部报告，不选最佳α。α为残差信号收缩，η=.25仍仅作两侧共同尺度派生；本轮没有GAE、whitening或训练。

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

α=.25相对zero成对loss差为−0.02916，两半−0.03917/−0.01915；α=.5为−0.03248，两半−0.05250/−0.01246。两个预设收缩均出现较低描述误差；这项对zero的比较依赖参考目标，不是exact相对plain的投影必降。但α=.5−.25虽全8为−0.00332，两半−0.01333/+0.00669翻号，不能选.5胜者。未缩exact及α=.75相对zero也分半翻号，按方案不可判。family pool相对zero增加0.02528，两半均正。

| 实际支持分层 | 原状态数 / 槽rep | zero | exact | α=.25 | α=.5 | α=.75 |
|---|---:|---:|---:|---:|---:|---:|
| 四单同订单 | 7 / 56 | 0 | 0 | 0 | 0 | 0 |
| 同族异订单 | 3 / 24 | 0.35255 | 0.28122 | 0.20915 | 0.14947 | 0.17349 |
| 不同动作族 | 1 / 8 | 0.24272 | 0.87883 | 0.35213 | 0.49462 | 0.67018 |

非零比较实际只有4个状态。三同族异订单状态的full8上，两个较强收缩各自改善；唯一跨族状态上全部收缩比zero差。因此不能将总体较低误差外推为跨族稳定有效，更不能事后按family选α或改gate。全同订单的56槽是结构性零目标控制，未提供额外方法区分。

作为**次要敏感度分析**，逐一移除11个支持anchor的全部两EST rep，full8的两个收缩对zero差均仍负；但移除p007/j15后第二REF半翻正（α=.25为+0.00552，α=.5为+0.03347）。这不是CI、显著性或新增完成门槛，反而说明少状态/有限REF下的稳定性仍有限。[独立小表重算与敏感度](../../runs/m2-onpolicy-credit-20261018-v1/m2_onpolicy_credit_actual_review_20261018.json)核对全部九公式、REF均值、原槽权重及分层通过。

新服务22:40:58—22:51:25 SGT约10分26秒；预热22:43:43完成，新三阶段驱动22:45:32封存启动、快准备22:46:09就绪，预热后存在约2分26秒接线/准备等待，未记成有效推理。随后父/head/branch共用同一worker，至22:51:20末次生成；读数0.18秒。重复8GB权重/全部行情哈希未运行；CPU/GPU/service/readout实际exit0，峰值35.01GiB，holder已恢复76GiB、claim=None。[轻量终验](../../runs/m2-onpolicy-credit-20261018-v1/final_execution_checks.json)通过；已完成的新驱动下次可在GPU加载时直接启动，避免本轮开发接线等待。

本轮目标仍是四个实际head中心化的有限Q代理，不是A^π、V^π或策略收益。zero仅指附加credit S=0，交易均由共同π生成；REF8/分半同向不代表精确Q或显著。与上一轮的状态、时点、head及seed不同，不能把两轮MSE拼成状态分布改变的因果对照。未更新模型、未开启正式Valid/Final OOS，旧结果与失败保留。

## 独立种子复验：同窗口的收缩信号方向复现

10/19先只核Train元数据：461个交易session形成441个21-session滑窗。与原8个窗口及既有12个Train-dev评估窗的完整区间之间至少留20个空交易session时，可选新窗口为0，详见[选择收据](../../runs/m2-onpolicy-credit-replication-selection-20261019-v1.json)。因此[取样前方案](../2026-10-19_M2_ONPOLICY_CREDIT_REPLICATION_PLAN.md)明确采用原8窗、新PARENT/HEAD/EST/REF namespace，检验同窗随机状态和采样重复性，不称新日期或未接触数据泛化。23744个潜在seed与旧69672个潜在seed均无交集，未用种子不回收补采。

14:58:01 SGT所有采样完成：8父轨迹COMPLETE、16个j5/15首次进入状态全CAPTURED；64个单次head中57合法、7拒单原样保留，不恢复head。25个同条件具体订单cell，114 EST+200 REF共314后续全部COMPLETE，25cell各8REF完整。父/head/EST/REF各174/64/1131/1966请求，总3335，低于23744上限。全部生成自然完整、无截断；2993推进、342拒单，其中父/后续335次拒单继续恢复，7head按单次规则不repair。最多3请求/session，预设4次上限不变，没有自动hold。

固定32组rep/128槽rep主分母，14个完整合法head状态各2EST构成28组rep/112槽共同支持；两个partial状态的4组rep/16槽九方法全NA。主结果是本批自己的完整合法组条件比较，没有与首批筛交集、删除失败或合并两批均值。16状态均取得，无条件错配；同订单6状态48槽为零目标控制，真正有动作分歧的是8状态64槽（5同族异订单、3跨族）。

方法、π、目标、时点、η及全部三α沿用首批：REF每cell八条完整Y均值按原四slot multiset重复权重中心化；原金融Y平方单位，非收益百分点平方。α只收缩raw exact residual，η=.25仅作两侧共同尺度派生。

| 固定方法 | REF8 MSE | REF前4 | REF后4 |
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

预先要求α=.25和.5各自相对zero的full8及两个REF半，共六个candidate−comparator loss差都小于−1e−12，才记为本批描述性方向共同复现。实际α=.25为−0.05468/−0.05149/−0.05787，α=.5为−0.07539/−0.06902/−0.08177；六项全部满足，判读为`PASS_DESCRIPTIVE_DIRECTION_ONLY`，没有挑一个α过关。此容差只判断符号，不是CI/p或显著性。

本批α=.5相对.25三panel均较低，但首批该排序在两半间翻转，因此没有跨两批的稳定最优α证据。α=.75及未缩exact本批也低于zero，仍完整报告，未据此改主判据。原family pool相对zero增加0.01328，两半均正；不能将收缩候选的结果写成原M2池化已获胜。

| 分层 | 状态数 / 槽rep | zero | exact | α=.25 | α=.5 | α=.75 |
|---|---:|---:|---:|---:|---:|---:|
| 同订单 | 6 / 48 | 0 | 0 | 0 | 0 | 0 |
| 同族异订单 | 5 / 40 | 0.21842 | 0.15030 | 0.13929 | 0.10156 | 0.10523 |
| 跨动作族 | 3 / 24 | 0.28826 | 0.33212 | 0.16497 | 0.13118 | 0.18690 |

两种预设收缩在本批两个异质分层的均值及REF两半均较低，不再只靠首批3个同族状态。但依旧只有8个动作分歧状态、8个共享Train日期窗口，不能外推至新日期或所有策略状态。预设次要的整anchor删除敏感度中，14次各去掉一个支持状态的全部两EST rep，两种α在full8及两个REF半均仍低于zero；这增加本批描述稳定性，不是置信区间或新增完成门槛。[独立小表重算](../../runs/m2-onpolicy-credit-replication-20261019-v1/m2_onpolicy_replication_actual_review_20261019.json)核对九公式、REF8/分半、原槽权重、分层及联合规则通过。

服务14:50:24—14:58:06 SGT约7分43秒。CPU快准备14:51:47已就绪，早于GPU预热14:53:08；随后父/head/branch连续复用同worker至14:58:01，预热至末次生成约4分53秒，没有本轮CPU接线等待。只复用原driver/runtime/worker及固定公式，新薄入口更换namespace和收据；未重复8GB模型或行情全量哈希。读数0.59秒，CPU/GPU/service/readout实际exit0，GPU峰值35.11GiB，holder恢复76GiB、claim=None，[轻量终验](../../runs/m2-onpolicy-credit-replication-20261019-v1/final_execution_checks.json)通过。

结果支持继续设计新的同前缀小规模学习试验，尚未验证训练收益。此处四head共享**完整模型可见前缀和状态**，原生产M2后期siblings常处于不同持仓/历史，不能把本轮exact-order均值直接粘进原collector。REF8是离线评价参考，不是训练可用oracle；EST和额外分支也不能在计算预算中消失。π本轮固定，未actor/critic更新，正式Valid/Final OOS未开启，旧结果和初版读数预检修复收据均保留。

## 当前下一步与证据

先固定可训练的同前缀分支方案，明确父轨迹/分支、head actor行、金融GAE和critic target如何对应。利用已有缓存做最小接线检查：训练端不读REF、不按合法head或未来Y悄悄改变π动作分布，partial/invalid保原分母，重复EST不能把同一head当多个独立父样本。若保留原异状态collector，应另验证状态混杂目标，不能套用本轮conditional Q证据。

小学习对照应包含相同α的未池化plain，区分收缩强度与精确池化作用。保留两种α时，候选矩阵可为zero、α=.25 plain/exact、α=.5 plain/exact五臂；这属于待冻结训练设计，不是已运行配置或选.5最佳。所有臂采用相同分支采集/真实parent及branch成本、共同actor/critic初始化、η、归一化顺序和更新/生成预算；REF仍只用于离线评价。先冻结小预算并验证真实训练接口，再考虑学习/收益判读，不追加同一复验种子追赢家。

后续同前缀collector实现、梯度掩码及`GAE+ηS→whiten/clip`尺度检查保留最小必要接线，能与GPU加载/工作并行的CPU处理不挡启动。若学习baseline，仍只用训练可见输入、episode/重叠horizon分折、时间purge及留出状态，未来oracle H不入训练。现有Train日期均属开发证据；正式Valid/Final OOS未开启。本轮3335请求已收口，没有自动加样或训练。

| 范围 | 证据入口 |
|---|---|
| 存档机制重算 | [状态/订单](../../runs/m2-action-pooling-mechanism-audit-20261014-v1/state_order_analysis.json)、[同批数学](../../runs/m2-action-pooling-mechanism-audit-20261014-v1/credit_math_analysis.json)、[费用 proxy](../../runs/m2-action-pooling-mechanism-audit-20261014-v1/cost_proxy_analysis.json) |
| 合成 oracle 与桥接 | [合同](../../runs/m2-credit-oracle-probe-20261014-v1/contract.json)、[指标](../../runs/m2-credit-oracle-probe-20261014-v1/metrics.csv)、[真实桥接](../../runs/m2-credit-oracle-probe-20261014-v1/bridge_replay.json) |
| 多状态实际结果 | [完整读数](../../runs/m2-state-order-oracle-20261017-v1/readout.json)、[完整性](../../runs/m2-state-order-oracle-20261017-v1/integrity_audit.json)、[独立重算](../../runs/m2-state-order-oracle-20261017-v1/independent_readout_audit.json)、[复现与最终核验](../../runs/m2-state-order-oracle-20261017-v1/README.md) |
| 同前缀菜单实际结果 | [采集完成](../../runs/m2-same-prefix-menu-20261018-v1/collection_completion.json)、[独立读数](../../runs/m2-same-prefix-menu-20261018-v1/readout.json)、[输入审计](../../runs/m2-same-prefix-menu-20261018-v1/snapshot_independent_audit.json)、[全量独立审计](../../runs/m2-same-prefix-menu-20261018-v1/final_independent_audit.json)、[复现与收据](../../runs/m2-same-prefix-menu-20261018-v1/README.md) |
| 原生同输入采样实际结果 | [读数](../../runs/m2-same-prefix-sampling-20261018-v1/readout.json)、[分组](../../runs/m2-same-prefix-sampling-20261018-v1/group_metrics.csv)、[生成审计](../../runs/m2-same-prefix-sampling-20261018-v1/generation_independent_audit.json)、[CPU证据审计](../../runs/m2-same-prefix-sampling-20261018-v1/evaluation_independent_audit.json)、[收据与复现](../../runs/m2-same-prefix-sampling-20261018-v1/README.md) |
| gate隔离候选实际结果 | [读数](../../runs/m2-same-prefix-gate-20261018-v1/readout.json)、[逐槽信号](../../runs/m2-same-prefix-gate-20261018-v1/slot_signals.csv)、[逐组信号](../../runs/m2-same-prefix-gate-20261018-v1/group_signals.csv)、[真实退出](../../runs/m2-same-prefix-gate-20261018-v1/execution_receipt.json)、[收据与复现](../../runs/m2-same-prefix-gate-20261018-v1/README.md) |
| 随机策略后续实际结果 | [完整读数](../../runs/m2-stochastic-continuation-20261018-v3/continuation_readout.json)、[逐路径](../../runs/m2-stochastic-continuation-20261018-v3/continuation_paths.csv)、[逐条件配对](../../runs/m2-stochastic-continuation-20261018-v3/continuation_cells.csv)、[退出/预算/归还检查](../../runs/m2-stochastic-continuation-20261018-v3/final_execution_checks.json)、[收据](../../runs/m2-stochastic-continuation-20261018-v3/README.md) |
| 独立估计/参考对照 | [完整读数](../../runs/m2-stochastic-credit-comparison-20261018-v1/credit_readout.json)、[逐槽](../../runs/m2-stochastic-credit-comparison-20261018-v1/credit_slots.csv)、[分层比较](../../runs/m2-stochastic-credit-comparison-20261018-v1/credit_strata_comparisons.csv)、[独立重算](../../runs/m2-stochastic-credit-comparison-20261018-v1/m2_credit_actual_readout_review_20261018.json)、[退出/预算/归还](../../runs/m2-stochastic-credit-comparison-20261018-v1/final_execution_checks.json)、[收据与复现](../../runs/m2-stochastic-credit-comparison-20261018-v1/README.md) |
| 当前π状态固定收缩 | [完整读数](../../runs/m2-onpolicy-credit-20261018-v1/onpolicy_readout.json)、[逐槽与NA](../../runs/m2-onpolicy-credit-20261018-v1/onpolicy_slots.csv)、[分层比较](../../runs/m2-onpolicy-credit-20261018-v1/onpolicy_strata_comparisons.csv)、[独立重算](../../runs/m2-onpolicy-credit-20261018-v1/m2_onpolicy_credit_actual_review_20261018.json)、[终验/归还](../../runs/m2-onpolicy-credit-20261018-v1/final_execution_checks.json)、[收据与复现](../../runs/m2-onpolicy-credit-20261018-v1/README.md) |
| 同窗独立种子复验 | [联合方向读数](../../runs/m2-onpolicy-credit-replication-20261019-v1/replication_readout.json)、[九方法原表](../../runs/m2-onpolicy-credit-replication-20261019-v1/onpolicy_readout.json)、[逐槽与NA](../../runs/m2-onpolicy-credit-replication-20261019-v1/onpolicy_slots.csv)、[独立重算](../../runs/m2-onpolicy-credit-replication-20261019-v1/m2_onpolicy_replication_actual_review_20261019.json)、[终验/归还](../../runs/m2-onpolicy-credit-replication-20261019-v1/final_execution_checks.json)、[复现说明](../../runs/m2-onpolicy-credit-replication-20261019-v1/README.md) |

## 2026-10-22：最小缓存 head 训练 demo 跑通

按用户明确要求改为 demo 优先，先复用 10/19 缓存 EST 的完整同前缀四槽组，α=.5、LR2e-6，执行四次全参数更新。新入口 [same_prefix_training_demo.py](../../src/agents/same_prefix_training_demo.py) 复用现成 chunked Adam；本次仅做 signed exact residual 的 head policy-gradient，不含 M1 的 GAE/critic、η 和白化，也未重新采集当前策略训练数据。

四个组、16个 head、4.02B 参数完成更新，更新本身合计31.6秒，CUDA分配峰值39.42GiB。BF16导出抽样6766个元素中314个发生变化；两个固定输入的贪心输出中，一个由买MSFT→.2变为卖MSFT→0。已导出的模型重新加载后，两条从缓存j15继续至原j20终点的后续全部完成，10次模型请求、零新增拒单。终局指标包含继承的旧prefix，不据此计算训练收益改善。结果见 [合并摘要](../../runs/m2-same-prefix-training-demo-20261022-v2/demo_result.json)、[更新日志](../../runs/m2-same-prefix-training-demo-20261022-v2/train_steps.jsonl)、[生成前后](../../runs/m2-same-prefix-training-demo-20261022-v2/before_after.json)、[checkpoint](../../runs/m2-same-prefix-training-demo-20261022-v2/checkpoint/)与[后续轨迹](../../runs/m2-same-prefix-training-demo-20261022-v3-eval/post_update_trajectories.json)。

v1参数抽样索引错误发生在更新前；v2完成更新/导出后，后续脚本导入路径报错；修正后v3-eval只补跑生成、exit0，未重复训练。旧失败状态与日志保留，holder已恢复76GiB、claim=None。此次完成的是可训练/可导出/可继续环境交互的最小缓存demo；下一步补当前π采集和共同GAE/critic后再做三组短对照。REF标签未参与训练，正式Valid/Final OOS未开启。

## 2026-10-24：在线 head GAE/critic 三组短对照完成

按 demo 优先推进，从共同恢复 SFT 起步，seed42、四个旧 Train 窗口0001/0009/0065/0073，比较同分支 S=0、α=.5 plain、α=.5 exact，η=.25，actor LR2e-6、critic LR1e-6。第一轮共享同一次 SFT 真实采样；第二轮每组用各自更新后的 actor 重新生成父轨迹、j15四head及每head两条EST。未使用REF。真实learned critic先估计预行动V，再用γ=λ=1的精确化简计算head GAE：剩余M1均分回报减V；共同无效调用罚为.05，S仍只用原金融Y。冻结旧V后先更新critic，再把U+ηαS统一白化，更新head actor。每轮新建Adam，仅保HF；本次不是完整StepPO轨迹更新复现。

| 组 | 第一轮后四窗平均净收益 | 第二轮后四窗平均净收益 | 第二轮平均金融Y |
|---|---:|---:|---:|
| S=0 | 4.27% | 3.47% | 3.180 |
| plain | 4.35% | 3.99% | 3.858 |
| exact | 4.19% | 4.52% | 4.491 |

共同SFT基线净收益3.73%。本批exact对S=0为+1.05个百分点，四窗3胜1负；对plain为+0.53个百分点，2胜2平。全部六轮各一次actor和critic更新完成，均有16个有效head；三组初始critic预测一致。实际共28条完整父/评估轨迹、128条完整EST、64个原生head，训练96次head曝光包含第一轮跨三组复用，不能当作96个独立head。总模型请求1250、输入5804698tokens、输出26350tokens；模型子进程累计墙钟34.5分钟（含加载/编译/导出）。行情对象缓存约1.31MB，避免重复预处理；holder最终恢复76GiB、claim=None。

这是一seed、四个已使用开发窗的正向学习趋势，不是日期泛化、稳定方法优势或严格等算力结论。首次控制器事件参数同名冲突在采样前退出，失败v1保留；修复后v2三组两轮全完成。证据见[结果表](../../runs/m2-same-prefix-online-demo-20261024-v2/results.csv)、[逐窗表](../../runs/m2-same-prefix-online-demo-20261024-v2/window_results.csv)、[完成摘要](../../runs/m2-same-prefix-online-demo-20261024-v2/completion.json)和[详细结果/模型路径](../../runs/m2-same-prefix-online-demo-20261024-v2/results.json)。下一项只换seed复跑同配置，不先改α、SFT或增加方法；正式Valid/Final OOS仍未开启。

## 2026-10-24：第二seed完成，首批学习优势未复现

seed43沿用共同恢复SFT、原四Train窗口、三组两轮与全部超参，未调α/η或修改训练数学。第一轮共享初批，第二轮重新采样。仅增加本次模型保留开关：第一轮actor/critic各自完整加载到GPU后释放原文件，最终评估完成后释放本次最终critic，保留三个最终actor；seed42模型不动。

| 组 | seed43第一轮净收益 | seed43第二轮净收益 | seed43第二轮金融Y |
|---|---:|---:|---:|
| S=0 | 3.710% | 3.763% | 3.786 |
| plain | 4.360% | 3.793% | 3.817 |
| exact | 3.766% | 3.506% | 3.410 |

共同SFT基线4.025%。exact−S=0为−0.25696个百分点，四窗1胜2负1平；exact−plain为−0.28707个百分点，0胜2负2平。seed42对应为+1.05254/+0.53442个百分点，故首批正向优势未跨seed复现。两seed平均差+0.39779/+0.12367个百分点仅作描述，不能用均值掩盖方向翻转；第一轮exact对plain两seed都低。两seed仍共享四个开发日期窗，八个seed×窗口配对不是八个独立市场样本。

本批三组六轮全部完成，actor/critic各6次全参数更新，28条父/评估轨迹与124条EST后续全部完整。plain第二批0065窗口的slot0/2为单次拒单，按既有完整组规则四槽整组NA，未补抽、未填零；该轮支持12head，其余五轮各16head，训练曝光共92。64个原生head中62合法，第一轮共享数据不能重复当作独立head。实际1243请求、5781334输入token、26300输出token，初批及模型阶段合计2043.0秒约34.1分钟（含加载/编译/导出）。仅保留最终三个actor与原始记录，中间模型释放有回执，GPU已归还自有holder。

证据：[seed43逐轮结果](../../runs/m2-same-prefix-online-demo-seed43-20261024-v1/results.csv)、[训练与模型记录](../../runs/m2-same-prefix-online-demo-seed43-20261024-v1/results.json)、[两seed汇总JSON](../../runs/m2-same-prefix-online-demo-seed43-20261024-v1/cross_seed_summary.json)与[CSV](../../runs/m2-same-prefix-online-demo-seed43-20261024-v1/cross_seed_summary.csv)。运行与小规模路径检查通过，未补做大审计；仍是head范围demo、每轮新Adam、无REF和正式Valid/OOS。建议下一批先扩到八个Train窗口、保持两轮及其余配置，检查更广状态下是否还值得投入；不继续在四窗上追种子或扫参数，该批次尚未启动。

## 2026-10-24：八窗扩大批次完成，当前exact仍未改善收益

经用户同意，仅把窗口扩大到0001/0009/0017/0025/0065/0073/0149/0157，沿用seed42、共同恢复SFT、三组两轮、α=.5/η=.25和原actor/critic学习率。新增0017/0025/0149/0157依据既有Train八窗名单与日期覆盖选择，未看方法收益；仍是已接触开发窗，不称未接触日期泛化。collector/trainer/controller扩展支持4/8窗，经济订单gate、四head完整组、EST2、GAE/奖励/白化/Adam、原终点和拒单处理不变。

| 组 | 第一轮八窗平均净收益 | 第二轮八窗平均净收益 | 第二轮金融Y |
|---|---:|---:|---:|
| S=0 | 2.56857% | 2.37712% | 3.59615 |
| plain | 2.24625% | 2.49309% | 3.67445 |
| exact | 2.03155% | 2.10079% | 3.26809 |

共同SFT基线2.56907%。最终exact−S=0为−0.27633pp（1胜4负3平），exact−plain为−0.39230pp（3胜5负）。原四窗/新增四窗分层对S=0分别−0.41399/−0.13867pp，对plain分别−0.15255/−0.63205pp；当前exact配置在两层都没有收益优势。三组最终都低于SFT，不因首批四窗seed42的正向结果继续扩大这一配置，也不把本次负结果推广成所有细粒度credit或项目的上限。

三组六轮actor/critic各6次更新全部完成；各轮32计划head中24支持、8整组NA，共144次训练head曝光。初批29合法/3拒单，第二批S=0为28/4、plain/exact各29/3；总128独立原生head、115合法、13拒单，无补抽。56条父/评估轨迹及230条合法EST后续全部完整，2418真实请求、11264410输入token、50750输出token。每轮实际支持6个状态组，订单分歧仅1–2组，八个计划窗口不能视作八个有效credit比较。模型阶段合计2232.0秒（37.2分钟，含共同初批和加载/编译/导出），不等于GPU计算内核耗时。

三个最终actor已保留，holder最终76GiB、paused=False、claim=None。仅有14GiB空间时，与GPU初采并行复用现成`parameter_grid_archive.pack_model`，将旧seed42三组首轮actor做无损XOR归档，原22.48GiB权重压为1.24GiB，净释放21.24GiB；逐字节恢复hash通过后仅移除其原safetensors，原metadata、实验数据、三个旧最终actor均保留。原目录的`LOSSLESS_ARCHIVE.json`记录依赖共同SFT、归档位置与恢复API；需保留该共同SFT。未做全仓库清理或重复审计。

证据：[逐轮结果](../../runs/m2-same-prefix-online-8window-demo-20261024-v1/results.csv)、[逐窗差](../../runs/m2-same-prefix-online-8window-demo-20261024-v1/window_results.csv)、[八窗分层读数](../../runs/m2-same-prefix-online-8window-demo-20261024-v1/eight_window_readout.json)、[完成摘要](../../runs/m2-same-prefix-online-8window-demo-20261024-v1/completion.json)、[归档记录](../../runs/m2-same-prefix-storage-20261024-v1/storage_summary.json)。新增4/8窗与分母小检查、真实全流程通过，未新增正式Valid/OOS或REF。下一步建议先做稍长、固定小预算的S=0学习曲线，检查共同head训练是否能带来改善，再讨论credit；不先扫α或再扩大当前exact，该下一批尚未启动。

## 2026-10-24：S=0八轮学习曲线完成，尚未持续改善整段收益

用户同意延长共同基线后，预定seed42、八个原Train窗、S=0单臂八轮，actor LR2e-6、critic LR1e-6、γ=λ=1、head范围和每轮新Adam均不变。第一轮从共同恢复SFT起步，其后继续前轮actor/critic。首批直接复用此前八窗common_initial，不重采；第1–7轮更新后新采八条父迹及四head/EST2供下轮训练，第8轮只评估八父迹，全部八轮照报，未选最好模型。

| 轮 | 八窗平均净收益 | 支持head/32 |
|---|---:|---:|
| 共同SFT | 2.56907% | — |
| 1 | 2.70101% | 24 |
| 2 | 1.98550% | 20 |
| 3 | 2.00005% | 20 |
| 4 | 2.00000% | 24 |
| 5 | 1.81283% | 24 |
| 6 | 2.38502% | 20 |
| 7 | 2.13947% | 16 |
| 8 | 2.28222% | 20 |

第8轮对SFT为−0.28685pp，3窗改善/5窗下降；原四窗−0.28920pp、新增四窗−0.28449pp。平均金融Y从3.72600变为3.43332（−0.29268）。只有首轮高于SFT，其余七轮都低；这条曲线未建立持续学习收益，结论限于当前单seed、已接触Train窗、head-only和重置Adam实现，不推广成RL或所有credit无效。

八次actor及八次critic更新全部完成，支持head共168/256曝光，各轮缺失组保持NA。新生成64条父/评估轨迹与340条EST均COMPLETE；七个新head批次224槽中170合法、54拒单，没有补抽。初批另有历史32槽/29合法/58EST，未重复计为新增样本；状态随策略变化，拒单比例不能单独归因于策略遗忘。实际新增3312请求、15498863输入token、69440输出token，初批461请求复用不计新增；模型阶段2885.8秒（48.1分钟，含导入/加载/编译/导出）。

两个针对实际疑点的轻量核查随GPU运行并行完成，未挡后续轮次：首轮与上次八窗S=0的超参及24行head优势（含tokens/EST/旧V/白化）相同，但更新数值已有差异，不能把复跑差异称为纯采样噪声；没有将旧首两轮拼接到本曲线。第3/4轮接近2%的均值来自不同窗口正负收益抵消，两轮分别有149/146次成交、约2.85万/2.76万费用；逐窗净收益与终值净资产一致，未见固定2%回报、fallback或abstain退化，不据均值判定策略收敛。

仅保本次第8轮actor，中间模型加载消费后退役，最终critic评估后释放。另将旧四窗seed42的S=0第2轮actor通过现成XOR归档无损压缩，原约7.49GiB→0.34GiB，完整恢复hash验证后释放原safetensors；原目录`LOSSLESS_ARCHIVE.json`与[存储记录](../../runs/m2-same-prefix-storage-20261024-v2/storage_summary.json)提供恢复位置/API，须保留共同SFT。其它旧最终actor不改，原始实验记录均保留，GPU已归还自有holder。

证据：[全曲线图](../../runs/m2-s0-learning-curve-20261024-v1/learning_curve.png)、[逐轮CSV](../../runs/m2-s0-learning-curve-20261024-v1/curve_results.csv)、[逐窗曲线](../../runs/m2-s0-learning-curve-20261024-v1/window_curve.csv)、[分层读数](../../runs/m2-s0-learning-curve-20261024-v1/learning_curve_readout.json)、[完成摘要/最终模型](../../runs/m2-s0-learning-curve-20261024-v1/completion.json)。新入口[Ruff/CLI及全八轮流程](../../scripts/diagnostics/run_s0_learning_curve.py)通过，REF及正式Valid/OOS未使用。

下一项建议只做固定相同SFT j15状态的短后续对照，比较SFT与第8轮actor。当前只训练j15一个head，却从j0整段重跑评估；需要先区分局部更新无效与早期策略变化带来的整段差异。这是待检验解释，并非已确定根因；暂不继续加轮数或扩大exact，该短对照尚未启动。

## 2026-10-24：固定j15状态短对照完成，局部小幅正向且拒单更多

用户同意后，冻结原八窗SFT初批的八个j15快照，各四条新后续，比较共同SFT与S=0八轮最终actor，共32对/64条。CPU精确回放原prefix，核对观察消息、prompt token、可见状态、净资产曲线和风险历史；两个模型均使用相同`fixed-j15/{episode}/r{rep}`种子namespace、seed42/round0。固定状态后由当前模型交易到原j20，保留4请求/session的模型恢复规则；不补抽失败路径、不自动hold、无训练或新父轨迹。

| 模型 | suffix平均净收益 | 完整终局净收益 | 金融Y | 首动作拒单 | suffix总拒单 | 新请求 |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 0.69375% | 2.44337% | 3.57543 | 5/32 | 17 | 177 |
| S=0第8轮 | 0.72627% | 2.47680% | 3.61854 | 5/32 | 32 | 192 |

主要读数为suffix净收益差+0.03252pp，完整终局净收益差+0.03343pp、Y差+0.04311。suffix以固定prefix最后记账净资产为分母，完整终局收益以原初始资金为分母，两种尺度分别报告。八窗均值4胜2负2平，原四窗/新增四窗差+0.04158/+0.02346pp；窗口0073的+0.25238pp贡献了均值增量的大部分。新SFT与训练模型都是本批新采样，不将其均值与此前整段重跑的2.569%/2.282%相拼。

每路径suffix交易费用均值1148.36→889.42，费用减少258.94；在各路径共同初始资金100万下，终值净资产均值差约334.27。费用差与这点收益增量处于相近量级，不能据小幅净收益差单独确认资产选择改善。首动作拒单数相同，后续总拒单17→32，虽64条均恢复到原终点，本批合法性与生成成本未改善。

64/64轨迹及32/32配对全部完整，全部配对初始账本净资产/prompt/financial-state身份相同；无缺失均值或填零。实际新增369请求、1726651输入token、7623输出token、49次suffix拒单，模型阶段258.8秒约4.31分钟（含加载/预热/回放），零模型更新、无模型导出或清理。GPU holder最终76GiB、paused=False、claim=None，REF与正式Valid/OOS未使用。

证据：[32对逐项CSV](../../runs/m2-fixed-j15-policy-probe-20261024-v1/paired_results.csv)、[窗口/分层读数](../../runs/m2-fixed-j15-policy-probe-20261024-v1/probe_readout.json)、[执行与成本](../../runs/m2-fixed-j15-policy-probe-20261024-v1/probe_result.json)、[SFT轨迹](../../runs/m2-fixed-j15-policy-probe-20261024-v1/sft/trajectory_results.jsonl)、[训练模型轨迹](../../runs/m2-fixed-j15-policy-probe-20261024-v1/trained/trajectory_results.jsonl)。复用原collector新增可选fixed-prefix分支，默认在线采样行为保留；Ruff/CLI和两模型真实完整流程通过，未加大审计。

结论收窄为：本批原始SFT j15状态上的局部收益只有微小正向，整段收益仍未改善；训练/评估对应关系值得检查，但尚未确定为根因。后续八窗、单seed、两轮的最小多步/整轨迹S=0训练已完成，见下节。

## 2026-10-24：多步S=0两轮完成，整段收益尚未改善

用户同意后，从共同恢复SFT起步，固定原八Train窗、seed42、actor LR2e-6、critic LR1e-6和原奖励。新[训练入口](../../src/agents/multistep_s0_train.py)仅读取完整role=parent轨迹，每次真实请求均作为actor/critic训练行，包括拒单与恢复；head/EST/REF全部排除。首批复用历史八窗初批的169个父请求（9次拒单）；第一轮更新后从j0重新采八条父迹，171个请求（11次拒单）用于第二轮。两轮均一次actor、一次critic更新，Adam每轮重建，权重逐轮继续。

对含T个真实请求的轨迹，金融Y均匀分摊到T个请求，最后一份校正浮点余数；拒单另扣原0.05惩罚。每行G为该请求至终点的奖励后缀和，γ=λ=1，U=G−本轮更新前冻结的learned V；全部请求共同按sample std白化。critic拟合逐请求G，actor仍用原response token平均logprob目标、全参数累计一次更新；不是完整PPO内循环或StepPO复现。不完整轨迹整条NA，本批八窗全部完整。

| 模型 | 八窗净收益均值 | 平均金融Y | 拒单 / 实际请求 | 每窗交易费用均值 |
|---|---:|---:|---:|---:|
| 共同SFT | 2.56907% | 3.72600 | 9/169（5.33%） | 5095.06 |
| 第1轮 | 2.50520% | 3.62479 | 11/171（6.43%） | 5170.68 |
| 第2轮 | 2.52528% | 3.68574 | 12/172（6.98%） | 5227.41 |

固定第二轮对SFT为−0.04378pp、Y差−0.04025，八窗2胜5负1平；原四窗+0.07143pp、新增四窗−0.15899pp。第2轮较第1轮略回升，仍低于基线，拒单率与交易费用也未改善。两轮critic训练MSE分别7.6026→6.8565、7.0199→6.3344；可确认真实拟合和参数更新，不能以此代替学习收益。

两次actor及两次critic更新均COMPLETE，共训练340行、含20次拒单。新16条父轨迹全部在原j20完成，343请求、1586036输入token、7187输出token，23次拒单均在原4请求/session上限内由模型恢复；无补抽、自动hold、缺失填零或最好轮次选择。历史源批总461请求中只用169个父请求，新增初批成本为0。模型阶段累计1466.55秒（24.44分钟，含加载、编译与导出）；仅保留本次第二轮最终actor，中间actor/critic加载消费后退役、最终critic评估后释放。systemd服务success，GPU holder恢复76GiB、paused=False、claim=None，REF与正式Valid/OOS未使用。

运行前将旧四窗seed42 plain第2轮最终actor无损归档，原8,044,982,000字节→329,868,679字节，验证精确恢复后释放原safetensors，约腾7.19GiB；[存储记录](../../runs/m2-same-prefix-storage-20261024-v3/storage_summary.json)和原目录`LOSSLESS_ARCHIVE.json`保留恢复方式，共同SFT须保留。至此旧四窗首批六actor中的三个首轮及S=0/plain最终actor为无损归档，exact最终actor仍可直接加载；原始实验数据与历史失败不变。

结论限于单seed、八个已接触Train窗、两次完整请求更新：扩展训练覆盖已跑通，尚未出现收益或合法性改善。新模式没有head/EST、每状态只用实际父请求，与旧head-only曲线不能直接构成严格单因素比较，也不能据两轮否定多步RL。沿用demo优先节奏，后续继续用小规模训练检验学习信号，暂不扩大exact或扫α。

证据：[逐轮CSV](../../runs/m2-s0-multistep-demo-20261024-v1/results.csv)、[逐窗CSV](../../runs/m2-s0-multistep-demo-20261024-v1/window_results.csv)、[分层、拒单及费用](../../runs/m2-s0-multistep-demo-20261024-v1/multistep_readout.json)、[完成摘要/最终actor](../../runs/m2-s0-multistep-demo-20261024-v1/completion.json)、[两轮控制器](../../scripts/diagnostics/run_multistep_s0_demo.py)。新训练入口Ruff、CLI、奖励守恒与整轨迹NA快检通过，完整GPU流程成功，未新增长审计。

## 2026-10-25：持久FP32/Adam、终局Y与同窗RLOO完成16次更新

用户同意先修训练基础后，以共同恢复SFT、seed42、四个原Train窗0001/0009/0065/0073起步，每窗每批4条完整父轨迹。金融Y仅在终局给，拒单即时扣0.05；每请求的reward-to-go减去同窗其余三条轨迹的完整回报作留一基线，全部请求共同白化。每批四个不相交轨迹minibatch，每个含每窗一条轨迹，完整请求和拒单均训练；使用旧token logprob及clip0.2，actor LR2e-6，固定16次optimizer.step。每请求权重按固定20个推进session归一化，不用实现后才知道的请求长度稀释拒单。无learned critic、head/EST/REF或M2局部项。

FP32主权重与Adam动量连续保留，BF16仅作独立推理副本。v3将完整分片存项目磁盘，真实恢复计数4→8→12→16通过；每批结束保存约46.4GiB续训状态，最终actor也落盘。参数探针确认FP32微小更新能积累，不能将其视为全模型随机样本或收益改善证明。

| 更新数 | 平均净收益 | 平均金融Y | 拒单 / 请求 |
|---|---:|---:|---:|
| SFT / 0 | 3.48148% | 3.40152 | 25/345 |
| 4 | 2.79211% | 2.64309 | 24/344 |
| 8 | 3.05951% | 2.97074 | 23/343 |
| 12 | 3.19202% | 3.17041 | 36/356 |
| 16 | 4.00232% | 4.13039 | 43/363 |

预定终点相对SFT为+0.52084pp、Y+0.72886；四窗3升1降，0001/0009/0065/0073分别+0.19987/−0.52310/+0.48430/+1.92229pp。0073贡献均值+0.48057pp，去掉它其余三窗仅+0.05369pp，不能确认稳定或全面改善。拒单率7.25%→11.85%，合法性回退，每轨迹费用4817.59→5010.98。16对最终初始prompt与首请求种子全部匹配，全部到原j20；中间点用新种子供下批训练，仅作描述。初批也训练、窗口已接触，仍是训练内单seed证据，不称日期泛化。

16次actor、0次critic更新完成，1388个训练请求含108次拒单。新增64条父/评估轨迹全部完整，1406请求=1280推进+126拒单，6482133输入token、29299输出token；历史初批345请求只复用，不计新增。模型阶段累计3742.72秒约62.38分钟（含加载、保存、评估和失败启动），全曲线照报，未选最好模型或追加32次。GPU holder已恢复76GiB、paused=False、claim=None。

执行问题保留：v1完成4次更新后因整份CPU模型、CPU Adam及RAM文件超过64GiB容器内存限制而保存停滞，3600秒超时，无可用checkpoint；v2在参考概率阶段按用户要求暂停，0次更新。用户另行授权大清扫后，删约298GiB旧失败状态/弃用权重，保留基线、压缩档、近期actor与轨迹，v3改GPU Adam与磁盘分片。首轮评估首次登记GPU worker时15秒IPC超时，实际随后登记成功；该次只写启动标记、未生成请求。原失败控制器记录快照保留，显式续跑只重试未生成阶段、登记等待改60秒，未重复前4次训练或补抽模型失败轨迹。

多个基础项同时变化，不能据本批确定各项因果贡献。结论是新基线终于有值得追的终点正向信号，但增量集中、拒单回退；按原计划16次收尾，下一步固定配置扩八Train窗、seed42/43验证收益和合法性，暂不直接加入M2或扫reward。第二步尚未启动。

证据：[完整读数](../../runs/m2-rloo-terminal-demo-20261025-v3/rloo_readout.json)、[曲线图](../../runs/m2-rloo-terminal-demo-20261025-v3/learning_curve.png)、[曲线CSV](../../runs/m2-rloo-terminal-demo-20261025-v3/curve_results.csv)、[逐窗CSV](../../runs/m2-rloo-terminal-demo-20261025-v3/window_results.csv)、[16对配对](../../runs/m2-rloo-terminal-demo-20261025-v3/paired_results.csv)、[完成摘要/模型路径](../../runs/m2-rloo-terminal-demo-20261025-v3/completion.json)、[清理清单](../../runs/disk-cleanup-20261025/cleanup_summary.json)。新训练、收集、控制与读取入口通过Ruff及必要的小规模奖励/NA/精度恢复检查，完整GPU流程通过。

## 2026-10-25：八窗、双seed RLOO复验完成，收益方向重复，拒单方向不一致

用户同意第二步后，固定第一步的终局金融Y、即时拒单−0.05、同窗四轨迹RLOO、LR2e-6、PPO clip0.2和持久FP32/Adam，扩至既有八Train窗0001/0009/0017/0025/0065/0073/0149/0157。seed42/43分别从共同恢复SFT开始，各采32条独立初批、四批各4次有效更新，预定终点均为16次；每次minibatch含每窗一条轨迹，完整请求及拒单恢复请求均训练，无critic、head/EST/REF或M2项。后台systemd服务在VPN断线期间持续运行，22:41（北京时间）完成。

| seed | SFT→第16次平均净收益 | 差值 | 平均金融Y | 拒单 / 请求 | 每轨迹费用均值 |
|---|---:|---:|---:|---:|---:|
| 42 | 2.29721%→2.49571% | +0.19851pp | 3.55092→3.95550 | 55/695→26/666 | 4855.18→5241.26 |
| 43 | 2.30193%→2.83938% | +0.53745pp | 3.57265→4.36610 | 67/707→81/721 | 4725.79→4433.22 |

seed42/43的原四窗差分别+0.14707/+0.46598pp，新增四窗+0.24995/+0.60893pp，去0073后七窗仍+0.17534/+0.44387pp；0073对总体均值的贡献占22.7%/27.7%，不再主要依赖单窗。逐窗4升4降、7升1降；0001均下降，0009/0073/0149/0157均上升，0017/0025/0065方向翻转。拒单率7.91%→3.90%、9.48%→11.23%，费用一增一减，因此不能把正向收益统一解释为学会少拒单或节约费用。

完整0/4/8/12/16次净收益曲线：seed42为2.29721%/2.01140%/1.95014%/2.23285%/2.49571%，seed43为2.30193%/2.44969%/2.24858%/2.52510%/2.83938%。最终64对的初始prompt和首请求种子全部一致、全部到原j20；中间点用新namespace，只作描述。没有选择最好轮次、追加更新、补抽失败路径、提高交互上限或自动hold。两seed全部320条初批/评估路径完整，窗口及初批均参与开发/训练，不称日期泛化，正式Valid/OOS保持未使用。

合计32次actor更新、0次critic更新，256条完整训练轨迹曝光、5582请求含462次拒单。实际新增6969请求、32306789输入token、146353输出token；各seed初批与四个更新后采样批次各计一次，评估兼下一批训练的数据不重复计成本。模型阶段累计约163.56分钟。两份FP32/Adam续训状态各约46.41GiB和最终BF16 actor保存在项目磁盘，manifest均COMPLETE/16次、引用文件齐全，GPU已恢复自有holder76GiB、paused=False、claim=None。

结论：新RLOO基线在八个训练窗口内的最终收益方向得到双seed重复，足以继续demo阶段的M2增量对照；合法性改善没有重复，也未建立稳定日期泛化。按原预算结束基线，不先延长或扫reward。第三步仍为四窗、seed42、固定α0.5/η0.25的RLOO/plain/exact小对照，局部credit仅接到少量j5/j15父请求，其余仍训练整段；尚未启动。

证据：[双seed汇总](../../runs/m2-rloo-eight-window-20261025-v1/seed_summary.csv)、[完整读数](../../runs/m2-rloo-eight-window-20261025-v1/comparison_readout.json)、[双曲线](../../runs/m2-rloo-eight-window-20261025-v1/learning_curves.png)、[逐窗曲线](../../runs/m2-rloo-eight-window-20261025-v1/window_results.csv)、[64对配对](../../runs/m2-rloo-eight-window-20261025-v1/paired_results.csv)、[完成摘要](../../runs/m2-rloo-eight-window-20261025-v1/suite_completion.json)。入口支持八窗/两seed并保留原四窗行为，Ruff、CLI及必要的NA/身份/恢复检查通过，实际完整GPU流程通过；未增加重复审计。

## 2026-10-26：RLOO＋plain/exact各16次完成，池化改善合法性但未增进收益

第三步固定四个原Train窗0001/0009/0065/0073、seed42、每窗每批4条完整父迹，从共同恢复SFT起步。基础RLOO直接复用已完成v3的16次曲线，避免重复GPU训练；plain/exact各新增16次，初批父迹与RLOO完全共用，只新增局部分支。终局Y、拒单即时−0.05、持久FP32/Adam、LR2e-6、PPO clip0.2和完整请求训练均保持，不使用learned critic。

每窗预定replica0在j5/j15首请求前取精确前缀，共8个anchor/批。slot0使用父实际请求，slot1..3为同一状态新采native动作；每个合法槽EST2至原j20。局部Q为终局金融Y减该head及后续拒单惩罚；plain用slot0Q减四槽均值，exact用slot0同经济订单池化Q减相同的四槽均值，保留原槽重复度。固定α0.5、η0.25，以0.125S加到对应父请求的RLOO优势后共同白化。任何缺槽/非法head/缺EST只使该anchor局部项NA/S=0，完整父请求仍训练，不补抽或把EST当actor样本。

| 模型 | 第16次平均净收益 | 平均金融Y | 拒单 / 父请求 | 每轨迹费用均值 |
|---|---:|---:|---:|---:|
| 共同SFT | 3.48148% | 3.40152 | 25/345（7.25%） | 4817.59 |
| RLOO（16次复用） | 4.00232% | 4.13039 | 43/363（11.85%） | 5010.98 |
| RLOO＋plain | 2.89390% | 2.78649 | 36/356（10.11%） | 4615.96 |
| RLOO＋exact | 3.27310% | 3.13301 | 11/331（3.32%） | 5418.88 |

exact比plain高+0.37920pp，但对SFT/RLOO分别−0.20837/−0.72922pp；去0073后相对SFT仍−0.48574pp。exact四窗2升2降，0001/0009/0065/0073相对SFT分别−0.61373/+0.25031/−1.09380/+0.62373pp，0065退化最重。plain对SFT/RLOO分别−0.58757/−1.10842pp，去0073后对SFT仍−0.82787pp。最终各组16对初始prompt与首请求种子匹配，全部到原j20，没有提高交互上限、自动hold或挑选最好轮次。

完整0/4/8/12/16次净收益曲线：plain为3.48148%/3.10259%/2.90451%/3.23399%/2.89390%，exact为3.48148%/3.41661%/3.13679%/2.97337%/3.27310%。中间点使用namespace1/2/3，最终复用初批namespace0，曲线不解释为严格同种子趋势。初批参与训练、窗口已接触，仍属开发证据，正式Valid/OOS未使用。

plain支持7/6/6/6个anchor（共25/32），非零项6/3/5/3，共17/1412训练请求；exact支持7/4/7/8（26/32），非零项6/3/5/5，共19/1380。每预定anchor的平均绝对实际修正分别约0.0224/0.0131/0.0418/0.0102及0.0122/0.0223/0.0263/0.0124。支持组均有重复订单，同订单不同输出片段数为0，因此本批主要检验重复订单的估值池化，不能写成文本碎片化改善的证据。局部项实际生效但稀疏，不据负结果断言全部局部credit无效。

新增32次actor更新、0次critic更新，2792训练请求含232次拒单；基础RLOO的16次更新/1388请求另列复用。新增128条父迹和398条实际EST全部完整，旧初批16条父迹新增采样为0。实际新增6655请求、30882294输入token、140712输出token，共享初批局部分支只计一次，评估兼下批训练不重复计成本。模型阶段累计约123.88分钟，未追加训练或扫α。plain最终评估成功后，仅退役本新run约29.97GiB Adam动量，保留FP32主权重/BF16 actor/原manifest及退役记录，不能完整恢复原Adam；exact完整16次FP32/Adam状态和actor仍保留到项目磁盘。

结论限于这批四窗单seed：exact的合法性改善明确，净收益优于plain但未超过共同SFT或基础RLOO，尚未证明额外采样值得。用户已授权完成后自主决定并继续，因此紧接着启动四个现有模型的namespace4新采样复评，共64条父迹、无训练/EST，用新随机数核对方法排序；仍使用原Train窗，不称日期泛化。两个阶段均由systemd后台运行，VPN断线不结束GPU任务。

证据：[三臂汇总](../../runs/m2-rloo-local-credit-demo-20261025-v1/summary/arm_summary.csv)、[完整读数](../../runs/m2-rloo-local-credit-demo-20261025-v1/summary/comparison_readout.json)、[曲线图](../../runs/m2-rloo-local-credit-demo-20261025-v1/summary/learning_curves.png)、[逐窗曲线](../../runs/m2-rloo-local-credit-demo-20261025-v1/summary/window_results.csv)、[最终配对](../../runs/m2-rloo-local-credit-demo-20261025-v1/summary/paired_results.csv)、[完成摘要](../../runs/m2-rloo-local-credit-demo-20261025-v1/suite_completion.json)。新collector/trainer/controller/reader通过Ruff和必要的公式、NA、身份及原模式保持快检，真实GPU完整流程通过；CPU准备与GPU并行，未加长审计。


## 2026-10-26：新采样复评保持局部两臂低于RLOO，启动一次exact辅助项强度试验

用户授权完成后自主决定并继续。原三臂完成后，后台自动接续四个现存模型的namespace4复评：原四Train窗、seed42、每窗4条父迹，64条全部完整，无训练或EST，6组16配对的初始prompt/首请求种子一致。新种子未参加训练，但窗口仍已接触，正式Valid/OOS未使用。

| 模型 | 新namespace4净收益 | 相对新SFT | 拒单率 | 费用均值 |
|---|---:|---:|---:|---:|
| SFT | 3.80587% | — | 7.25% | 4737.76 |
| RLOO | 4.17843% | +0.37256pp | 9.86% | 5282.85 |
| plain | 3.28826% | −0.51761pp | 10.61% | 4557.50 |
| exact | 3.19992% | −0.60595pp | 4.48% | 5517.69 |

plain/exact均在四窗低于RLOO，差分别−0.89018/−0.97852pp。exact−plain从原namespace0的+0.37920pp翻成−0.08834pp，合法性改善保持，收益优势未保持。三种训练策略相对SFT均仅在0073改善；去0073后的RLOO/plain/exact差分别−0.35569/−0.88259/−1.03389pp，四窗平均正向不能称全面改善。新增1393请求、6431133输入token、29336输出token，模型阶段487.69秒约8.13分钟，无新checkpoint。GPU holder随后恢复76GiB、claim=None。

随后的CPU量级诊断显示，raw局部修正/RLOO优势RMS仅0.49%/0.43%，非零请求1.2%/1.4%；混合白化std与纯RLOO几乎相同，不能归因为归一化bug或断言实际梯度比。各窗口的replica独立shuffle，锚点可分散到不同optimizer minibatch，诊断只按replica记录的统计不等于实际minibatch。EST2粗SE也提示放大可能同时放大续写噪声。

因此停止扩大原mixed配置，另以一次固定16步exact_aux试验检验作用强度：完整RLOO单独白化、原基础PPO权重1/(4×20)保持；局部exact_S除同一RLOO标准差，独立PPO裁剪后按ηα/8=0.015625计权，8是整个collection预定anchor数，不按实际支持数或当前minibatch数量除。局部每请求标量约为旧配置10倍，不称实际梯度10倍。α0.5/η0.25、reward、LR、完整父请求和分支预算均固定，不扫参数；16次后同时报告namespace0和4。01:55（北京时间）已由systemd启动，初批16父迹与原局部分支直接复用、新增初采为0。

为新状态腾空间，在旧训练及fresh64条均成功后，退役旧plain FP32主权重16.43GiB和旧exact Adam动量29.97GiB；最终BF16模型、旧exact FP32主权重、原manifest/小日志/指标/轨迹保留。旧plain/exact均不能再完整恢复原Adam，退役记录明确；新aux按完整状态逐批保存。

证据：[fresh完整读数](../../runs/m2-rloo-local-credit-fresh-eval-20261026-v1/fresh_readout.json)、[四模型汇总](../../runs/m2-rloo-local-credit-fresh-eval-20261026-v1/summary.csv)、[原/新采样差值图](../../runs/m2-rloo-local-credit-fresh-eval-20261026-v1/paired_delta_comparison.png)、[量级诊断](../../runs/m2-rloo-local-credit-demo-20261025-v1/summary/credit_scale_diagnostic.json)、[状态退役](../../runs/m2-rloo-local-credit-demo-20261025-v1/aux_pilot_storage_retirement.json)、[新aux状态](../../runs/m2-rloo-local-aux-demo-20261026-v1/exact_aux/rloo_state.json)。新入口Ruff与必要的公式/零局部退化/clip/路径/元数据及真实初批检查通过，没有改变旧各臂默认行为。


## 2026-10-26：exact独立辅助损失16次完成，强度调整未救回，当前配置停止扩训

上述单臂试验已完整完成16次有效更新、namespace0和4各16条终点评估，按预定预算收尾，无中途调参或最好轮次选择。共同SFT、初批16条父迹及已有局部分支直接复用；基础RLOO单独白化，局部PPO同一次forward/ratio计算、合并loss一次backward，固定全批8锚点分母，α0.5/η0.25不变。完整FP32/Adam真实跨批4→8→12→16接续、最终16次状态与1195个组件齐全、BF16 actor导出完整后进行评估。

| exact_aux相对参考的净收益差 | 原namespace0 | 新namespace4 |
|---|---:|---:|
| 对SFT | −0.39068pp | −0.75450pp |
| 对RLOO | −0.91152pp | −1.12707pp |
| 对plain | +0.19689pp | −0.23689pp |
| 对原exact | −0.18231pp | −0.14855pp |

aux两组净收益3.09080%/3.05137%，金融Y2.99973/2.90535，拒单33/353（9.35%）、34/354（9.60%），费用4695.83/4780.79。两组都在四窗低于RLOO；原采样对SFT2升2降，新采样四窗全降。去0073后对SFT仍为−0.58215/−0.91955pp。8组对照各16配对均完整，无终局指标NA，不把复用的配对行当独立新样本。所有最终初始prompt/首请求种子一致，未改reward、成本或恢复上限，正式Valid/OOS仍未使用。

完整0/4/8/12/16曲线净收益3.48148%/3.02162%/2.93954%/2.76932%/3.09080%，随后新种子3.05137%；金融Y3.40152/2.92798/2.80010/2.55866/2.99973，随后2.90535。局部支持26/32（7/5/8/6），非零13/1403训练请求（6/3/3/1），仍很稀疏。辅助loss在8/16次实际更新中非零，各事件绝对loss和约为base的0.457%；有符号loss存在抵消，这些数值不等于梯度范数或梯度贡献比例。调整使局部每行标量系数约10倍，并未获得预期收益，不能把当前问题简单解释成白化抹去信号或权重太小。

新增16次actor、0次critic更新，1403个父请求含123次拒单参与训练；新增80条父迹与176条实际EST均完整，原始初批及历史参考不重复计生成成本。五批新采样共3488请求（1765父请求、1723局部分支请求）、16163180输入token、74063输出token，模型阶段约60.05分钟。整个三臂、fresh四模型和aux后续累计新增48次更新、11536请求；旧基础RLOO16次更新复用，不计新增。没有无限加轮、参数搜索或选择最好checkpoint。

结论与出口：当前原生同前缀候选下，稀疏局部credit未带来基础RLOO之外的净收益，单次独立辅助项强度调整也未救回。保留已能学习的RLOO基线作为demo，停止扩跑当前M2及此强度假设。下一次方法设计应先检验可获得的真实动作差异、续写估计质量和任务适用性，不能据本批否定全部credit、推断金融项目上限或声称已获得日期泛化。

全部结果与最终BF16 actor保存，正向RLOO基线完整续训状态保留。aux两组终评及正式读数成功后，退役约29.97GiB Adam动量，保留其FP32主权重/BF16模型、metadata/原manifest及指标轨迹；因此aux当前不能无损续接原Adam，不冒充完整可续训状态。旧plain/exact此前退役组件记录也保留。磁盘可用恢复约50GiB，GPU已归还holder76GiB、paused=False、claim=None，无后续GPU任务。

证据：[aux完整读数](../../runs/m2-rloo-local-aux-demo-20261026-v1/summary/aux_readout.json)、[两组终点对照](../../runs/m2-rloo-local-aux-demo-20261026-v1/summary/comparison_summary.csv)、[完整曲线图](../../runs/m2-rloo-local-aux-demo-20261026-v1/summary/learning_curve.png)、[逐窗曲线](../../runs/m2-rloo-local-aux-demo-20261026-v1/summary/window_results.csv)、[128条计划配对](../../runs/m2-rloo-local-aux-demo-20261026-v1/summary/paired_results.csv)、[完成摘要](../../runs/m2-rloo-local-aux-demo-20261026-v1/completion.json)、[终态动量退役](../../runs/m2-rloo-local-aux-demo-20261026-v1/terminal_optimizer_retirement.json)。训练/控制/读取入口Ruff与必要的公式、零局部退化、裁剪、固定分母NA、身份及旧metadata兼容快检通过，真实GPU流程通过；没有扩大重复审计。


## 2026-10-27 主动订单监督三步：能拟合标签，整段稳定性退化

在既有八个Train窗及冻结八窗RLOO seed43模型上，复用父迹的j3/j7/j11/j15共32个状态，按可见状态菜单构造128个不同合法订单，每单4次随机续写。512/512分支全部到j20终点，5879次真实后续请求；前2次选优、后2次要求相对原订单同向改善，共11条标签，9处前半无改善、10处后半未保持、2处原动作非法不支持。11条标签覆盖6窗，buy8/hold2/sell1；买入均为AAPL，与确定性菜单偏向有关，不是普适择股证据。

从同一冻结RLOO复制候选actor，以独立Adam和持久GPU FP32主权重完成16次监督更新（32条winner使用＋32条正常回放），不增加critic或修改reward。标签训练NLL1.885→0.092、正常回放NLL0.027→0.136；只保存最终BF16 actor，没有新完整FP32/Adam磁盘副本。初次字段合并错误发生在0次更新前，修复后复用原采样继续；旧失败日志保留，不额外采样或重放已有更新。

固定八窗、seed43、namespace5/6、每窗4父迹，共六组192个原定槽均已运行：SFT与冻结RLOO各64/64到终点；新模型两组各29/32，5条生成到4096token上限、1条到每session四请求上限。六条失败保持NA，不补抽、不自动hold；新模型整段收益、Y及费用的固定分母均值保留NA，不能说其整段收益一定更低。

SFT两组净收益2.27805%/2.24413%，RLOO为3.08679%/3.10666%，差+0.80875/+0.86252pp；逐窗8升0降/6升2降、去0073后仍+0.78037/+0.75702pp。RLOO拒单9.60%/11.36%，SFT10.24%/9.09%，行为维度未同步稳定改善。新模型AAPL买单占47.29%/43.49%、target0.1占31.52%/29.98%，两项RLOO均0，说明标签偏好被学入；同时环境错误165/772、173/802（21.37%/21.57%），包括工具格式、参数、买卖方向等退化。生成不完整另外统计，不与环境拒单混淆。

**出口：STOP_MODEL_ROLLOUT_FAILURES。** 这一次主动局部监督没有转成稳定整段策略，按预定出口结束本配置，保留RLOO为demo基线。不能由小样本、固定候选菜单的负结果推断全部反事实监督无效或唯一根因；新namespace仍是既有Train窗口，不是日期泛化，正式Valid/Final OOS未使用。

总新增生成10305请求、48189742输入token、249851输出token，16次监督另计298018输入token/1304标签token；模型阶段约36.31分钟。最终模型、全部失败与原始轨迹保留，GPU已归还holder76GiB，磁盘约239GiB可用，没有后续GPU任务。

证据：[demo入口](../../runs/m2-counterfactual-order-demo-20261027-v1/README.md)、[最终读数](../../runs/m2-counterfactual-order-demo-20261027-v1/summary/readout.json)、[六组终点表](../../runs/m2-counterfactual-order-demo-20261027-v1/summary/summary.csv)、[行为分析](../../runs/m2-counterfactual-order-demo-20261027-v1/summary/behavior_readout.json)、[比较图](../../runs/m2-counterfactual-order-demo-20261027-v1/summary/comparison.png)。


## 2026-10-28 扩大版RLOO与订单评分器：完整Valid主实验结束

**2026-10-28：扩大版三步已执行结束，状态`COMPLETE`。** 新增RLOO更新64次、累计80次；Valid预定600槽，记录600槽、600条到原终点。**未建立完整学习方案对简单基准的优势，下一阶段应转向任务或策略设计。**

| 方案 | Valid平均净收益 | 金融Y | 完整轨迹 |
|---|---:|---:|---:|
| 共同SFT | -0.08305% | -1.43893 | 120/120 |
| 原16步RLOO | 0.15323% | -1.09566 | 120/120 |
| 32窗RLOO终点 | -0.04260% | -1.43391 | 120/120 |
| 共同菜单控制 | 0.01370% | -1.33925 | 120/120 |
| 学习订单评分器 | -0.31100% | -1.77124 | 120/120 |
| 全现金（CPU） | 0.00000% | -1.30078 | 60/60 |
| 等权持有（CPU） | 0.77548% | -0.05280 | 60/60 |

固定分母不完整的整组收益保持NA，不以成功子集替代。Valid的60个窗口重叠，这是一轮开发比较，不是60个独立市场样本，也不是Final OOS。最终读数出口`NO_COMPLETE_LEARNING_ADVANTAGE_OVER_SIMPLE_BASELINES`，看到Valid后没有追加训练、换窗口或调reward。

实际新增生成49827请求、236113337输入token、1050340输出token；读取原采样与重复训练使用另列。最终模型、失败、轨迹和完整读数保留；完成时磁盘可用约183.5GiB。后台GPU阶段已结束，资源归还记录见state.json。

证据：[完整读数](../../runs/m2-broad-rloo-order-main-20261028-v1/summary/readout.json)、[最终报告](../../runs/m2-broad-rloo-order-main-20261028-v1/FINAL_REPORT.md)。


### 扩大版主实验的逐窗补充

原RLOO−SFT为+0.23628pp，39胜21负，去最大正贡献窗后+0.20064pp；扩大RLOO−原RLOO为−0.19583pp，26胜34负；评分器−共同菜单为−0.32470pp，24胜36负。原RLOO比等权持有低0.62225pp。原生SFT/原RLOO/扩大RLOO拒单率分别14.38%/16.98%/19.11%；菜单和评分器的零拒单来自外部约束。结束扩训和本评分器变体，下一阶段转向任务/策略设计。GPU已恢复自有holder76GiB、claim=None。


## 2026-10-28 组合管理第一步：共同底仓与组合动作已接通

仅执行新三步的第一步。共同扣费等权建仓、每5日四动作、日度账务及原自然终点已接通；动量/反转向满仓3%/5%/7%模板作L1≤0.2的小步移动，可以在防守后恢复股票敞口。32个Train窗×5固定规则160条及32条共同SFT零样本轨迹全部完整，128次动作选择、0新训练更新；hold/cash共64项独立共享账本参考校验通过。

被动等权/动量/反转/防守/零样本平均净收益分别1.44707%/1.34967%/1.36345%/0.72362%/1.28109%，零样本主动超额−0.16598pp。合法性来自4动作ID约束，不能当作LLM恢复学习；没有训练优势或日期泛化结论。第二/三步未启动，本步未读取Valid或Final OOS。

首次LLM启动的vLLM约束输出API差异在0生成前修复，160条规则结果直接复用，原失败保留。证据：[第一步报告](../../runs/m2-portfolio-management-step1-20261028-v1/REPORT.md)、[实际轨迹](../../runs/m2-portfolio-management-step1-20261028-v1/summary/example_trajectory.json)、[完整读数](../../runs/m2-portfolio-management-step1-20261028-v1/summary/readout.json)。


## 2026-10-28 组合管理第二步：确定性教师与共同SFT

354个学习窗的3540状态、14160条候选执行后被动续持至原终点的确定性分支，用于一轮LoRA动作ID监督，共222次更新。45个开发窗与学习终点隔22个交易日，42个交界窗口排除；两区各32预定评估窗口独立报告，失败保持NA。学习区净收益1.49954%→1.61741%（0.11788pp，32/32完整）；开发区净收益2.46718%→2.50110%（0.03392pp，32/32完整）。教师SFT在本轮开发区高于原共同SFT，但没有超过被动等权持有。教师值不是当前策略Q，未来结果只作标签；基础模型有旧Train暴露，开发隔离仅针对新增监督。第三步RL未启动，本轮未访问Valid/Final OOS。[完整报告](../../runs/m2-portfolio-teacher-sft-20261028-v1/REPORT.md)、[配对与贡献读数](../../runs/m2-portfolio-teacher-sft-20261028-v1/summary/readout.json)。


第二步行为补充：开发区新模型117/128决策为反转（91.41%）、没有hold，费用2044.78→2593.92；净收益2.50110%低于固定反转2.53715%与被动2.59195%。相对旧模型+0.03392pp在去最大正贡献窗后转为−0.00762pp，不建立稳固开发增量。仅报告本轮固定终点，不据此追加轮数或改变模型选择。
