# M2 多状态具体订单功劳验证方案

2026-10-17。用户已授权继续CPU机制验证。本方案在真实回放前保存并绑定SHA，检验具体订单分组、分歧gate和状态基线各自的作用。主目标为原窗口终局金融Y在同状态动作干预下的变化，后续规则固定为hold至原窗口末日；它不是原模型继续行动时的策略条件优势。

## 固定样本和执行范围

仅使用[已完成对照批次](2026-10-12_M2_ACTION_POOLING_COMPARISON_REPORT.md)中M2、seed42/43的原始训练轨迹。窗口与轮次固定如下，不按收益、动作效果或回放成败换样本。

| 原训练轮次 | 窗口ID后四位 |
|---|---|
| 1 | 0001、0065 |
| 2 | 0009、0073 |
| 3 | 0017、0149 |
| 4 | 0025、0157 |

每窗固定零起算决策session `j=0,5,10,15`。共 `2 seeds × 8 windows × 4 dates = 64` 个source group，每组保留原四兄弟槽0–3，共256个anchor实例。j=0单列为共同初态控制，经济风险状态SHA实际核验相同后解释；不与后期状态异质性混为一个结论。

先完整重放64条原历史轨迹，核对全部请求、推进顺序、完成状态和终局指标。再在各指定成功推进请求前保存独立状态。Prefix包含该请求之前的所有原请求，也包含同一session先前的invalid、retrieve和compute，不能仅筛选更早交易日。

每个anchor尝试本source group原四个成功推进订单的multiset，保持槽位及重复次数；另设hold参考分支。未去重上限为1024个移植cell和256个hold参考分支。这是候选分支数，后续逐日hold的环境调用另计；只按相同状态、订单及合同身份安全去重或缓存，保留所有逻辑cell映射与实际资源计数。

使用原21个冻结session、原初始NAV、完整prefix权益与风险历史、原scorer/calibration和费用口径。分支执行一次指定动作后全部hold，直到原窗口终点，不重置回撤、换手或最大持仓记录，不截短期限。主Y来自原金融terminal score，不使用含无效请求罚分的训练reward sum，也不以一日净收益或NAV百分点替代。全程不请求模型、不使用GPU、不更新训练参数，不读取正式Valid或Final OOS。

## Oracle和状态身份

在每个source group中，i表示anchor所属原兄弟，j表示原订单槽。定义：

- `Q_ij`：从anchor i执行订单j，然后hold到原终点的金融Y。
- `H_i`：从同anchor直接hold到原终点的金融Y。
- `q_ij = Q_ij − H_i`：相对同状态hold的动作贡献。
- `Y_i`：原历史轨迹实际终局Y；`F_i = Y_i − Q_ii`。

附加hold只作为H参考，不成为第五个订单槽。每条原对角动作必须可复现并完整到达终点，满足 `Y_i = H_i + q_ii + F_i`。F包含原策略后续行为与固定hold后续的差异，并非独立随机噪声。

状态匹配使用实际经济风险状态的canonical SHA，绑定session/date、现金、股份、原初始NAV、权益/收益与换手历史、最大仓位和高水位等reward相关状态。不得用兄弟编号充当state，也不看oracle调整近似分桶。对角actual financial-risk SHA加完整订单键仅在原source group边界内诊断，仍减原全组均值，沿用具体订单gate；不额外实现会改变目标的state内中心化，不能把更细分组解释为已经消除了状态基线。完整订单键为工具名和规范化的全部订单参数。

移植订单以实际env.step原样尝试判断合法性，不用近似方向阈值代判。请求合法性、是否推进、next-open skipped order和zero-fill分别记录。请求被接受且合法推进的skip/zero-fill保留，不因未成交、收益为负或oracle较小而过滤。非法cell保存info和session未推进的证据，直接结束该分支，Y/q记为NA；非法或未完整结束的cell保留总分母，不改订单、不补零、不补采，不静默删除anchor。

缓存键保守绑定financial SHA、visible SHA、工具及无效请求计数、env kwargs SHA和具体action键；peer统计只按financial SHA及指定订单键计算，不把缓存命中当成独立样本。

## 三个分析面板

### A 同状态四订单的结构检查

仅当某anchor的四个订单均合法且各分支完整时，以该行四槽的 `q_ij − mean_j(q_ij)` 为目标，比较plain、family pool、具体订单pool保留原family gate、具体订单pool加具体订单gate。分组和gate分别变化，hold参考不进入四槽均值。

报告全部256行中的完整行、非法行、skip/zero-fill及peer覆盖；不完整行不能改用较少订单重心化后加入主MSE。MSE保留全部四槽，排序只在同anchor比较真实效果不同的合法订单，预测tie计0.5；重复槽和去重订单的分母分别记录。

该面板采用确定性fixed-hold outcomes：ungated plain应精确恢复中心化q，具体订单重复也应有相同q。它检查粗分组与gate的表示损失，不检验独立扰动平均或RL收益。

### B 原四兄弟终分与局部贡献的组件一致性

全部对角贡献 `q_ii = Q_ii − H_i` 与F独立落盘。原四兄弟各自在自己的状态，其 `center(q_diagonal)` 不是同状态动作优势。原Y或fixed-hold对角Q产生的残差与该向量比较，只报告组件一致性。

分别呈现H、q、F尺度、交叉协方差和精确重构；保留全零预测器，不将各分量平方能量直接相加称为误差归因比例。减去H的额外参照使用未来hold oracle，必须单列为特权baseline；原Y减H仍保留F，不保证误差改善，不作为可部署候选。

### C 同状态订单菜单下的对角贡献

仅对完整四订单行定义 `tau_i = Q_ii − mean_j(Q_ij)`。Partial行不改变菜单、不重心化。Tau相对的是各自同状态四槽菜单，区别于相对hold的q；因此 `center(q_diagonal)` 不能替代tau。

只在16个移植cell全部合法完整的source group上，比较原历史proxy与四维tau的一致性。原中心化残差必须零和，但不同状态的tau未必零和：记录不可避免的表示误差下界 `(mean_i(tau_i))²`。另对tau直接施加同一中心化、pool和gate算子得到 `H(tau)`，作为表示受限的oracle参照，不称为可部署baseline。Partial group及缺失cell继续计入总覆盖分母，不将其余成员拼成新组。

这里的比较属于固定hold continuation目标下的一致性，不能据此证明原策略梯度存在偏差。跨不同状态的向量序一致性也不称为同一个决策问题中的动作排序。

附加支持行描述仅在该行四订单均合法时保存原始tau及proxy误差，不对partial group重心化、不计算跨状态动作排序，也不与full16主组分解混合。Panel B另保留`原Y−H`这一特权输入，隔离减baseline而不移除F的变化；`固定hold对角Q−H=q_diagonal`也单列，二者均不称为可部署baseline。

## 判读和完整性

所有统计均为描述性，按窗口、seed、j及剩余期限保留明细和分层汇总。共享轨迹、多日anchor和重叠行情不是独立样本，不作iid p值或金融收益置信区间。所有方法共享同一逻辑分母；MSE、真实非零效果覆盖、排序tie、gate关闭和singleton分别报告，避免全零收缩掩盖动作区分损失。

真实回放前封存本方案、采集器、读数脚本和实际输入合同。必要核验为：64条原历史完整重现，干预前状态与风险历史相等，hold相对自身q为0，同状态同订单结果相同，原对角动作合法完整，H/q/F重构成立。意外回放或账务不一致时保留失败并停止相应科学解释；移植订单非法是预设覆盖结果，不自动等同程序失败。

如果细分订单后仍有系统误差、精确state匹配大量退化为singleton，或原proxy与tau不一致，应分别报告目标、表示与覆盖边界，不通过换窗口、改期限或放宽state键修饰结果。候选通过本轮最多支持进一步Train内开发确认，不自动授权扩大RL。

本阶段不学习baseline。Training-only crossfit需另行固定可见输入、episode及重叠horizon分折、时间purge和留出评估，不能按step或兄弟随机拆分。本轮未来hold特权参照不替代这些要求，也不改变[评估协议](../EVAL_PROTOCOL.md)的正式训练与OOS门禁。
