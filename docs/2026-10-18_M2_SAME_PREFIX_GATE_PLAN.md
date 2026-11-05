# 同输入具体订单gate隔离验证

2026-10-18，SGT（UTC+8）。用户批准继续[原生首请求诊断](history/CREDIT_ORACLE_DIAGNOSTICS.md)后的gate单因素检查。本方案在新诊断执行前固定；设计已经知道上一轮29个完整组、7个gate遗漏的结果，因此是实现和覆盖验证，不是独立确认或新的方法收益试验。

## 固定范围与目标

只读`runs/m2-same-prefix-sampling-20261018-v1`的全部128槽、32个anchor及其封存来源；不新增模型、环境或GPU调用，不训练，不读取正式Valid/Final OOS。旧目录、源码与完成收据不改写。新增隔离模块和CPU诊断，不接生产训练；旧`compute_sibling_credit`仍要求完整四兄弟20决策轨迹，不把首请求伪装成该接口的完整轨迹。

目标仍是同真实输入、同金融状态、同模型快照下，动作后固定hold至原21-session终点的局部相对贡献。记原终局分为Y，同状态hold为H，q=Y−H，组内相对目标τ=q−mean(q)。四槽共用H，故τ=Y−mean(Y)；主实现直接输入原Y，保持旧plain中心化公式，q和H仅验证平移恒等，不接入候选算子。它不是原策略继续交互条件下的Q或advantage。

## 四个算子：只改变gate

每个完整组先计算r_i=Y_i−fsum(Y)/4，固定四个输出：

| 算子 | raw S |
|---|---|
| zero | 0 |
| ungated_plain | r_i |
| family_gate_plain | D_family × r_i |
| order_gate_plain | D_order × r_i |

D_family表示四槽有至少两种动作族；D_order表示至少两种具体经济订单。经济订单使用合法原调用：buy/sell按工具名、symbol.strip().upper()和有限数值target，0/0.0/−0.0同值；hold和abstain保持不同，abstain reason不划分新订单。数值字符串、bool、非有限target及不合法key拒绝进入算子，不修订单。固定η=0.25仅另列ηS，不重调尺度、不做pooling、baseline学习、GAE、whitening或actor更新。

## 完整性、分母与输出

128槽全部进入输入快照。仅四槽均生成完整、合法推进且固定hold标签完整的组进入主算子；验证anchor、完整prompt哈希、金融状态、模型快照和slot0…3唯一一致。任何partial组的四槽、四个算子全部NA，zero参照也NA；不能只给4个非法槽NA而对其余合法子集重心化。保留原env状态与非法原因，已合法skip仍合法。

执行前封存新源码、测试和方案，以及旧合同、输入、原始返回、标签、读数、独立审计和模型身份收据的哈希。核对继承合同的全部封存源码/权重；执行后再核对新旧身份。所有输出在新目录排他创建，失败保留，不覆盖旧run。

报告32组/128槽总分母、full/partial支持、每算子有信号的组和槽、family→order改变/恢复/丢失的组和槽；给出逐槽原Y、q、raw S及固定ηS。核对四槽zero-sum、输入排列等变、q/Y平移一致、原family gate打开时两gate输出完全相同，以及同订单同标签时order gate与ungated关系。浮点恒等采用绝对和相对容差1e−12，是否有非零分歧按固定绝对阈值1e−12单列，不以结果调整阈值。

不计算新的pooling MSE、动作排序、金融收益或显著性；本轮signal变化由既有确定性标签与gate公式决定，不是降噪证据。全同订单组即使重复率高也没有动作对比；不能只报告11个分歧组。夹具、真实读数和独立实际审计分别保存，工程完成须有真实CPU退出码0。

## 后续边界

本轮完成后再固定pool键的单因素候选；若目标是训练信号降噪，需另行冻结后续策略π、截止日及Q^π的随机continuation/独立参考估计预算，并采集该共同策略自身访问的状态。不能把确定性hold标签当作随机continuation，也不能把未来H输入训练baseline。该后续GPU采样、pooling或RL不在本轮执行范围内。
