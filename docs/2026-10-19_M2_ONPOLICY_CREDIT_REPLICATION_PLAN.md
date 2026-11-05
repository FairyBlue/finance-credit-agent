# 当前π状态与固定收缩：独立种子复验

2026-10-19，SGT（UTC+8）。用户授权继续下一步，保持GPU优先、CPU快准备与加载并行。本轮是原窗口上的独立随机状态/Monte Carlo复验，不是新日期泛化；不训练，不读取正式Valid/Final OOS。

## 日期选择与范围

先仅核Train窗口元数据：461个交易session、441个21-session滑窗。要求与原8窗及既有12个Train开发评估窗口的**整个窗口之间保留至少20个空交易session**，满足条件的新窗口为0。精确收据为`runs/m2-onpolicy-credit-replication-selection-20261019-v1.json`。未读取新Y、Valid/OOS或完整行情库存；原日期也曾用于奖励标定，不能称未接触。

因此使用原8窗：formal-train-0001/0009/0017/0025/0065/0073/0149/0157，每窗1条全新共同恢复SFT π父轨迹，从reset至原21-session终点。固定j5/15首次进入、该session任何请求之前的实际完整观察，16个planned anchor。沿用[首批冻结方案](2026-10-18_M2_ONPOLICY_CREDIT_PLAN.md)的完整prefix/账本/工具记忆和NA规则；未到达不补父轨迹、不换窗口/seed，已达到j5不因父后来失败删除。

## 独立性与固定预算

replication_id与种子首namespace均为`m2-onpolicy-credit-replication-20261019-v1`。PARENT/HEAD/EST/REF四role独立，预列全部23744潜在请求key/seed；对旧pilot、旧状态EST/REF和第一批当前π状态的**完整潜在种子表**核无碰撞，旧未用种子也不回收。只复用冻结driver/runtime/reader公式，不复制首批随机轨迹或标签。

每状态4个原生独立head，每slot仅1次请求。非法、信息工具、截断或未推进的head保NA，不repair、不代选hold。每合法slot2条独立EST；每同anchor具体订单在首步后金融状态/完整消息核同后共享8条独立REF，无额外hold条件。条件不一致整anchor NA，不盲合或重抽。最多128 EST+512 REF分支，实际条数仅由合法head/同条件去重规则确定，不依Y。

共同π、T1/p1/k−1、原自由工具接口、8192输入/4096输出、最多32 active、原终点不变；父及续轨每session最多4请求，拒单/信息工具/恢复均计入。硬预算为父640+head64+EST4608+REF18432=23744请求，不能把未花预算用于追加父轨迹或挑状态。

## 比较与预设判读

沿用九方法和全部固定α=.25/.5/.75，保留zero、ungated plain、两种gate plain、family pool和未缩exact，不选最佳系数、不按family事后选α/gate。α乘raw exact residual，η=.25只作两侧共同尺度派生，不调η、不加入GAE/whitening。REF每cell8条完整均值按原四slot multiset重复权重中心化τ；固定前4/后4分半。

主分母16 anchor×2EST=32组rep/128槽rep。四head合法、该rep四EST完整且各所需REF八条完整才有共同支持；任一缺失整组九方法含zero全NA。支持由本批自己确定，不与首批筛共同状态或只保动作分歧组。完整报告parent/head/branch失败和原分母。

预先关注α=.25和.5各自相对zero的成对loss差（候选减参照）。**两者在full8、REF前4、REF后4六个差均小于−1e−12**，才记为本批描述性方向共同复现；这仅是符号/浮点容差规则，不是显著性门槛。任一未满足，报告未共同复现/参考敏感，不改挑另一α为胜者。若无支持或没有动作分歧，报告不可判，不为凑复现补跑。所有九方法/三α、.5−.25及同订单/同族异订单/跨族分层仍并列报告。

逐window/anchor及整anchor删除仅作固定次要敏感度，不变成事后完成门槛；不计算正式p值或将两个EST rep当独立市场样本。不合并首批与复验挑有利均值，不把有限Q代理称A^π/V^π或收益。

## 执行与完成

复用已成功且SHA固定的原1000行driver、GPU worker和九方法reader；新增薄入口只固定新namespace、来源身份及复验摘要。启动前保必要模型stat、小merge收据、种子预算、源码/输入和租约核对，不重复8GB模型或行情全量哈希。CPUdriver已可直接随GPU加载启动，读数和小表独立复核在GPU运行时准备。

完成需所有planned阶段及固定规则派生的分支被分类，CPU/GPU/service/readout真实退出，完整NA分母、种子/预算及原holder归还收据齐备。未复现也照样保结果，不回收闲置预算补采；后续学习或新设计另定，本轮不自动训练。
