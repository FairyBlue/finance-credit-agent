# 当前策略访问状态的credit与固定收缩验证

2026-10-18，SGT（UTC+8）。上一轮旧M2状态的fresh EST/REF对照已完成，zero附加credit的有限参考MSE最低。用户授权继续在共同π自身状态验证，并明确GPU优先、CPU准备/读取并行。本轮使用共同恢复SFT π，不训练，不读取正式Valid/Final OOS，不修改旧运行和生产代码。

## 状态、首动作与预算

沿用前轮全部8个已接触Train窗口，每窗口一个全新π父rollout，从原episode初始状态运行至原21-session终点。固定取j=5、15的首次进入观察，即该session任何请求之前，保留完整原生观察、工具记忆、所有prefix请求/拒单、账本与prompt token。共16个planned anchor；未到达保NA，不补父轨迹、不换seed/窗口，不按父终局收益筛选。已到达j5保留，即使父后来未到达j15或失败。

每anchor原输入下4次独立head请求，原自由工具输出；每slot只采一次。拒单、信息工具、截断或未推进都保该head NA，不repair/retry，不替换hold。合法head各2次独立EST续写；每anchor每个实际同条件的合法具体订单各8次独立REF续写，不额外加入hold条件。EST保原slot原文独立采样，不复制随机后续；REF共享前需证明首步后的完整金融状态和真实观察一致。

若同anchor/economic-order的首步后账本或完整消息不同，保留所有原head并显式记录CONDITIONAL_STATE_MISMATCH，整anchor主指标NA；不盲合REF、不换状态或重抽。未执行的分支说明结构性跳过原因，不算模型完成路径，潜在种子不回收给其他分支。

父轨迹与continuation使用相同π、T1/p1/k−1、输入8192/输出4096、原21-session终点，每session最多4请求，拒单/信息工具/恢复都计入。父轨迹、head、EST、REF四类固定namespace，全部潜在请求key/seed预列且无碰撞，不复用旧pilot/EST/REF种子。无补抽、自动hold或加步数。

| 类别 | 最大请求数 |
|---|---:|
| 8条父rollout，20个推进session，各最多4请求 | 640 |
| 16 anchor×4个单次head | 64 |
| 每合法slot×2EST，j5/15首步后分别剩14/4session | 4608 |
| 每anchor最多4个具体订单×8REF，同上剩余session | 18432 |
| 总硬上限 | 23744 |

最多128条EST、512条REF后续路径；实际合法head和unique条件决定较少的路径数，是预设结构规则，不依赖Y。缺失父anchor或非法head仍保全16anchor/64head槽与32组rep/128槽rep主分母。最大32条active请求；父rollout起初仅8条并行，后续分支合批至32，整个过程共用一个GPU worker，减少重复加载。

## 固定算子与有限参考

REF按每具体cell八条完整Y求均值，再按原四slot multiset重复权重中心化为τhat。主组需四个合法head、该rep四条EST完整、各所需cell八REF完整；任何缺失则整组四槽全方法NA，zero也不补零。REF固定前4/后4另作敏感度，不把8次均值当精确Q。

九方法并列：zero、ungated plain、family gate plain、order gate plain、family pool/order gate、exact order pool/order gate，以及α=.25/.5/.75乘raw exact residual。α=1和0分别对应已有exact及zero。原六公式不变；不在本批选最佳α、不按状态调α，不调整训练η。主loss比较αS与raw τ；统一η=.25仅作两侧同缩放派生，loss为raw的1/16，不加入GAE或whitening。

报告同支持的原金融Y平方MSE、成对candidate−comparator loss差、固定REF4+4方向和同订单/同族异订单/跨族分层。α相对zero是参考依赖的预设对照；不能只拿exact相对plain的投影必降充作方法证据。两半翻号则不可判，同向也不表示参考精确、显著或金融收益改善。不计算动作排序或正式p值；8父轨迹、重叠Train日期、两个EST rep不作iid市场样本。

## 执行顺序与完成

先封存方案、上限与必要模型stat/已验证收据，立即启动GPU；CPUdriver在GPU加载期间完成最小IPC、首请求前单独seal；读数器在GPU运行时完成并于真实Y读取前seal。不重复8GB权重或全部行情checksum扫描；保必要真实输入/账本/请求对应、原终点和GPU租约。独立新controller/worker/driver/reader，不修改旧封存源码。

完成需所有planned父轨迹、anchor、head及由固定结构规则确定的分支被分类并保留失败，CPU/GPU/service/readout实际退出，预算与种子核验，原holder归还。只有本批证据支持后才另定学习或扩样；本轮没有自动训练或加预算。
