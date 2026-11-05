# 随机后续下的gate/pool估计对照

2026-10-18，SGT（UTC+8）。154路径可行性pilot已完整完成；按用户继续设计并检验pooling、优先GPU有效采样的要求，进入fresh估计/独立参考对照。不把pilot两rep重划成两侧，不训练、不读取正式Valid/Final OOS。GPU模型加载与CPUdriver/读数准备并行，沿用已验证模型stat与快行情loader，重复大哈希后补。

保持共同恢复SFT为π、T1/p1、原自由工具接口、8192输入/4096输出上限和原21-session终点。来源仍为全部32旧Train前缀；124个合法原slot各2条独立估计，共248条；77个合法订单/hold条件各8条独立参考，共616条，总864条路径。原4个非法首动作保留NA。每原slot/每rep估计都独立续写，不dedup复制后续；参考使用独立REF seed namespace，估计使用EST namespace，与pilot也分离。参考同条件跨原slot共享的前提是首步后实际观察/账本完全一致，已做小文件核对。

当前session最多4个模型请求，信息工具/拒单/恢复都计入，未推进保NA；截断、输入超长保NA，不补抽、不自动hold。估计最大11408请求、参考27616，总硬上限39024，预先列全部seed槽；不根据结果追加预算。所有864路径及原128头slot保留。

定义原金融Y。参考cell c需8条全部完整，才得均值Qhat_c；主target按原四slot multiset中心化：tauhat_i=Qhat_c(i)−sum_four_slots(Qhat)/4，重复订单保留原权重，不能对unique orders等权。它不是精确Q或A^π，hold也不是V^π。

每个estimate replicate独立形成原四slot组，以下六算子同一输入与原四均值：zero、ungated_plain、family_gate_plain、order_gate_plain、family_pool_order_gate、exact_pool_order_gate。三个order-gate算子只改变pool键；pool用对应cell估计Y均值再减原四slot均值，singleton保持原值。η统一0.25仅派生，不加入GAE、whitening、学习baseline或参数更新。

主分母始终32组×2rep=64组rep/256槽；原3个partial组×2=6组rep的24槽全部NA。其他组还需四估计路径完整、所有所需参考cell各8条完整；任一缺失整组NA，不能平均完成子集冒充无条件Q。六方法同支持交集，额外报告order gate打开/关闭分层，避免18个全同订单组掩盖比较。

报告raw金融Y平方单位下的逐槽/逐组平方误差、同支持成对loss差，以及参考0…3/4…7两半各自按同multiset中心化的敏感度。两半方法差方向不稳只记不可判；即使方向一致也不称参考精确或正式显著。不给本批旧状态/重叠日期/两rep独立市场样本解释，不以MSE下降替代收益、排序或策略梯度结论。本轮不计算动作排序或正式p值。

GPU沿既有租约95GiB外层上限和2GiB物理余量，独立新controller/worker/driver保存身份。driver在GPU加载时完成，在首个请求前单独seal；reader在采样期间完成、读数前seal。原pilot、零请求启动失败及所有旧源码不改写。执行完成需真实CPU/GPU/service退出与预算、原终点、holder归还收据；科学结论按参考精度和支持限制报告。
