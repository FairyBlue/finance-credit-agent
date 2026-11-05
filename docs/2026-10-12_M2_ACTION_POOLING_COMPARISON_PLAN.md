# M2 第二步：动作池化四轮对照

2026-10-12，SGT（UTC+8）。用户授权开始[下一步规划](2026-10-11_M2_NEXT_STEPS.md)的第二步。本文在任何本批训练及模型响应前固定。第一步的[最终恢复验收](2026-10-11_M2_ORDER_RECOVERY_SFT_REPORT.md)已完成1200/1200、raw/行为/完成核验PASS；本批从同一新共同SFT重新训练，不导入R5或第一步单轮actor作为新对照臂。

## 固定对照与预算

| 臂 | Actor LR | η | 用途 |
|---|---:|---:|---|
| M1 | 1e-7 | 0 | 相同LR下residual整套增量 |
| M1 | 2e-7 | 0 | 较强M1设置参考 |
| M2-plain | 1e-7 | 0.25 | 同gate、同位置的普通组相对信号 |
| M2 | 1e-7 | 0.25 | 动作池化相对plain的增量 |

各臂使用seed42/43、四轮，每轮2个窗口×4个siblings＝8条训练轨迹，每臂32条；actor及critic各四次更新，critic LR统一1e-6。seed42按上表顺序，seed43反序；每个seed先评估共同SFT0。SFT两个请求身份共享一份权重，不能算两个独立训练样本。

训练窗口固定`0001/0065/0009/0073/0017/0149/0025/0157`，每轮顺序和配对seed沿用原训练覆盖；评估为`0210/0231/0252/0273/0294/0315/0336/0357/0378/0399/0420/0441`。所有均为原Train开发窗口，已接触奖励标定或选参，不称untouched holdout。

第2及第4轮各12窗×（greedy1＋sampled4）；SFT0每seed60条。全批固定**1336条＝256训练＋1080评估**，包含8个RL臂、32次actor及32次critic更新。不扩大LR/η网格，不新增seed、日期或预算。四轮只在global_step4导出最终actor；第2轮仅评估。

## 唯一消融差别

同一episode occurrence、同policy、同推进session位置的四个siblings中，令`D_j`表示至少两种合法推进动作family产生分歧；使用原金融终局`Y_i`。

- plain：`S_ij = D_j × (Y_i − mean_k(Y_k))`。
- M2：`S_ij = D_j × (mean_{k: a_kj=a_ij}(Y_k) − mean_k(Y_k))`。
- 两者均为`A_actor = global_real_step_whiten(U_GAE + η×S)`，共同critic targets仅由既有金融credit及拒单惩罚得到，不加入residual。
- retrieve/compute/invalid的residual仍为0，动作family、对齐位置、分歧gate、GAE、whitening、critic规则、η均相同；只改变池化。plain包含项目的critic与位置gate，不能称为标准GRPO。

所有臂共同初始化指纹`083ad9e7f56d02a3e7dd376571656d1e3af05c1ec62ba7b2467a59c36d637fa5`，使用`history_features_recovery_v2`、相同API/nonthinking提示和0.05训练拒单惩罚。critic head在同seed内包括SFT0的五个cell必须相同。原始奖励标定、交易执行、成本、数据、reference model及critic初始骨干保持共同合同。

M2-plain的核心计算原已存在，本次只补共同惩罚、critic初始化及配对评估的入口注册，不改变旧M1/M2默认公式。所有源码在prepare/preflight后统一封存，活跃运行严格核验当前源码。第一步已完成产物按其历史ZIP及原PASS收据校验，不改写旧manifest或要求旧源码等于新源码。

## 评估、诊断与判读

保持T=1、top_p=1、top_k=-1的sampled四路及greedy一路；非thinking，prompt上限8192、response上限4096、总请求上限256。原生评估沿用该总上限，不冒充另行执行了第一步的四次后续请求严格门禁；原始行为记录另统计连续拒单和恢复情况。没有自动hold、自动改动作或增加上限。

只有评估的真实256-step未完成prefix可按`retain_prefix_v1`保留；终局收益和金融Y为空，不当作0。训练未完成、身份缺失、基础设施异常或超出输入/响应边界仍硬停止。预定槽位全部计入覆盖；共同完整槽位上的配对差值必须同时展示排除数，不筛掉失败后声称全覆盖。

第4轮sampled配对净收益是主开发读数，同时报告金融Y、回撤、波动、换手、费用、成交、完整率及逐窗差值；greedy独立呈现。第2轮只用于趋势观察，不据其选择较优checkpoint。固定比较：M2−plain、plain−M1(1e-7)、M2−M1(1e-7)、M2−M1(2e-7)，以及各臂−SFT。每个seed分别报告，两seed均值不能掩盖方向相反。

并行保留residual/GAE尺度、分歧覆盖、pooled与plain信号变化、actor优势符号翻转及更新前后行为。运行诊断的`std_S`已含η，尺度比为`std_S/std_U`，不再次乘η。[旧credit审计](../runs/m2-order-recovery-audit-20261011/m2_credit_audit.json)仅作机制依据，未替代新行为证据。

若M2不优于plain，不主张动作池化有效；若不能超过较强M1，不主张调优后性能优势。即使两个开发seed同向，也仅支持下一步独立确认。正式Valid与FinalOOS继续关闭，所有执行完成状态与方法优势判断分开。

## 存储与持久执行

当前约67GiB磁盘不足四轮通用门槛。本批保留原**107GiB实际可用空间**准入：90GiB moment代际共存、9GiB当前HF导出、8GiB reserve；不能借用第一步的一轮30GiB预算，也不把已删除字节加回df。

启动前仅对固定六组已存在的旧4B actor做无损XOR归档，baseline仍保留原共同SFT；包括一组经核验的两路径硬链接，按一份物理文件计数。归档写新`runs/m2-action-pooling-storage-20261012-v1/`，全部原shard解码SHA一致后，另一步才允许退役清单内原始shards。原summary、lineage、export receipt、token/轨迹/财务证据及HF元数据保留，模型可按原SHA恢复。第一步四个新actor、新共同SFT、旧约24GiB moment暂存均不触碰。

本批每个成功RL臂先核验真实四次更新及HF导出，再退役其owned临时moments、无损归档新actor。仅在这些已核验退役后最多等300秒真实df归还，满足107GiB才首次启动下一臂；失败即停止，无隐式重试。全部八个最终actor以可逐字节恢复的归档保留，baseline不退役。

独立systemd user服务，Restart=no、KillMode=control-group、MemoryMax64G、RuntimeMax48h；原controller直接为主进程，结束后ExecStopPost记录本次真实退出码，避免wrapper占用协调器cgroup。沿用持久holder的显存交接和总体80GiB停止线，不操作外部进程；启动时按实际GPU资源准入。服务、模型、源码及预算全部绑定新合同，不覆盖旧批次。

当前状态：实现及CPU检查进行中，尚未启动本批训练。实际启动时间与收据另记执行报告，本文封存后不改写。
