# M2 训练后退化：日期与解码方式交叉诊断

制定时间：**2026-09-14 20:10 SGT（UTC+8）**。新增 GPU 评估尚未启动。

用户同意按[八轮报告的下一步](2026-09-14_M2_TRAINING_TUNING_REPORT.md)检查 actor 更新、训练回报与
持续训练后验收退化的关系。本轮先使用已保存模型定位问题，再决定训练调整。

## 为什么先做这项检查

对齐相同八个训练窗口后，第 2 epoch 的 sampled 净收益均值比第 1 epoch 上升：
M1 约 2.6494%→2.9798%，M2 η=1 约 2.6500%→3.1914%，η=0.25 约 2.6636%→3.2523%；
下降的是另一组日期上的 greedy 验收。旧比较同时改变了模型阶段、日期和解码方式，
不能直接得出“actor 更新方向反了”或“训练目标全面恶化”。同窗两 epoch 的 sampled 轨迹
也并非同一组样本；此处只把它作为需要进一步区分因素的线索。

静态核对未发现 actor loss 符号、response mask 或微批归一化的明确 bug。
当前每批一个 minibatch、一个 PPO epoch，Agent-R1 在 on-policy 分支把本次 forward 的
log_prob.detach() 用作 old anchor，所以 backward 时 ratio=1、clipfrac=0 是该路径的自然结果。
这不能保证 optimizer 更新后的策略变化很小。若随后需要调整 actor，考虑增加同批 pre/post
概率变化测量，而不凭零 clip 指标认定更新稳定。

## 固定评估矩阵

| 模型 | 8 个训练窗口 | 4 个既有 Train 验收窗口 | 合计 |
|---|---|---|---:|
| v3 未 RL 的共同 SFT | 每窗 greedy 1 + sampled 4 | 每窗 greedy 1 + sampled 4 | 60 |
| M1，第八轮 actor | 同上 | 同上 | 60 |
| M2 η=1，第八轮 actor | 同上 | 同上 | 60 |
| M2 η=0.25，第八轮 actor | 同上 | 同上 | 60 |

总计 **240 条完整轨迹预算**（48 greedy、192 sampled），无参数训练。
训练窗口沿用 `0001/0005/0009/0013/0017/0021/0025/0029`，验收窗口沿用
`0113/0117/0121/0125`，完整 ID 前缀 `formal-train-`。每条 21 sessions，全部从原始现金状态开始，
不添加教师标签、持仓前缀或人工交易规则。两组日期都属于 Train；滚动窗口可重叠，不当作独立正式 OOS。

上下文直接来自上一轮封存的 train/dev parquet；保持账本、成本、calibration、scorer、
`history_features_v1`、`single_tool_portfolio_v1` 和 non-thinking。
四模型使用同一 standalone vLLM 服务配置、BF16、seed42、top_p=1、top_k=-1、repetition_penalty=1。
greedy temperature=0，sampled temperature=1、每请求 n=1，四条独立 sibling lane。
固定请求种子在四模型之间相同，但不等于旧训练第八轮的种子序列。

直接复用 coldstart_eval 的共同完整回合上限 **128 requests**，prompt cap8192、response cap4096。
原 trainer 的 agent 上限为256；本批为统一128的诊断，若触及任何 cap 记录 INCOMPLETE，
不自动追加预算、重抽或将拒单改成成交。每个模型新目录单次执行，缺失/重复/未结束轨迹不得记完整。

## 区分哪些解释

先在本批相同重载执行方式中，分别计算四模型相对 SFT 的四格读数：
训练日期×greedy、训练日期×sampled、验收日期×greedy、验收日期×sampled。
sampled 先对同窗四 siblings 等权平均，再对固定窗口等权平均，完整分母固定。
同时保留净收益、terminal Y、成本、合法/非法请求、hold、成交股票、末期持仓和现金。

- 若训练日期的两种模式改善、验收日期的两种模式下降，优先调查日期/状态泛化。
- 若两组日期 sampled 改善而 greedy 下降，优先调查采样训练与确定性部署之间的差异。
- 若两种因素都有影响，保留混合解释；少量样本不能据此宣布唯一原因。

新重载 greedy 与旧同-worker greedy 分栏核对。两种执行路径可能产生轨迹差异，
本批的主要比较只用共同 standalone 条件；不将跨引擎差异直接归因于训练，也不展开底层 vLLM 扩展排查。
无第2轮模型可重载，不能补造早期 checkpoint；已有第2/4/6轮轨迹与指标仍保留。

## 执行、产物与后续

新目录 `runs/m2-policy-transfer-eval-20260914`。只加载现存 SFT 和三个第八轮 BF16 actor，
不保存新大模型、不删除旧模型。源代码、12上下文、模型收据/身份在启动前固定；
保持旧训练源码与 runtime hooks 不变，新增入口放在 scripts/diagnostics。

由独立 systemd user service 运行，Linger=yes、Restart=no、日志在服务器；客户端断线不影响执行。
实时读数与最终汇总独立运行；完成后写同目录 `FINAL_RESULTS.md`，并更新本轮报告。
单个模型若异常退出，保留失败与其余当前记录，停止后续自动执行，等待分析。

本批不自动启动下一轮 RL。根据交叉结果，再决定是否用共同较小 actor LR（候选如2e-7）、
相同 critic LR1e-6 和同批更新测量进行新的有界 M1/M2 对照。

## 参考实现经验

PPO 原论文把采样和 surrogate objective 优化分开，不能把一次更新前的 clip 指标当作更新后行为保证。
本项目的具体 ratio=1 来自上面的本机代码核对。[PPO 原论文](https://arxiv.org/abs/1707.06347)

Hugging Face 的复现记录展示了采样温度、KL 和 Adam 实现细节对 PPO 行为的影响；
它提供了检查这些量的理由，并不替本项目确定根因，也不意味着要更换 optimizer。
[RLHF with PPO 实现复现](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
