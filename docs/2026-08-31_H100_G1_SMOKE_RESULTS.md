# G1：A1 / 1.7B 接线结果（精简）

历史结果日期：2026-08-31；整理日期：2026-09-14。**G1 PASSED**，只证明训练接线，
不构成收益或正式比较结论。完整过程见[H100 与 M1 阶段总结](history/01_H100_AND_M1.md)，
整理前原文及哈希见[归档清单](archive/README.md)。保留此路径供 SETUP 等已有文档引用。

| 项目 | 结果 |
|---|---|
| 模型 / 规模 | Qwen3-1.7B，单 H100 NVL，4 Train episodes，n=2，seed42 |
| 轨迹与请求 | 11 条轨迹，259 requests / 259 unique seeds |
| 真实更新 | 2 次 optimizer 更新，有限且非零的抽样参数变化 |
| 终止 | launcher/trainer 退出 0，未访问 Final OOS |
| 输入预算 | 620 个 cases，最大 3045/4096 prompt tokens |
| 已验收代码 | `2d48bf6db8961998b1450644ef542a293ea8f95d` |

第一个训练 step 出现非零 reward/advantage；第二个训练 step 和最终 validation mean reward
为零。第二步仍有非零 KL、梯度和参数变化，这不等于策略质量改善。参数探针只覆盖固定
4096 个元素；本次没有保存完整 checkpoint。

原始证据：`runs/20260831T013112864440Z-smoke-a1-qwen3-1p7b-2d48bf6d/`。
`summary.json` 原 SHA-256 为
`5e895710ca044d9d20f249cdd49d2808057b8935f667dd8e04a0ae22a2647932`。

随后 [T2 v2](2026-09-01_T2_V2_RESULTS.md) 和
[M1 单独接线](2026-09-02_H100_A3_M1_WIRING_SMOKE_RESULTS.md)分别通过；它们不替代本次
代码身份，也不把本次 G1 改为正式训练。最新进展见[文档导航](README.md)。
