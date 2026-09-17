# A3-M1 / 1.7B 接线结果（精简）

历史结果日期：2026-09-02；整理日期：2026-09-14。**A3-M1 WIRING SMOKE PASSED**。
本次是 G1 之后的独立工程验证，不是正式 M1 消融或收益比较；最新实验见[文档导航](README.md)。

M1 采用 uniform immediate-step reward：完整轨迹将终局 A_fin 均分到实际 interaction steps，
最后一步吸收浮点余差。它进入真实 reward timeline，之后仍由 GAE、critic 和 GSPO actor
计算更新，不是将 optimizer advantage 均匀广播。

| 项目 | 实际结果 |
|---|---|
| 已验收代码 | `fb07e727b8077c9f25684ad3bfa5222feef84035` |
| 模型 / 资源规模 | Qwen3-1.7B，seed42，n=1，train batch1，单 H100 NVL |
| 轨迹 | 更新前 Valid、Train、更新后 Valid 各 1 条，共 63 steps / requests |
| critic / actor | 各 1 次真实更新，梯度和抽样参数变化有限非零 |
| 完整性 | 3/3 到终局，step credits 与 A_fin 守恒，seed/credit 轨迹身份一致 |
| 正常结束 | launcher/trainer 退出 0，Final OOS 未访问 |

| 阶段 | steps | A_fin / credit sum |
|---|---:|---:|
| Initial Valid | 21 | -55.28775219319622 |
| Train | 20 | 1.5457017480377786 |
| Post-update Valid | 22 | -55.28775219319622 |

两次 Valid 分数相同而轨迹步数不同，不能解释为性能提升。本次 save_freq=-1，没有完整
checkpoint；参数变化来自固定 4096 元素样本，不是全模型质量指标。

早期 host 限流问题通过当时的 critic 参数常驻 GPU、BF16 reference 和小 batch 布局缓解；
该配置只用于 wiring。完整 global step 约 154 秒，总作业含预检/初始化约 25 分钟。
后续 4B 资源布局与验收见[screening 工程总结](history/02_M2_SCREENING_ENGINEERING.md)。

原始证据：`runs/20260902T070540291149Z-smoke-a3-m1-qwen3-1p7b-fb07e727/`，
`summary.json` 原 SHA-256 为
`99329dbf4c624a611a06505af96164fbdbace5acf221b186ebcf5cb27d908fdf`。
原始提交、完整轨迹/更新数表和文件哈希可从[原文归档](archive/README.md)查回。

[G1 精简结果](2026-08-31_H100_G1_SMOKE_RESULTS.md) ·
[阶段总结](history/01_H100_AND_M1.md) · [评价协议](../EVAL_PROTOCOL.md)。
