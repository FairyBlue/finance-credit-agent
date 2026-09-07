# M2 screening 恢复执行记录（2026-09-07）

用户阅读当前报告并认可“先处理 4B 工程问题，再判断生成完整性问题”的顺序，现明确授权恢复。
本记录更新此前暂停状态；[05:53 UTC 实验报告](2026-09-07_M2_SCREENING_EXPERIMENT_REPORT.md)
仍作为暂停时的历史快照保留。

本次首先运行 R2 的 4B M2 Train-only 工程 pilot：相同两个 Train windows、seed 42、n=4、
两个完整 update rounds，逐 round 保存，并在同一 worker 中执行最终两条 Train greedy 评估。
generation、scorer、dtype、学习率和预算边界均沿用 R1/R2。模型从相同 exact base 初始化，
使用新目录，不把原失败重命名为成功，也不拼接旧样本。

运行入口为 `scripts/run_m2_screening.py --stage pilot --only-arm M2`；命名阶段日志用于定位
old-logprob、reference、value 和更新阶段的主机压力。R2 的 GPU 遥测错误不会主动终止 trainer。
如需进一步调整共同资源布局，将记录依据、验证和新代码版本后重跑必要 pilot。

当前授权的处理顺序不代表 1.7B 接线门禁已经通过，也不自动启动主 screening/dev。若 4B 也
出现生成不完整，先处理共同生成契约；若后续以 4B 接线替代 1.7B 门禁，需同步更新计划。

H100 继续使用 systemd user service、Linger=yes、stdin=null 和文件日志。当前进度、运行目录、
service、实际 commit 和失败证据更新在本机 `runs/M2_SCREENING_LATEST.md` 及对应 JSON 中。
