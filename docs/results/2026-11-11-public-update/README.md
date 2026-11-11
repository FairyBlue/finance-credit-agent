# 2026-11-11公开更新：V3—V6关键结果

采集时间：2026-11-11T13:55:24+08:00。本次补充48份结果与身份收据，共578,609字节。

公开入口：[仓库README](../../../README.md)、[当前状态](../../CURRENT_STATUS.md)、[当前金融Brief](../../../RESEARCH_BRIEF.md)。[Brief 2.0](../../../RESEARCH_BRIEF2.0.md)是尚未启动的未来备选方向。

| 运行 | 结论 | 完整成功 |
| --- | --- | --- |
| `agentic-rl-coverage-20261105-v3` | [报告](../../../runs/agentic-rl-coverage-20261105-v3/REPORT.md) · [结论](../../../runs/agentic-rl-coverage-20261105-v3/CONCLUSION.md) | 42/64 |
| `agentic-rl-step-focus-20261110-v4` | [报告](../../../runs/agentic-rl-step-focus-20261110-v4/REPORT.md) · [结论](../../../runs/agentic-rl-step-focus-20261110-v4/CONCLUSION.md) | 41/64 |
| `agentic-rl-invalid-feedback-20261110-v5` | [报告](../../../runs/agentic-rl-invalid-feedback-20261110-v5/REPORT.md) · [结论](../../../runs/agentic-rl-invalid-feedback-20261110-v5/CONCLUSION.md) | 44/64 |
| `agentic-rl-v5-dev-check-20261110` | [报告](../../../runs/agentic-rl-v5-dev-check-20261110/REPORT.md) · [结论](../../../runs/agentic-rl-v5-dev-check-20261110/CONCLUSION.md) | 80/100 Dev |
| `agentic-rl-action-q-20261111-v6` | [报告](../../../runs/agentic-rl-action-q-20261111-v6/REPORT.md) · [结论](../../../runs/agentic-rl-action-q-20261111-v6/CONCLUSION.md) | 42/64 |

原SFT在复用留出采样面板为43/64、旧Dev为82/100。各轮预算和训练内容不同，留出面板参与开发，不能按结果表声称独立方法确认或严格等算力收益。V6完整训练后没有增量，原SFT底座保留，当前仍优先金融探索。

[manifest.json](manifest.json)记录本次暂存结果的大小与SHA-256。文件保留原`runs/`路径，读取历史快照需使用对应Git提交；后续提交可以继续更新这些文件。

结果文件保留原文；模型权重、optimizer、原始数据、完整请求与训练轨迹不随本次公开上传。数据清单是身份元数据，不能代替原始数据或模型来完整复现训练。

本次新增代码的五份定向CPU测试文件合计22项通过。公开前扫描了全部可达Git历史、工作区候选文件及文档ZIP内文件，未命中所检查的凭据格式；新增选定结果同样通过。该格式扫描不构成对所有类型隐私内容的完整证明。
