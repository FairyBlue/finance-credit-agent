# 维护与变更记录

## 1. 仓库与提交身份

2026-11-05起，项目由FairyBlue独立维护，GitHub同步目标为
[`FairyBlue/finance-credit-agent`](https://github.com/FairyBlue/finance-credit-agent)。
2026-11-11按用户要求改为公开仓库，公开实现、研究过程与实际结果。
不再向原合作者的仓库同步；原有提交历史及作者记录保留，后续提交使用FairyBlue身份。
本地`origin`应指向`git@github.com:FairyBlue/finance-credit-agent.git`。

## 2. 日常保存与变更报告

日常改动可以直接提交到`main`；需要隔离工作时使用分支与可选的Draft Pull Request。
提交或PR说明简要记录目的、影响和下一步，例如：

```text
docs: summarize credit mechanism results

目的：保存已完成的功劳分配机制验证
影响：更新文档中的当前状态和证据链接
下一步：按当前滚动计划继续研究
```

`.github/workflows/change-broadcast.yml`在`main`更新和PR变化时生成变更报告，
检查`git diff --check`并保存14天可下载的Actions artifact。
它不再发布Issue或PR评论，也不读取`CHANGE_BROADCAST_MENTIONS`。
变更报告不增加审批要求；测试、实验冻结合同与实际运行收据仍按原协议执行。

## 3. 同步范围

Git保存代码、配置、文档、文档历史归档和选定的关键实验报告、汇总及身份收据。
`runs/`默认忽略，已明确纳入版本控制的小型结果文件会继续跟踪；迁移时的文件清单和哈希见
[迁移快照](results/2026-11-05-backup/README.md)；最新补充见
[11月11日结果快照](results/2026-11-11-public-update/README.md)。
模型权重、optimizer、原始行情/财报数据、缓存、完整请求/响应与训练轨迹保留在本地。
本仓库的关键结果不能单独恢复模型或完整训练现场。

## 4. 与实验版本的关系

变更播报记录“仓库发生了什么”，不改变实验身份。G1 与 M1 接线均已完成，见
[历史总结](history/FOUNDATIONS_AND_DIAGNOSTICS.md)；后续实验使用各自合同封存的 exact commit、源码/配置
SHA 和运行收据。文档或自动化更新后的 `main` HEAD 不能冒充已经验证的 frozen code。

当前入口是[实验接手指南](README.md)和[文档导航](README.md)。旧结论从阶段总结查阅；
历史执行合同与 dated backup 继续保留，原始哈希核验使用
[原文 ZIP](archive/pre_2026-10-10_documents.zip)和 [manifest](archive/pre_2026-10-10_manifest.json)；
最新进度读导航，不从旧计划推定仍有待执行任务。

## 5. GPU与CPU工作顺序

2026-10-18用户明确要求优先有效使用稀缺GPU资源。已验收的同一模型和数据可复用身份收据，
重复全量哈希、详尽CPU预处理、账务重算和大规模夹具尽量后补，或与GPU采样并行；GPU被其他任务
占用时再集中完成这些CPU工作。启动前保留真实模型/输入对应、预算、显存租约与必要的实际接线检查。
日志区分GPU加载/warmup、实际模型请求与占位，按实际有效采样进度报告。

## 6. 长期Brief、滚动三步与研究回顾

2026-10-30用户明确：`RESEARCH_BRIEF.md`作为长期研究目标，具体三步在`docs/`迭代维护。
Brief只更新研究问题、核心假设、候选贡献、证据边界和重要方向变化；不堆近期任务、预算和运行日志。
每轮新三步使用带日期的独立文档，`CURRENT_STATUS.md`与文档导航指向当前计划，旧计划保留实际结果或被替代状态。

每轮三步结束，对照Brief的研究问题简短回顾“新增支持、仍缺什么、是否调整”；出现关键反证或持续一段时间没有推进时提前回顾。
没有长期方向变化时，只在状态页记录回顾结论。此约定不新增审批、定时任务或阻塞GPU试验的审计流程。
