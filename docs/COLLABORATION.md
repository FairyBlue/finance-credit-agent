# 两人协作与自动变更播报

## 1. 目的

这套机制只解决一件事：无论哪位合作者修改仓库，GitHub 都自动发布一份可读的变更说明，
让另一位合作者知道谁改了什么、为什么改、可能影响什么，以及下一步是什么。

双方使用完全相同的规则。这里不增加数字签名、W&B、额外审批平台或复杂安全流程。

## 2. 日常使用

推荐每项工作使用一个仓库内分支，并尽早打开 Draft Pull Request。PR 只需填写模板中的三项：

```text
目的：为什么修改
影响：会影响什么
下一步：合并后由谁继续做什么
```

打开 PR 后，每次继续 push，GitHub 都会更新 PR 中同一条“自动变更播报”，不会为每个小提交
重复刷屏。报告自动包含：

- GitHub 修改人、时间、分支和 commit 范围；
- 文件列表和改动规模；
- 代码、配置、数据规则、reward、依赖、测试、文档或协作自动化分类；
- 对实验边界的路径级提示；
- commit 记录、完整 diff 链接和 GitHub Checks 链接；
- `git diff --check` 和报告生成结果。

报告由 `scripts/generate_change_report.py` 根据 Git 差异生成。它不会用 AI 猜测研究动机；
“目的、影响、下一步”来自 PR 模板，缺失时会尝试读取最新 commit 中同名字段。

## 3. 合并后的正式播报

每次 `main` 更新，`.github/workflows/change-broadcast.yml` 会：

1. 自动创建或复用标签为 `change-broadcast`、标题为“自动变更播报”的固定 Issue；
2. 在该 Issue 中追加本次正式变更报告；
3. 保存一份 14 天可下载的 workflow artifact；
4. 如果固定 Issue 曾被关闭，自动重新打开。

第一次创建后，两位合作者都应在该 Issue 点击 **Subscribe**。此后无需人工转发。直接 push
到 `main` 也会播报，但日常改动仍推荐先走 PR，以便另一人看到合并前报告。

如果希望每次正式播报都明确 @ 两个人，可以在仓库中设置一个普通 Actions variable：

```text
Name: CHANGE_BROADCAST_MENTIONS
Value: @github-user-a @github-user-b
```

位置是 GitHub 仓库的 **Settings → Secrets and variables → Actions → Variables**。它不是密码，
不需要放进 Secrets；不设置也不影响播报，双方订阅固定 Issue 即可。

## 4. 提交说明

为了让直接推送或合并后的报告仍然可读，commit 可以沿用下面的简短格式：

```text
docs: add H100 validation notes

目的：同步本次 H100 验证结果
影响：只更新文档，不改变实验配置
下一步：合作者复核结果并决定是否进入 G1
```

不要求每个临时 commit 都写成长报告；PR 中填写三项即可。最终合并 commit 应保留准确标题。

## 5. 自动提示不等于实验批准

自动报告只陈述 Git 中可以验证的事实。它不能替代以下判断：

- 测试是否足以保护实验结论；
- 数据是否真的没有泄漏；
- reward 或交易环境是否科学正确；
- H100 smoke 或正式实验是否通过；
- 是否可以开放 Final OOS。

报告中的“未检测到实验风险路径”只表示文件路径没有命中已知规则，不表示实验已经安全。
完整 pytest、GPU 检查和实验门禁仍以各自的 GitHub Checks、运行产物和实验文档为准。

## 6. 失败处理

如果自动播报 workflow 失败：

1. PR 中查看失败的 `Change broadcast` Check；
2. 下载已生成的 artifact（如果生成阶段已完成）；
3. 修复 workflow 或报告脚本后重新运行；
4. 不要手工伪造机器人报告。

播报失败不会删除代码或运行产物，但表示另一位合作者可能没有收到完整变更说明，因此应在
合并下一项实验改动前恢复它。

## 7. 与实验版本的关系

变更播报记录“仓库发生了什么”，不改变实验的 commit 归属。比如 G1 仍然绑定冻结提交
`8a090b86f667216a28a3846520b717348b4a6764`；之后新增文档或协作自动化产生的新 `main` HEAD，
不能被写成 G1 已在新提交上验证。
