# 同前缀菜单 CPU 验证与输入包

2026-10-18，00:39:32 SGT（UTC+8）已完成，采集耗时16分43秒。10,892个菜单请求及256个重复buy、256个原订单控制全部合法，共11,404次实际分支、1,280个完整终点；采集及独立读数实际exit0、工程PASS。256份真实输入逐token匹配历史，全部前缀、费用与账务检查通过。脚本四槽探针不是模型采样，不代表模型交易或学习结果。

[执行前方案](../../docs/2026-10-18_M2_SAME_PREFIX_MENU_PLAN.md) · [当前状态](../../docs/CURRENT_STATUS.md) · [机制主题与结果](../../docs/history/CREDIT_ORACLE_DIAGNOSTICS.md)。

## 封存与输入

- `contract.json`：回放前封存的方案、源码、原数据及本地 tokenizer 文件 SHA；`source_manifest.json` 保留固定64条旧轨迹及256个决策位置。
- `anchors.jsonl`：完整金融状态、实际可见状态、菜单、脚本探针槽及原结果参照。原结果仅用于离线核对。
- `prompt_packets.jsonl`：真实 Observation.messages、原动作协议后的消息、工具 schema、模板参数、渲染文本及逐 token 匹配历史请求的输入 IDs。菜单和 oracle 标签未注入输入。
- `snapshot_completion.json`：在任何新分支结果产生前封存全部菜单与输入包的 SHA。金融状态与真实模型输入分别核验，未通过 `_state_text` 重建替代观察。

后续模型请求应只使用 `prompt_packets.jsonl` 中封存的 prompt IDs 及共同生成配置；`anchors.jsonl` 的旧金融终点与分支标签继续留在离线评价侧。恢复原输入并不表示恢复了历史训练轮次的模型权重。

## 实际执行与读数

- `branches.jsonl`：每个菜单订单实际执行一次 first-step，另有独立 `repeat_buy` 和 `original_control`。角色与 anchor 明确记录，没有候选结果缓存；原四槽仅是逻辑引用。
- `collection_completion.json`：实际环境请求、构造数量、first-step 和完整终点分支数、输出 SHA；`start_receipt.json` 绑定启动合同，`live_status.json` 记录当前阶段。
- `readout.json`、`checks.json`：独立重建菜单和来源、核验逐 token 旧请求、账务、重复与原对角控制，报告工程覆盖。
- `menu_coverage.csv`、`branch_summary.csv`：每状态的菜单大小、买卖支持、原订单 off-menu、合法 skip / 零成交，以及执行角色与日期分层。跨互斥分支相加的费用是工程计数，不能作为一个策略的成本或收益。
- `collector_fixture_receipt.json`：46项纯函数正反例检查；`tokenizer_smoke_receipt.json`：4例普通/拒单反馈的离线逐 token 核验。
- `snapshot_independent_audit.json`、`final_independent_audit.json`：完整输入与实际结果的独立核对；`execution_receipt.json`、`final_verification.json`：实际退出状态与最终SHA核验。夹具源码保存在`audit_scripts/collector_fixture.py`。
- `sampling_proposal.json`：下一项32前缀×4次首请求的固定清单、128个无碰撞seed和预算，状态为`PREPARED_NOT_STARTED`；未加载模型权重或启动GPU。它面向共同SFT在M2历史状态上的条件行为，不能冒充原各轮M2策略。

菜单覆盖原20股，目标网格为0%、10%、20%，方向与可见权重至少相差1e−6；它是有限诊断菜单，并非原连续动作空间。合法请求仍可能因次日开盘方向跨越而 skip，也可能因现金约束零成交。所有此类分支均保留，不依据未来是否成交改菜单。

## 在新目录复现

从仓库根目录执行。输出路径必须不存在，源码和原输入须符合合同；本轮目录与旧证据不覆盖。

```bash
.venv/bin/python -B scripts/diagnostics/m2_same_prefix_menu.py prepare \
  --output runs/m2-same-prefix-menu-reproduction
.venv/bin/python -u -B scripts/diagnostics/m2_same_prefix_menu.py run \
  --output runs/m2-same-prefix-menu-reproduction
.venv/bin/python -B scripts/diagnostics/m2_same_prefix_menu_readout.py \
  --run runs/m2-same-prefix-menu-reproduction
```

使用本地 tokenizer 进行 CPU tokenization，不加载模型权重或推理后端。先通过实际冻结行情的数据门禁，再逐请求还原 prefix。读数脚本只读已完成证据，不重跑环境。所有方法身份检查与当前CPU工程范围均不开放正式 Valid / Final OOS 或新训练。
