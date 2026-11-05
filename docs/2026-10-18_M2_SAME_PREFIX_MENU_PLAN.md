# M2 同前缀分支与合法订单菜单验证方案

2026-10-18，SGT（UTC+8）。用户已批准继续[上一轮结果](2026-10-17_M2_STATE_ORDER_ORACLE_REPORT.md)中的下一步。本轮先完成 CPU 接口与账务验证，输出可供后续模型同状态取样使用的输入包；不发起模型生成、训练或正式 Valid / Final OOS 评估。

## 固定状态与真实模型输入

沿用上一轮全部 256 个 anchor：M2、seed42/43、8 个已固定 Train 窗口、零起算 session 0/5/10/15、原四兄弟各自的成功推进请求前状态，不按表现换样本。每个分支均重新 reset，并逐条重放该请求之前的**全部**历史请求，包括同 session 的拒单、retrieve 和 compute；逐项核对 interaction 与 credit trace。

保留 reset 或最后一次真实 step 返回的 Observation.messages，不通过重新调用 `_state_text` 构造替代观察。原环境按 prompt prefix 加当前观察构造请求，历史信息由 recent_events、tool_result_memory 和拒单反馈承载；不额外拼接全部 assistant 对话。分别绑定完整金融风险状态、实际 observation messages、工具 schemas、动作协议与输入 token IDs。

使用原共同 SFT 导出的本地 tokenizer，按原 single_tool_portfolio_v1、工具 schemas、add_generation_prompt 与 enable_thinking=False 渲染，仅作 CPU tokenization；每个 anchor 的 token IDs 和 SHA 必须与对应历史 request 逐 token 相同。只读取 tokenizer/config 文件，不加载模型权重。四个未来取样槽共享完全相同的起始输入，但本轮不实际生成四条模型样本。

## 菜单只由当前可见状态决定

纯函数仅接受真实观察中的 market、positions、当前日期及 session 等字段，不接受环境、未来价格、旧 Y/Q 或后续历史动作。保持原冻结 20 股池；按 symbol 字典序，对固定目标网格 `{0, 0.1, 0.2}`：

- 目标比当前可见 weight 高超过 `1e-6`，加入 buy；低超过 `1e-6`，加入 sell；其余不加入。
- 永远加入 hold，具体订单以工具名及完整参数去重，按固定顺序保存。
- `1e-6` 为对可见权重 8 位舍入的保守间隔；目标网格是新增诊断菜单，不能代表原连续动作空间。

该菜单表示请求合法性，不保证次日开盘成交。环境 position_limit / turnover_limit 是终分风险罚项，不是硬拒单上限。不得依据未来 fill、skip、终分或旧收益筛选菜单，也不按可见现金过滤买入；真实账务继续处理现金截断。网格外原订单只计 off-menu，不映射或修正成菜单订单。

先完成全部 256 个状态与菜单的构造，将 anchors 和真实 prompt packets 的 SHA 落盘，再执行第一笔新干预。菜单构造期间不调用 next-open 分支；历史 prefix 重放仍按原输入和数据门禁执行。源码、方案、tokenizer 和输入身份在这一阶段前另行封存。

## 分支预算与核验

对每个 anchor 菜单中的每个订单，以独立 reset + 完整 prefix 执行一次真实 env.step，核对起始金融状态及实际观察、只推进一个 session、完整风险历史延续、费用与现金/股份守恒。合法 next-open skip 与零 fill 全部保留。若预称合法的菜单请求被拒，保留 NA 和报错，工程结果不能记 PASS，不修目标、不补 hold、不换状态。

额外固定四个脚本探针槽：`[hold, 字典序首个 buy, 同一个 buy, 字典序最后一个 sell（若无 sell 则最后一个 buy）]`。选单仅依赖已封存菜单。若无 buy，保留该状态并记支持失败，不以 hold 冒充 buy。至多三个不同菜单订单加一个独立重复 buy，固定 hold 至原窗口终点；重复 buy 必须独立重放，不用缓存替代。四槽是脚本接口探针，不是策略采样。

另对全部 256 个原对角订单独立重放至原终点，固定 hold 后续；其第一步 info 必须精确匹配原轨迹，终分匹配上一轮同状态对角 Q。菜单 hold 同样匹配上一轮 H，验证费用、风险历史及终点没有回退。

每状态网格最多 `1+20×3=61` 个不同订单。总预算上限为 15616 次菜单 first-step、256 次重复 buy、256 次原对角控制，共 16128 次 first-step；完整终点分支最多 1280 个。prefix 与后缀 hold 的环境调用另计，实际计数落盘。不跨 anchor 缓存候选结果，不补采失败。

## 完成标准与解释边界

完成标准：256 个状态全部保留；token 输入逐项匹配；所有菜单在新干预前冻结；菜单请求全部合法推进或完整报告失败；全部分支起始状态和观察一致，费用、股份与风险历史核对通过；256 个重复 buy、原对角控制及 hold 终点一致。报告菜单大小、买卖覆盖、原订单 off-menu、合法零 fill、skip 和各类实际执行分母。

本轮不比较新的 pooling MSE 或排序，不把确定性重复一致称为降噪，不从脚本探针推断模型交易能力或收益。菜单若用于未来模型提示或约束解码，将构成新的动作协议，须与原自由工具输出分别定义；不能把本轮脚本菜单视作从原策略抽得的样本。

通过后准备一个有界的同状态模型取样方案：明确原自由输出还是菜单协议、完整 prefix 与共同模型版本、n=4 覆盖、非法与非推进处理、后续策略及终点。未来 hold oracle 仍仅用于离线诊断，不输入模型或学习 baseline。当前无新增 GPU 或 RL 任务。
