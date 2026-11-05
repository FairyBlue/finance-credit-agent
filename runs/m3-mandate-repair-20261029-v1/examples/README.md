# 原模型的真实任务演示

原模型在预定的全部 100 个开发任务上成功 **1/100**。这两份案例来自这批真实模型轨迹；原始评估文件保留，未重新生成模型回答。此处展示的是模拟委托、真实历史 Train 报价和固定已知报价成交下的工具任务，不涉及预测未来收益，也不代表市场 alpha。

- [original_success.json](original_success.json)：唯一成功任务 `mandate-0411-v1`，因此选择规则明确。模型先读分类与限制，过早构造过强约束得到不可行结果，随后读取有效修订、重新求解，并在第 8 次调用提交正确方案。这是一条成功案例，不代表整体表现。
- [original_typical_failure.json](original_typical_failure.json)：按开发任务文件的固定顺序取第一个失败 `mandate-0400-v0`，没有按失败特征挑选。模型读过相关材料，但第 6–8 次 `solve` 都缺少接口要求的 `constraints` 外层字段，工具返回真实错误，8 次用尽仍未提交。

每份 JSON 保存公开初始消息、模型原始文本、逐次工具返回、生成元数据及终局指标。`exchanges` 是当时实际记录；`terminal_result` 使用补齐不可行证明检查后的统一重放评分。评分重放未改变这两条轨迹的成败和费用。`replay.raw_tool_output_exact_match_by_call` 显式记录当前环境返回与原始返回是否完全一致，避免将后补的验证字段冒充当时输出。

从项目根目录执行下列代码，可重放已记录动作；无需加载模型或 GPU：

```python
import json
from pathlib import Path
from src.agents.mandate_repair_env import MandateRepairEnv

root = Path("runs/m3-mandate-repair-20261029-v1")
demo = json.loads((root / "examples/original_success.json").read_text())
tasks = map(json.loads, (root / "data/development.jsonl").open())
task = next(t for t in tasks if t["task_id"] == demo["task_id"])
env = MandateRepairEnv(task)
for exchange in demo["exchanges"]:
    env.step(exchange["model_response_text"])
assert env.result()["success"] == demo["terminal_result"]["success"]
print(env.result()["success"], env.result()["metrics"])
```

完整的 [original_error_analysis.json](../baselines/original_error_analysis.json) 保留 100 个任务的固定分母：97 个未形成有效提交，2 个提交同时存在证据与业务问题，1 个成功，系统或生成失败为 0。独占分类优先记证据问题；重叠标签仍保留两条业务失败，不能将其误读为“没有业务错误”。99 个失败中，98 个观察到求解调用错误；全批工具返回 391 次缺少 `constraints`、1 次引用未报价资产。现有证据首先指向新工具接口使用和按预算完成流程的问题，不能仅据此断言模型缺乏金融推理能力。

这些文件只读取开发集和既有轨迹；没有读取保留测试集、Formal Valid 或 Final OOS。

## 同一任务的 SFT 对照

固定一轮共同 SFT 后，完整开发集成功 **92/100**，100/100 均完成有效提交。[sft_same_task_success.json](sft_same_task_success.json) 使用已经选定的原模型首失败 `mandate-0400-v0`，公开初始消息逐字相同；没有根据 SFT 结果另挑任务。该任务原本无需交易：SFT 模型用 7 次调用完成检索、读取、计算、合法求解及提交，实际交易费用为 0。将上面的重放代码改为读取这份 JSON 即可复现。

流程改善与新组合泛化需要分开判断。原模型的主要失败是工具参数错误和未提交；SFT 的 8 个失败均来自新的 mixed 组合，对应 4 个基础情景的两种改写。它们读到了正确有效修订、限制和行业表，行业映射及买卖限制也正确，却把文档中的 `Financials: 0.16` 写成求解输入的 `Technology: 0.20`（7 条）或 `Technology: 0.16`（1 条）。求解器正常满足了模型传入的错误问题，实际金融行业权重仍超出 16%。

[sft_error_analysis.json](../baselines/sft_error_analysis.json) 保存逐条证据和字段差异。16 个 mixed 任务中，完整约束输入只有 2 个与有效委托完全一致；另外 6 个虽有漏项或错项，最终组合仍恰好满足真实条款而获得任务成功。因此 92% 表示既定终局任务成功率，不能解读成 92% 的完整条款抽取准确率。以上是只读诊断；本轮不据开发集改数据、奖励、模型或训练预算，字段核对与组合泛化的进一步改进留到下一轮。
