# 真实同题轨迹：首版SFT失败到新SFT成功

来自本轮固定200题开发集，未重采、未改工具上限、未读取保留测试。优先展示多行业上限的业务错误改正，各示例使用不同基础情景。这是说明性选择；整体效果以完整200题固定分母读数为准，不从三个案例推断平均优势。

两模型的每条实际请求均与同一任务初始观察及真实工具回放一致，JSON保存原始模型输出、工具响应和实际messages，不向提示加入gold。

- **composition-1012-v0**：Consumer≤4.5%、Industrials≤8%、Technology≤9.5%。[首版SFT失败](01_composition-1012-v0_original.json) → [新SFT成功](01_composition-1012-v0_sft.json)；实际工具调用7→8。原字段差异：`['cash_min', 'exposure_max', 'sector_max:Consumer', 'sector_max:Industrials', 'sector_max:Technology']`；新模型完整提取有效条款。

- **composition-1013-v1**：Consumer≤9.5%、Financials≤9%、Technology≤6%。[首版SFT失败](02_composition-1013-v1_original.json) → [新SFT成功](02_composition-1013-v1_sft.json)；实际工具调用8→8。原字段差异：`['sector_max:Consumer', 'sector_max:Financials']`；新模型完整提取有效条款。

- **composition-1014-v0**：Consumer≤3%、Financials≤15%、Industrials≤9%。[首版SFT失败](03_composition-1014-v0_original.json) → [新SFT成功](03_composition-1014-v0_sft.json)；实际工具调用7→8。原字段差异：`['sector_max:Consumer', 'sector_max:Financials', 'sector_max:Industrials', 'sector_max:Technology']`；新模型完整提取有效条款。

[完整端点错误归类](../summary/endpoint_error_analysis.json) · [固定分母结果](../REPORT.md)
