# 检索日志（novelty / 事实核验）

> 日期与查询串留痕，供"To our knowledge"声明溯源。均为 arXiv API / HF / 网络检索。

| 日期 | 查询 | 结果 | 判读 |
|---|---|---|---|
| 2026-09-08 | arXiv all:"UltraX" | 仅原论文 2607.08646 | 无 follow-up 切片审计（as-of 当日） |
| 2026-09-08 | arXiv abs:"perplexity"+filtering+long-tail / cleaning+rare | 0 命中 | 分层 PPL 审计清洗流水线空位 |
| 2026-09-08 | arXiv long-tail AND synthetic data AND minority | 1 篇（CV，2408.16273） | LLM/数据治理侧空位 |
| 2026-09-08 | UltraX 全文（arxiv html 2607.08646v1） | §4.2: 16B=45.49 > 20B Raw 45.08/ProX-C 45.05；Table 9: Ultra-FineWeb docs 99.26%、tokens −3.8%；§4.4 声称"优势不来自删除更多" | 16B>20B 与保留率统计 **VERIFIED**；0.6B 仅在模型卡（非正文） |
| 2026-09-08 | 质量过滤长尾偏差（候选引文） | 2605.06901、2303.18223、Marion/When-Less-is-More、Lee/dedup | **待核**：写入 Related work 前逐条查证 |
