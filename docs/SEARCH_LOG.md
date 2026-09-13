# 检索日志（novelty / 事实核验）

> 日期与查询串留痕，供"To our knowledge"声明溯源。均为 arXiv API / HF / 网络检索。

| 日期 | 查询 | 结果 | 判读 |
|---|---|---|---|
| 2026-09-08 | arXiv all:"UltraX" | 仅原论文 2607.08646 | 无 follow-up 切片审计（as-of 当日） |
| 2026-09-08 | arXiv abs:"perplexity"+filtering+long-tail / cleaning+rare | 0 命中 | 分层 PPL 审计清洗流水线空位 |
| 2026-09-08 | arXiv long-tail AND synthetic data AND minority | 1 篇（CV，2408.16273） | LLM/数据治理侧空位 |
| 2026-09-08 | UltraX 全文（arxiv html 2607.08646v1） | §4.2: 16B=45.49 > 20B Raw 45.08/ProX-C 45.05；Table 9: Ultra-FineWeb docs 99.26%、tokens −3.8%；§4.4 声称"优势不来自删除更多" | 16B>20B 与保留率统计 **VERIFIED**；0.6B 仅在模型卡（非正文） |
| 2026-09-08 | 质量过滤长尾偏差（候选引文） | 2605.06901、2303.18223、Marion/When-Less-is-More、Lee/dedup | **待核**：写入 Related work 前逐条查证 |
| 2026-09-13 | 核实 2605.06901 | "Reflections and New Directions for Human-Centered LLMs"（Ziems/Zhao/…/Diyi Yang，位置论文） | **非对手、非引文**——与过滤偏斜无关，从候选清单剔除 |
| 2026-09-13 | 核实 2303.18223 | "A Survey of Large Language Models"（Zhao et al. 综述，FCS 2026 刊出版） | 非对手——综述，仅可作背景引用 |
| 2026-09-13 | Web 检索 + OpenReview/arXiv 深挖 | **近邻确认：Apple "The data-quality illusion: Rethinking Classifier-based Quality Filtering for LLM Pretraining"（arXiv 2510.00866，Nait Saada/Béthune/…/Grangier/Ablin，ICLR 2026 接收）**：论证 CQF 的 per-example loss ≈ 概念频率代理 → 硬过滤不成比例地删长尾概念；修复 = 按概念频率加权损失。同队 ICML 2026 "Removing Noise, not Finding Gold"（同 arXiv 号演化版）谈 CQF 隐式过滤高质量集，与长尾无关 | **部分撞车（主张层）**："质量过滤伤长尾"已非首创。**差异化仍在**：①被测对象=部署级流水线（UltraX 编辑操作留痕）而非通用 CQF；②机制=位置性尾截（机械零模型证明非频率感知）≠ 他们的频率感知解释；③E2 从零训探针测切片级下游代价（他们没做）；④Stage C 廉价指标校准（空位仍在）。Related work 必须引并定位 |
| 2026-09-13 | UltraX follow-up（arXiv API/S2 限流，WebSearch 补查） | 未发现直接审计 UltraX 的论文（as-of 2026-09-13；9-8 检索亦无） | "对具体流水线的切片级取证"空位仍在，限流未扫全，投前建议再扫一轮 |
