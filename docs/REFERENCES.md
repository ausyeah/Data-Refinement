# 参考文献核验表（SliceAudit）

> **诚信红线**：只收录经检索核实的一手源。`已核` = 直接检索确认；`转述` = 二手提及，需再核。
> 写作前逐条复核格式（作者/年份/会议/编号），禁止凭记忆填写。

## 核心被测对象与数据
| 引用 | 标识 | 状态 | 用途 |
|---|---|---|---|
| UltraX | arXiv **2607.08646**（OpenBMB/MiniCPM 系） | 已核 | 被测对象 |
| UltraX-Preview 数据集 | HF `openbmb/UltraX-Preview`（Apache-2.0） | 已核 | 实验数据 |
| MiniCPM5-2B | HF `openbmb/MiniCPM5-2B` + GitHub（**无 arXiv 论文**） | 已核 | 背景（不可当论文引） |
| Tiered Data Management (UltraData L0–L4) | arXiv 2602.09003 | 已核 | 数据治理框架参照 |

## 阶段 A 相关
| 引用 | 标识 | 状态 | 用途 |
|---|---|---|---|
| wordfreq zipf | Robyn Speer, wordfreq 库（Zenodo/readthedocs） | 已核 | 稀有度代理 |
| PopQA | Mallen et al., ACL 2023, arXiv 2212.10511 | 已核 | 外部参照（预期地板） |
| 小模型知识基准地板数字（Pythia-410M: MMLU≈25.5–27.3, GSM8K 0.7） | 见 2510.21866 等缩放研究 | 转述 | 地板效应依据（需再核原始出处） |

## 阶段 B 方法理由（关键引用）
| 引用 | 标识 | 状态 | 用途 |
|---|---|---|---|
| DAPT（域自适应预训练） | Gururangan et al., ACL 2020, arXiv 2004.10964 | 已核 | 证明 CPT 中规模即有效（对照组） |
| 事实记忆与遗忘（幂律、边际事实先遗忘） | Chang et al., NeurIPS 2024, arXiv 2406.11813 | 已核 | 证明 CPT 对长尾有遗忘偏置 → 从零训理由 #2 |
| 知识记忆规模阈值（70M–30B 知识任务 ~19–20%） | arXiv 2510.21866 | 转述 | 地板效应（需再核） |

## 暂缓/禁止
| 对象 | 说明 |
|---|---|
| ProX | 未检索到论文编号，**禁止引用**（仅有 FineWeb-ProX-Doc 数据集） |
| 机器之心公众号（2026-09-08 MiniCPM5/UltraX 报道） | 不可引用，事实已回溯一手源 |
| Prismatic / G-Vendi | 会议标注未确认，如需引用只写 arXiv 2505.20161 并标注"待核会议" |
