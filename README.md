# SliceAudit — Sliced Auditing of Data Refinement Pipelines

**Does industrial data refinement help everyone — or only the head of the distribution?**

> 中文说明见文末。

[![Preprint](https://img.shields.io/badge/preprint-ChinaXiv%20202609.00041-blue)](https://doi.org/10.12074/202609.00041)
[![Hardware](https://img.shields.io/badge/hardware-RTX%204060%20Laptop%208GB-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Preprint

**SliceAudit: Positional Tail-Truncation in LLM Data Refinement — A Sliced Audit of UltraX**
黄浩 (sole author) · ChinaXiv preprint, 2026-09-08

- 中文版：数据精炼的切片级审计：以 UltraX 为例的位置性尾截与稀有度代价测量
- DOI: [10.12074/202609.00041](https://doi.org/10.12074/202609.00041) · CSTR: 32003.36.ChinaXiv.202609.00041
- Cite as: `ChinaXiv:202609.00041V1`

> Preprint — not yet peer reviewed. Stage A (sliced audit) is complete; Stage B (downstream
> cost, E2) is the OPEN part of the paper and is reported as such.

## The question

Data quality is now a first-class lever in LLM training. FaceWall's **UltraX**
(arXiv:2607.08646) is a representative industrial refinement pipeline: a 0.6B model predicts
structured edit operations (`keep / delete / replace / insert`), and a deterministic executor
applies them to raw corpus text.

UltraX reports that a 1B model trained on 16B refined tokens beats one trained on 20B raw tokens.
**That is an aggregate result.** It does not say *for whom* refinement helps.

This project asks the question UltraX did not:

> **Are the edit operations systematically skewed on long-tail / rare concepts?
> And when does a cheap quality metric stop predicting actual gains?**

## Why this might be true

A refinement model is trained to maximise aggregate quality. Rare concepts are, by definition,
under-represented in that objective. There is therefore a plausible mechanism for
**tail-specific deletion**: content that is rare, unusual, or off-distribution looks
"low quality" to a model optimised for the head.

If that happens, the aggregate gain hides a **slice-level loss** — exactly the pattern that
aggregate metrics are known to mask.

## Method

The key design choice: **UltraX is the object under audit, not a tool we use.**
`openbmb/UltraX-Preview` ships a `(raw text, refined text, edit operations)` triple —
a ready-made controlled experiment with full intervention traces, available for free.

| Stage | What | Cost |
|---|---|---|
| **A** | Count edit-operation types stratified by rarity. Primary test: `deletion rate × rarity stratum` interaction. | **No GPU** |
| **B** | Train 150M–500M probes on raw vs. refined corpus; measure **sliced** downstream delta. | 8GB single card |
| **C** | Calibration curve: cheap quality metric ↔ sliced gain, and where it breaks. | 8GB single card |

### Dependent variable is a *sliced delta*, not an average

This is the line between an audit and a reproduction report. UltraX proved aggregate
effectiveness; it did not prove *for whom*. We report per-slice deltas.

## Pre-registered design

- **Primary endpoint (intersection–union test)**: (i) interaction term `edit × rarity` CI excludes 0
  **AND** (ii) tail-slice loss is significant under a binomial test. Both must hold.
- **Kill criterion**: if refinement is *also* better on the tail, the hypothesis is dead.
  We then run a **TOST equivalence test** and publish an informative null.
- **Confound control**: difficulty must use an **independent** proxy (reasoning steps / length /
  human annotation). **Pass rate and NLL are forbidden** as difficulty proxies — they are
  already rarity proxies, which would make the interaction term circular.
- **Negative controls**: deletion applied to *head* slices; unstructured noise at matched
  intensity; 500 label permutations for the null distribution of the interaction.
- **Training noise**: ≥3 (target 5) seeds with explicit variance decomposition.
  Sample-level bootstrap alone underestimates CI for QLoRA runs.

See [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) and [`configs/preregistration.md`](configs/preregistration.md).

## Repository layout

```
src/00_probe_ultrax.py   Stage A step 0 — stream-probe the real schema of UltraX-Preview
docs/RESEARCH_PLAN.md    Full research plan
configs/                 Pre-registration (frozen before any test-set result)
reports/                 Outputs (large intermediates gitignored)
```

## Quick start

```bash
# 1. Point HuggingFace cache at a large drive (see SETUP.md — C: is too small)
export HF_HOME=/e/hf-cache
export HF_ENDPOINT=https://hf-mirror.com      # China mirror

# 2. Install
pip install -r requirements.txt

# 3. Probe the dataset schema (streaming — does NOT download the full 487GB)
python src/00_probe_ultrax.py --n 200
```

## Roadmap

- [x] Research plan + pre-registration
- [x] **Stage A step 0**: confirm UltraX triple schema
- [x] Rarity stratification protocol
- [x] Stage A: edit-operation × rarity distribution *(no rarity main effect; deletion is positional)*
- [x] **Preprint published on ChinaXiv** (2026-09-08) — [DOI 10.12074/202609.00041](https://doi.org/10.12074/202609.00041)
- [ ] Stage B: probe training, sliced delta (E2) — *paused; reported as OPEN in the preprint*
- [ ] Stage C: calibration curve and failure boundary

## Citation

If you build on this audit, please cite the preprint:

```bibtex
@misc{huang2026sliceaudit,
  title  = {SliceAudit: Positional Tail-Truncation in LLM Data Refinement --- A Sliced Audit of UltraX},
  author = {Huang, Hao},
  year   = {2026},
  doi    = {10.12074/202609.00041},
  note   = {ChinaXiv preprint, ChinaXiv:202609.00041V1}
}
```

## Attribution

This work audits **UltraX** (arXiv:2607.08646) and uses `openbmb/UltraX-Preview`
(Apache-2.0). We are grateful to the OpenBMB / MiniCPM team for open-sourcing not just
weights, but the data-refinement pipeline and its intervention traces — that decision is
what makes this audit possible at all.

---

## 中文说明

本项目审计工业级数据精炼流水线（以面壁 UltraX 为**被测对象**）在**长尾/稀有概念**上的系统性偏斜。

UltraX 证明了聚合效用（1B 模型 16B 精炼 tokens 优于 20B 原始 tokens），但**未回答"对谁有效"**。
本项目统计其编辑操作（保留/删除/替换/插入）在稀有度分层上的分布，用 150M–500M 探针测分层 delta，
并给出廉价质量指标的失效边界。

核心设计：**因变量是分层 delta，不是平均分**——这是"审计"与"复现报告"的分界线。
全流程在单张 RTX 4060 Laptop 8GB 上可复现。

**状态**：阶段 A 已完成并发布 ChinaXiv 预印本（2026-09-08，DOI [10.12074/202609.00041](https://doi.org/10.12074/202609.00041)）；
阶段 B（下游代价 E2）暂停，稿中如实标为 OPEN。
