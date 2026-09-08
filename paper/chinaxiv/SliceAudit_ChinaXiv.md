# 数据精炼的切片级审计：以 UltraX 为例的位置性尾截与稀有度代价测量

作者：黄浩（北京邮电大学 信息与通信工程学院）

> 预印本说明：本文为投稿前预印本（preprint），尚未经同行评审。开源代码、预注册与全部结果
> JSON 存于 https://github.com/ausyeah/artical（MIT；正式公开前对审阅人开放）；审计轨迹以仓库
> 远端 main 的 commit 为准（docs/AUDIT_MANIFEST.md）。本文为 measurement-first audit：
> 阶段 A（切片审计）完成；阶段 B（下游代价 E2）按预注册协议进行中，属本文 OPEN 部分。
# 摘要

**目的[Objective]**　检验工业级数据精炼流水线（面壁 UltraX，arXiv 2607.08646）的编辑行为是否在长尾/稀有内容上存在系统性偏斜，以及其聚合收益声明（16B 精炼 tokens 优于 20B 原始 tokens）是否掩盖切片层面的代价。

**方法[Methods]**　以 UltraX 公开的「原文+精炼+编辑操作」三元组为被测对象，在 4 个英文语料（AICC/FineWeb/RedPajama-V2/Ultra-FineWeb，去重后 14.6 万篇）上做行级切片审计：稀有度用 wordfreq zipf 并剥离 OOV 结构通道；删除判定用行级模糊匹配并与操作日志交叉验证；主推断用文档聚类稳健 Logistic；并以「纯后缀截断」机械零模型分离流水线规则与语料自身形状；预注册确认性终点与修订日志公开。

**结果[Results]**　正文行内稀有度对删除无主效应（p=0.43）；删除以位置为主导；稀有度×位置交互被机械尾截完全复现（−0.966 vs −0.916），表明删除规则是位置性的而非稀有度感知的；词级分析（接触、探索性）显示被删稀有 token 的知识占比（74.7%）低于保留池（98.0%）与语料基准（94.0%）——相对比例上删除更偏向非知识 token，但绝对量仍移除约每文档 2 个知识型稀有词，净效应方向未测。

**局限[Limitations]**　本稿为测量侧审计，尚无下游代价证据（E2 未完成）；单一精炼系统案例；行级匹配仅在计数层面验证（r=0.894）；E1c 语料形状前提在行级功效不足；知识/噪声代理粗糙且部分循环；仅英文语料。

**结论[Conclusions]**　审计侧结论稳健：UltraX 的删除规则是位置性（非稀有感知）的，聚合收益是否及在多大程度上来自尾截、被切尾部是廉价噪声还是稀有知识，取决于进行中的 E2。方法学贡献（行级审计 + 机械零模型分离规则与语料形状）可复用于同类精炼系统的评估。

**关键词**　数据精炼；语料清洗；位置性尾截；长尾稀有度；预训练数据质量；切片评估
# Abstract

**Objective** To test whether the edit behavior of an industrial data-refinement pipeline (UltraX, arXiv:2607.08646) is systematically skewed against long-tail/rare content, and whether its aggregate gain claim (16B refined tokens beating 20B raw tokens) conceals slice-level costs.

**Methods** We audit UltraX's public (raw, refined, edit-operation) triples as the object of study across four English corpora (AICC / FineWeb / RedPajama-V2 / Ultra-FineWeb; 146k deduplicated docs): line-level rarity via wordfreq zipf with OOV stripped as a structural channel; deletion judged by fuzzy line matching cross-checked against operation logs; inference via document-clustered robust logistic models; a pure suffix-truncation mechanical null separates the pipeline's rule from the corpus's own shape; confirmatory endpoints and a revision log are pre-registered.

**Results** Rarity has no main effect on deletion within body prose (p=0.43); deletion is position-dominated; the rarity×position interaction is fully reproduced by mechanical suffix truncation (−0.966 vs −0.916), indicating the deletion rule is positional rather than rarity-aware. Word-level analysis (contact, exploratory) shows lost rare tokens are 74.7% knowledge-like vs 98.0% in the kept pool and 94.0% in the corpus base rate — relative deletion favors non-knowledge tokens, yet ≈2 knowledge-like rare tokens per document are still removed; the net direction is unmeasured.

**Limitations** Measurement-side audit only; no downstream-cost evidence yet (E2 outstanding); single refinement system as case study; matcher validated at count level (r=0.894); E1c corpus-shape premise underpowered at line level; knowledge/noise proxy coarse and partly circular; English corpora only.

**Conclusions** The audit-side result is robust: UltraX's deletion rule is positional, not rarity-aware; whether and how much of the aggregate gain comes from tail-cutting, and whether the cut tail is cheap noise or costly knowledge, depends on the ongoing downstream stage (E2). The methodology (line-level audit plus a mechanical null separating rule from corpus shape) is reusable for evaluating learned refinement systems.

**Keywords** data refinement; corpus cleaning; positional tail-truncation; long-tail rarity; pretraining data quality; sliced evaluation

# 1. Introduction

Data quality is now a first-class lever in LLM pretraining. UltraX, released by the MiniCPM/OpenBMB
team in 2026, refines web-scale corpora by training a **lightweight model** (released weights:
`openbmb/UltraX-0.6B-Preview`) to predict structured edit
operations (`keep` / `delete` / `replace` / `insert`), which a deterministic executor applies to
raw text. The claimed result is large and aggregate: less data, better models.

Aggregate claims about *better models* are silent about *better for whom*. A refinement model is
optimized for corpus-level statistics; rare content is, by definition, under-represented in its
objective. It is therefore a live empirical question — not a rhetorical one — whether refinement
systematically strips content about rare concepts, whether such stripping is a *cost* or a
*blessing* (deleting rare *noise* is cheap, deleting rare *knowledge* is not), and whether
aggregate evaluations would hide either outcome.

This paper contributes a **sliced audit** of UltraX as an object of study:

- We exploit UltraX's own release design: `openbmb/UltraX-Preview` ships
  `(raw, refined, edit operations)` triples — a ready-made, industrial-scale controlled
  experiment with full intervention traces, Apache-2.0.
- We measure *who* gets deleted along document position and rarity strata, with an explicit
  **mechanical null model** (suffix truncation) to separate what the pipeline's rule is from
  what the source corpus's structure does.
- We pre-registered the confirmatory endpoints *before* running the downstream stage
  (`configs/preregistration.md`, v1.1), and publish negative results (the naive rarity-deletion
  hypothesis is falsified; the rarity×position interaction is mechanical).

**Why this matters beyond UltraX.** Web corpora are known to contain boilerplate at document tails,
and any "clean smarter" pipeline is under pressure to cut length. If the *shape of the data*
(rarity grows toward document ends) makes positional cleaning expensive in rarity terms, then a
whole class of truncation-based cleaners could carry a hidden sliced cost that aggregate gains
conceal. We treat this class-level claim as a *hypothesis* for future work: the present paper's
contribution is the audit of one industrial system and a reusable methodology for separating a
cleaner's rule from the corpus shape it acts on.

---

# 2. Related Work

**Data refinement and cleaning.** UltraX (arXiv 2607.08646) and its tiered governance framework
(arXiv 2602.09003) sit in a long line of corpus-cleaning work (e.g., Ultra-FineWeb, RefineX).
These works validate with aggregate downstream numbers. To our knowledge, none performs a
slice-level audit of the *editor's behavior* on long-tail strata. arXiv full-text search
(`UltraX`; `perplexity AND (cleaning OR refinement) AND long-tail`) returns no follow-up
audit as of 2026-09-08.

**Sliced / stratified evaluation.** The dangers of aggregate metrics are well documented
(e.g., PeerBench-style benchmark audits). Our contribution is to apply slice-level discipline
to a *data-production pipeline* rather than to a benchmark.

**Rarity in LLMs.** PopQA (Mallen et al., ACL 2023) established popularity-stratified QA;
small-model knowledge benchmarks are floor-bound below ~300M parameters, which constrains
downstream probe design (we use language-modeling diagnostics rather than capability
benchmarks for this reason).

**Positional structure of web text.** Web documents concentrate navigation/boilerplate at tails;
we contribute a measurement of *rarity concentration at tails* as a corpus-level property
relevant to positional cleaners.

---

**Related work gap acknowledged.** A large adjacent literature studies how *quality filtering*
itself skews distributions (e.g., perplexity-based filters systematically retaining high-resource
varieties and excluding dialectal / low-resource content; classifier-based filters over-deleting
informal-but-factual text), plus work on data pruning and deduplication effects on rare sequences
(candidate refs: arXiv:2605.06901 on perplexity-filter bias; arXiv:2303.18223 survey on filtering
effects; Marion et al., "When Less is More"; Lee et al. on deduplication — to be verified and
cited precisely in the final version). Those studies measure the *bias of filters themselves*;
ours measures the *edit behavior of one specific learned refiner* and separates its rule from the
corpus shape. Positioning against this literature is completed in the next revision.

# 3. Object and Data

We audit `openbmb/UltraX-Preview` (Apache-2.0; ≈114M records). Fields: `uid`,
`raw_content`, `cleaned_content`, `processed_functions` (line-oriented, function-call format),
`source`. The dataset contains five source directories; we use four English corpora:
**AICC, FineWeb, RedPajama-V2, Ultra-FineWeb**.

UltraX's function space (paper §3.2/Table 1) is inherently *contiguous-range* editing:
`remove_lines(start_line, end_line)` deletes a consecutive line range, `replace_str(line, …)`
is localized string replacement, `add_line(…)` inserts, and `keep_all()` / `remove_all()` are
document-level. The refiner is a lightweight model; the released weights are
`openbmb/UltraX-0.6B-Preview` (the "0.6B" figure is from the model card, not the paper body).
**Because deletion is range-based, positional structure is partly present by design — our audit
measures what falls inside those ranges (rarity strata), not whether deletion is contiguous.**
The paper's data-efficiency headline is verified from the full text (§4.2): a ≈1B MiniCPM model
trained on 16B UltraX-refined tokens scores 45.49, above Raw / ProX-C trained on 20B tokens
(45.08 / 45.05). The paper itself argues (§4.4) that its advantage "does not stem from simply
deleting more content" — our slice-level audit interrogates precisely that claim.

Edit-operation taxonomy on our sample [DONE]: `remove_lines(a,b)` dominates;
`replace_str` is mostly global formatting (newline removal), separated from content deletions
throughout. Median character retention 0.848 and line count 31 → 10 come from a 300-doc
diagnostic (provenance note; not re-derived in committed artifacts).

Samples: main audit n=4,800 docs / 101,250 sampled lines (regression analysis sample n=101,046
after dropping rows with missing covariates); pilot pipeline n=146,185 deduplicated documents.
Note: Ultra-FineWeb is itself an already-refined version of FineWeb; auditing it means applying
UltraX a second time (second-order refinement), which we flag because its weaker signal
(§5.2) may partly reflect a cleaner input distribution rather than corpus quality.

---

# 4. Method

## 4.1 What counts as "deleted"

We do **not** reconstruct deletions from operation logs (the semantics of `replace_str`
over line numbering are underdetermined). Instead, a raw line is *kept* if it fuzzily appears
in the refined text: exact-substring fast path, else `rapidfuzz.partial_ratio` against the
deduplicated set of refined lines, threshold 90 (sensitivity 85/95 reported). Line-to-line
matching (not whole-document) avoids a length-truncation bias on long documents.

**Matcher validity — count-level cross-check [DONE]**:
judged-deleted line count vs. operation-log implied count (`remove_lines` span sizes) correlates
at **Pearson r = 0.894** across documents. Two honest caveats: (i) counts agree at the aggregate
level but judged deletion is systematically **lower** than op-implied by ≈24%
(Σ judged 11,192 vs Σ ops 14,808), consistent with the matcher missing merged/rewritten lines;
(ii) 62.5% of documents contain `replace_str` (line-merging/rewriting), the regime where fuzzy
matching is least reliable, while only 37.5% are pure-`remove_lines` documents. A line-level
human audit (protocol §5, T3) is planned; until then the r=0.894 is reported as a count-level
cross-check, not line-level precision.

## 4.2 Rarity

Token rarity = wordfreq zipf frequency (en). Out-of-vocabulary (OOV) tokens (URLs, code,
non-English) are **separated from the rarity channel**: OOV-heavy lines form a structural
channel (deleted 6.1% OOV vs 2.4% kept). Lines are classified head / tail-knowledge /
tail-noise by position and vocabulary content using a **frozen** classifier
(thresholds fixed a priori, not tuned on evaluation data).

## 4.3 Estimands and models

Document-level stratification (raw document as the atomic unit); Calibration/Train/Eval
splits; evaluation anchors are raw-form text shared by both arms.

Logistic models, clustered standard errors by document:

- **M1** (body prose only): `kept ~ rarity + log_len + pos + oov`
- **M2** (all): adds `is_heading` and `rarity × is_heading`
- **M3** (all): `kept ~ rarity × pos + log_len + is_heading + oov`

## 4.4 The mechanical null (T2)

A suffix-truncation rule — "delete the last k lines where k = the document's observed deletion
count" — is fitted under the *same* regression. If the pipeline is position-truncating, M3
coefficients under the simulated labels will match the observed ones; the *excess* of observed
over simulated is the genuine signal.

---

# 5. Results [DONE unless marked]

## 5.1 Deletion is position-dominated, not rarity-aware

| model | pos coef (OR) | rarity coef | rarity×pos | note |
|---|---|---|---|---|
| M1 (prose) | **−1.48 (OR 0.23, p<1e-10)** | +0.096 (**p=0.43, null**) | — | naive rarity-deletion **falsified** |
| M3 (all) | +2.99* | +0.60 | **−0.916 (p<1e-10)** | *conditional at rarity=0; see text |
| suffix-truncation sim | −2.05 | +0.68 | **−0.966** | reproduces the *interaction* |
| random-deletion sim | −0.14 | +0.02 | +0.04 | null control behaves |

\* M3 includes the rarity×pos interaction; its pos coefficient (+2.99) is the conditional effect
at rarity=0. At representative rarities the position marginal effect is negative: at rarity≈4.3
(average), pos effect ≈ 2.99 + 4.3×(−0.916) ≈ −0.95, consistent with M1's negative pos effect.
The sign difference between M1 and M3 is therefore a consequence of the interaction, not a
contradiction.

**Reading.** Within body prose, rarity *per se* does not predict deletion (M1 rarity p=0.43).
Deletion tracks document position. The rarity×position interaction — superficially
"tail rare content is cut" — is **reproduced by mechanical suffix truncation** acting on the
source corpus's own rarity~position structure, whereas the simulated and observed *position
main effects differ in sign* (+2.99 vs −2.05), indicating the pipeline is not a pure suffix
truncator (its deletion is spread across the document with a milder positional gradient).
**UltraX's deletion rule is positional, not rarity-aware.** Positional tail-cutting of boilerplate
is standard practice in web-corpus cleaning (e.g., length and boilerplate heuristics in C4/Gopher
style pipelines); the contribution here is not the existence of such behavior but the *methodology*
that separates the cleaner's rule (positional) from the corpus shape it acts on (rarity-at-tail),
and the quantification of what the cut tail contains. Whether any aggregate gain of refinement is
attributable to tail-cutting, and whether the cut tail is cheap noise or costly knowledge, are
**open questions** addressed by the downstream stage (§6), not established facts.

## 5.2 Cross-corpus consistency

Prose-gap (mean zipf of deleted − kept lines, within-doc), per corpus:

| corpus | prose gap | share docs negative |
|---|---|---|
| AICC | −0.46 | 0.69 |
| RedPajama-V2 | −0.48 | 0.77 |
| FineWeb | −0.15 | 0.56 |
| Ultra-FineWeb | −0.12 | 0.52 |

Direction consistent everywhere; magnitude splits into two tiers (rawer corpora stronger).
Threshold sensitivity (85/90/95) pooled gap: −0.354/−0.359/−0.372 — stable.

## 5.3 Structure channels

- **Headings are exempt** (is_heading OR≈100; deletion 5.5% vs 11.3% body).
- **OOV is a separate structural channel** (deleted 6.1% vs kept 2.4%; OR 0.07).
- Deletion by length bucket (v3, 4 corpora): short lines (1–5 words) deletion targets *common*
  junk (gap +0.13); mid-length content lines (6–15 words, gap −0.31; 16–40 words, gap −0.28;
  40+ words, gap −0.19) all show rare skew; bucket-token-weighted primary gap −0.21.
  (An earlier single-corpus run showed larger skews, superseded by the cross-corpus numbers.)

> Figure 1 (repo `figures/fig2_stageA_v3.png`): left panel = per-bucket zipf gap (deleted − kept)
> with bootstrap CIs; right panel = deletion rate vs. rarity decile. Both panels illustrate §5.1–§5.3.

## 5.4 Word-level loss: knowledge or noise? [contact, exploratory]

Multiset token difference over 2,000 documents: of rare tokens lost by refinement, 74.7% are
knowledge-like and 25.3% are noise-like under our classifier; of rare tokens *kept*, 98.0% are
knowledge-like; the merged corpus base rate is 94.0% ((4070+26039)/(5448+26570)).
**Caveat: knowledge/noise here is dictionary-membership-based (in-dictionary rare, proper-noun,
or long lower-case OOV) — a coarse, partially circular proxy**; the classifier is inconsistent
with the Stage-A OOV treatment in places and will be replaced by human-labeled subsamples in the
refined word-loss analysis. Descriptive only (no significance test; numeric-only noise tokens
missed). Under this proxy: lost rare tokens are less knowledge-rich than the kept pool and the
corpus base rate — i.e., in relative terms deletion enriches the surviving rare pool for
knowledge and concentrates deletion on the noisier rare tokens; in absolute terms it still
removes ≈2 knowledge-like rare tokens per document. **The two facts pull in opposite directions
for any "cost" narrative; net downstream direction is left to Stage B (§6).**

## 5.5 Corpus-shape premise (E1c) [contact, open]

Whether rare vocabulary actually concentrates toward document ends in *raw* corpora — the
premise that makes positional deletion expensive in rarity terms — is **inconclusive and
underpowered** at the line level: within-doc rare-line position gap −0.012, p=0.38 on a coarse
proxy with only n=114 qualifying documents — computed on the *exploratory* Stage-A sample,
not on the independent confirmatory split required by the pre-registration. This null neither
confirms nor refutes the premise. The confirmatory token-level measurement (knowledge-token
restricted, within-document permutation null, fresh split) is outstanding; **if E1c fails at
that level, the rarity-cost narrative collapses to pure
length normalization** and is reported as such (pre-registered falsification rule).

---

# 6. Ongoing Work (Stage B) — the load-bearing experiment

Pre-registered endpoint (v1.1): **E2** — do probes trained on refined vs raw corpora show a
*sliced* downstream difference on tail-knowledge strata (Δ_tailK = PPL(refined) − PPL(raw) on
the same raw-anchored eval lines > 0 means refined worse; rare-word recovery MRR is the
co-primary, with Δ_MRR < 0 meaning refined worse — sign convention to be unified in the frozen
δ_min entry), or is refinement genuinely no worse on the tail (informative null via TOST)?
Protocol: `docs/STAGEB_PROTOCOL.md`. Probes are 50–300M from-scratch transformers (from-scratch,
because continued pretraining on an already-trained model anchors the corpus difference away and
has a forgetting bias against low-exposure tail content, Chang et al., NeurIPS 2024). PopQA
long-tail serves as an external reference expected to be floor-bound at this probe scale.

**Implementation status (honest).** Data prep (146k deduped docs, splits, arms) is done.
The trainer runs (CPU smoke passed) and the evaluator is written, but the *evaluator has not
been executed end-to-end on trained checkpoints yet*; an adversarial code review during
preparation of this revision found and fixed two defects (non-causal attention in the probe;
context-length overflow in per-line scoring) and identified pending items: a per-arm metadata
file with epoch/token coverage (added), the tail-restored counterfactual arm (not yet built),
and the pilot-gate aggregation script (03_gate, not yet written). E2 results therefore are not
yet available; nothing in this preprint's Stage-A conclusions depends on them.

---

# 7. Limitations

1. **Measurement-only audit; no downstream-consequence evidence yet.** This preprint establishes
   *what* refinement deletes and *that* its rule is positional; it does not yet establish whether
   the deletion has a measurable downstream cost. That evidence is the load-bearing experiment
   (Stage B / E2) and is outstanding — the paper should be read as a measurement-first audit,
   not as a demonstration of harm.
2. Single refinement system as case study (UltraX); class-level claims are hypotheses,
   not established.
3. Line-fuzzy matching is validated at the count level (r=0.894); line-level misclassification
   (merged/rewritten lines) is not fully audited; a human-labeled line audit is planned.
4. E1c corpus-shape premise is inconclusive at line level (underpowered); token-level
   measurement is outstanding; the paper's "rarity cost" reading is contingent on it and on E2.
5. Word-level knowledge-vs-noise proxy is coarse and descriptive (no significance test);
   numeric-only noise tokens are missed; results are contact-stage.
6. Conclusions are limited to ≤2B-scale language-modeling diagnostics; no capability
   benchmark is used as primary endpoint (floor effects at probe scale, verified).
7. English corpora only. UltraX's aggregate claim is asserted on corpora that include Chinese
   data (the system is by the OpenBMB/MiniCPM lab); the publicly released intervention traces
   used here are the English shards, which is the reason for the English-only scope.

# 8. Reproducibility and Open Source

All code, data hashes, and outputs: **https://github.com/ausyeah/artical** (MIT license).

| Asset | Where |
|---|---|
| Audit scripts (line matching, rarity, strata, regression, mechanical null) | `src/01_slice_audit.py`–`src/04_mechanical_null.py` |
| Stage-B probe pipeline (data prep → from-scratch trainer → sliced eval) | `src/stageb/00…02…` + `docs/RUN_GUIDE.md` |
| Pre-registration with timestamped revision log | `configs/preregistration.md` |
| Stage-B protocol (slice construction, leakage rules, pilot gates) | `docs/STAGEB_PROTOCOL.md` |
| Result artifacts (every number in this draft) | `reports/step2_*`, `reports/step3_*` |
| Figures | `figures/fig2_stageA_v3.png` |

Audited dataset: `openbmb/UltraX-Preview` (Apache-2.0). We download only sampled shards
and stream the rest (`datasets streaming`); full-data access is documented in `SETUP.md`.
Single RTX 4060 8GB suffices; the audit side is CPU-only. Pilot probe data (AICC, 146k
deduplicated docs, raw ≈88M / refined ≈66M tokens, equal-doc arms) is prepared under
`runs/pilot/`; the GPU probe-training script passed a CPU smoke test (`docs/RUN_GUIDE.md`).

# References

- UltraX. arXiv:2607.08646.
- Tiered data management for AGI. arXiv:2602.09003.
- Mallen et al. PopQA (ACL 2023). arXiv:2212.10511.
- Chang et al. (NeurIPS 2024). arXiv:2406.11813.
- Gururangan et al. DAPT (ACL 2020). arXiv:2004.10964.
- wordfreq: multiscript word frequency data. R. Speer.