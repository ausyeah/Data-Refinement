# Research Plan — Sliced Auditing of Data Refinement Pipelines

> Version 1.0 · 2026-09-08 · Status: pre-experiment
> Companion repo for the preprint *"SliceAudit"* (working title).

---

## 1. Motivation

Industrial LLM data refinement is now a first-class lever. **UltraX**
(arXiv:2607.08646; OpenBMB) refines web-scale corpora by having a **0.6B** model predict
structured edit operations (`keep / delete / replace / insert`), applied by a deterministic
executor. Reported result: a 1B model trained on **16B refined tokens** outperforms one
trained on **20B raw tokens**.

That is an **aggregate** claim. It does not answer:

> For *whom* does refinement help? Does it systematically strip content about
> **rare concepts** — precisely the knowledge that is scarce by definition?

This suspicion is grounded: a refinement model is optimised for aggregate quality, and
rare/off-distribution content is statistically the most likely to look "low quality" to it.
If tail content is disproportionately deleted, aggregate gains mask a **slice-level loss**.

This mirrors a pattern the author has already documented once in a different domain:
in a controlled 36-cell study of class-imbalance handling (UNSW-NB15), a post-hoc
threshold calibration that was **globally best** exhibited a **transfer-failure case**
(test Macro-F1 0.672 → 0.452 when the calibration objective changed), and SMOTE
interpolation artifacts caused a **systematic −0.05 recall drop on one minority class**
(3/3 seeds). Aggregate optimisation hiding slice-level damage is the same failure,
in a new organ.

## 2. Object under audit

`openbmb/UltraX-Preview` (HF; Apache-2.0; ≈114M records / ≈100B tokens) ships
**(raw text, refined text, edit operations)** triples with full intervention traces.
This is a ready-made controlled experiment: the intervention already happened,
industrially, and was logged. **We therefore do not inject artifacts ourselves in Stage A**
— we audit real ones. (If self-injection is later required, intensities must first be
calibrated to measured natural occurrence rates; see §5.3.)

## 3. Stages

### Stage A — zero-training sliced audit (go/no-go gate)

* Stratify the corpus by a **rarity index** (three proxies, cross-validated:
  corpus frequency of entities/n-grams, small-model pass rate, question/text NLL).
* Count edit-operation types per rarity stratum.
* **Primary test**: `deletion rate × rarity stratum` interaction, pre-registered,
  bootstrap 95% CI.
* **Cost**: no GPU. This stage alone yields the go/no-go decision.

### Stage B — lightweight training verification

* Probes of **150M–500M** (8-bit optimiser, gradient checkpointing, ~100 tokens/param),
  3–5 seeds, trained on raw vs. refined subsets.
* Dependent variable: **sliced downstream delta** (per rarity stratum), never an average.

### Stage C — calibration of cheap metrics

* Predictors: edit-operation density, perplexity, classifier quality scores, diversity metrics.
* Gold label: Stage-B sliced gains.
* Deliverable: **calibration curves + failure boundary** — at which slice does a cheap
  metric stop predicting reality?
* Prior work does corpus-level ranking only (DataDecide, ICML'25); slice-level
  calibration decay is unoccupied.

## 4. Pre-registration summary

**Primary endpoint (intersection–union test).** Reject H0 only if
(i) interaction term CI excludes 0 **and** (ii) tail-slice loss significant vs. nominal α.
IUT controls family-wise error without further correction.

**Kill criterion.** If refinement is also *better* on the tail, the main hypothesis is dead.
We then run a **TOST equivalence test** (95% CI ⊂ ±δ_min) and publish an informative null.

**Exploratory family** (BH-FDR, q=0.05): per-stratum AUROC/ECE, ablations, indicator
comparisons, prompt-wording sensitivity. Threshold sweeps are **descriptive only** —
significance is judged solely on the pre-registered operating-point grid.

## 5. Validity safeguards

### 5.1 Difficulty ≠ rarity (circularity is the top threat)

Pass rate and NLL are *both* rarity proxies *and* difficulty measures. Using them as the
difficulty axis makes the interaction term circular. Difficulty therefore uses
**independent** proxies (reasoning steps, length, human annotation), and we report a
**3×3 grid** (rarity × independent difficulty) with dedicated *"rare but easy"* and
*"common but hard"* subsets. If the interaction vanishes after controlling for difficulty,
the main hypothesis is falsified.

### 5.2 Negative controls

1. Interventions applied to **head** slices (interaction should ≈ 0);
2. **Unstructured noise** at matched intensity (should differ from structured edits);
3. **500 label permutations** → null distribution of the interaction term;
4. 2×2: {train-side / test-side} × {raw / refined}.

### 5.3 Artifact realism

If self-injection is used, injection intensities must be **calibrated to measured natural
occurrence rates** in real generated data, otherwise results describe a dose nobody produces.

### 5.4 Training noise

QLoRA run-to-run variance can exceed sample variance. **≥3 (target 5) seeds** with explicit
variance decomposition; sample-level bootstrap alone systematically underestimates CI.

### 5.5 Power

Pilot of 50–100 samples per cell → estimate per-class rates and intra-cell ICC →
parametric bootstrap over n ∈ {50,100,200,400} → choose smallest n with power ≥ 0.8.
Budget prioritises tail cells (higher variance, higher marginal information).

### 5.6 External validity

Conclusions are stated conditionally on **≤2B models, QLoRA, long-tail classification**.
API-scale models are used for **sign consistency only**, never magnitude extrapolation;
no scaling laws are fitted. Unexplored dimensions (>7B, full fine-tuning, generative tasks)
are listed explicitly.

## 6. Related work positioning

| Work | What it does | What this project adds |
|---|---|---|
| UltraX (2607.08646) | Refinement pipeline; **aggregate** gains only | **Slice-level audit** of the same pipeline |
| Tiered Data Management (2602.09003) | L0–L4 governance framework | Empirical audit of refinement behaviour on tails |
| DataDecide (ICML'25, 2504.11393) | Corpus-level ranking, ~80% transfer | Slice-level metric calibration & failure boundary |
| DCScore (ICML'25, 2502.08512) | Diversity metric | Validated against *sliced* gains, with failure cases |
| Beyond Model Collapse (ICLR'25, 2406.07515) | Verifier filtering | Failure of verifier thresholds on tail slices |
| ToEdit (ICML'25, 2412.14689) | Token-level edits w/ theory | Empirical tail audit of an industrial editor |
| Outlier Gradient Analysis (ICML'25 Oral, 2405.03869) | Hessian-free harmful-sample detection | Cheap-indicator calibration on slices |

Retrieval checks (2026-09-08): arXiv full-text `long-tail AND synthetic data AND minority`
returns one CV paper only; no LLM/data-governance work performs a controlled
artifact-sensitivity audit on long-tail slices.

## 7. Constraints

Single RTX 4060 Laptop 8GB · API budget ≈ ¥100/month · no cluster access.
All experiments are designed to run on this hardware; this constraint is treated as a
reproducibility feature, not a limitation to hide.

## 8. Output plan

| Track | Output | Timeline |
|---|---|---|
| Primary | arXiv preprint + open-source release | 2–4 months |
| Long | A-conference submission (NeurIPS/ICLR **D&B**, ACL, COLM) | 8–18 months |
| Fallback | Invention patent (filing number) | 3–6 months |

> **Ordering note**: the patent application is filed **before** the arXiv preprint goes
> public (public disclosure before filing destroys novelty).
