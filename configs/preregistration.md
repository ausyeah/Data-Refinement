# Pre-registration — SliceAudit

> **Must be frozen (git commit hash) before any test-set result is observed.**
> After freezing, every change requires an entry in §9 with justification.
> Status: 🟡 DRAFT — to be frozen after Stage-A schema confirmation.

---

## 1. Primary endpoint (confirmatory, single)

**Intersection–union test (IUT).**

```
H0: { edit-op × rarity interaction = 0 }  OR  { tail-slice loss ≤ α }
```

Reject only if **both** components are individually rejected at α = 0.05:
- Component 1: interaction term of the pre-registered model, bootstrap 95% CI excludes 0
- Component 2: tail-slice loss significant (binomial test, p < 0.05, point estimate > α)

IUT controls FWER ≤ α; **no additional multiplicity correction applies** to this family.

**Interpretation is pre-bound:**
- Both hold → mechanism supported, paper proceeds as an audit.
- Not both → the work is a **negative-result / replication study**; no new mechanism is claimed.
- Refinement better on tail → **kill criterion** triggered; run TOST and publish informative null.

## 2. Exploratory family

All of the following are **exploratory**, BH-FDR q = 0.05, marked `*` in tables:
per-stratum AUROC / ECE / coverage; ablations; indicator comparisons;
prompt-wording sensitivity; bin-count sensitivity.

**Threshold / sweep curves are descriptive only.** No significance stars on curves;
only the pre-registered operating-point grid enters statistical testing.

## 3. Pre-registered operating-point grid

| Parameter | Grid |
|---|---|
| training tokens | {0.5B, 1B, 2B} |
| synthetic/refined share | {0, 25, 50, 75, 100}% |
| probe size | {150M, 300M, 500M} |
| bins | {3, 5} |
| α | {0.05, 0.10, 0.15} |

## 4. Rarity stratification (frozen definition)

| Proxy | Definition | Role |
|---|---|---|
| `r_freq` | corpus frequency of subject entities / n-grams | **primary axis** |
| `r_pass` | small-model empirical pass rate, K=8, T=0.6 | covariate only |
| `r_nll` | length-normalised NLL of the text | secondary proxy |

- Bins: quantiles computed **on the calibration split only** (head / torso / tail).
- Mandatory cross-checks: Spearman ρ(r_freq, r_nll), ρ(r_freq, r_pass), and the
  **r_freq × difficulty 3×3 grid** (see §5).
- Independent control: an externally defined tail subset (not derived from calibration bins).

## 5. Difficulty axis (circularity control — binding)

**Forbidden as difficulty proxies:** pass rate (`r_pass`) and NLL (`r_nll`) — they are
already rarity proxies; using them makes the interaction term circular.

**Allowed:** reasoning steps, text length, human-annotated difficulty.

Required analyses:
- 3×3 grid (rarity × independent difficulty), with dedicated
  *"rare but easy"* and *"common but hard"* subsets reported separately;
- regression includes difficulty main effect **and** difficulty × edit interaction;
- **falsification rule**: if the interaction vanishes after difficulty control,
  the main hypothesis is rejected and reported as such.

## 6. Splits, leakage and access discipline

- Splits by content hash: **calibration 30% / test 70%**; frozen in `configs/splits.json`.
- All thresholds, bin quantiles and indicator calibrations are fitted **on the calibration
  split only**.
- The test set is evaluated **once**, by a single script that emits aggregate results only.
- Intermediate test-set tables are never written to disk.

## 7. Seeds and variance

- Primary experiments: **3 seeds minimum, 5 targeted**, with explicit variance decomposition
  (seed variance vs. sample variance).
- Bootstrap CIs must incorporate training noise; sample-level bootstrap alone is insufficient.

## 8. Reproducibility of external components

- Dataset/dataset-version, model revision, temperature=0, sampling seed, prompt hash
  recorded for every call.
- Where a closed model is used: core subset re-sampled at a ≥4-week interval;
  ≥2 generators cross-checked, one of them a locally hosted open-weight model;
  main conclusions must hold across generators.

## 9. Revision log

| Date | Change | Reason |
|---|---|---|
| *(empty until freeze)* | | |
