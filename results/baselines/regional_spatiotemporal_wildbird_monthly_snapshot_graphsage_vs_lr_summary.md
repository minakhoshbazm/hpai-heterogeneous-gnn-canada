# Regional Spatial-Temporal Validation: Monthly-Snapshot GraphSAGE vs LR

**Validation Purpose**: Test geographic generalization to unseen Canadian regions in 2025 using models trained strictly outside each held-out region (2022–2024).  
**Models**:
1. Final Retrained Wild-Bird Logistic Regression Baseline
2. Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE (3-Seed Ensemble)

---

## 1. Regional Results Table (2025 Test Set)

| Regional Block | Held-Out Provinces | Test Rows | Positives | Prevalence | Reliability | LR PR-AUC | LR Lift | GraphSAGE PR-AUC | GraphSAGE Lift | GraphSAGE Win Rate | Bootstrap Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Atlantic** | 10, 11, 12, 13 | 9,141 | 5 | 0.000547 | **UNSTABLE (N_pos < 10)** | 0.0165 | 30.08x | **0.0290** | **52.99x** | 70.9% | **UNSTABLE (N_POS < 10)** |
| **Quebec_Ontario** | 24, 35 | 20,449 | 16 | 0.000782 | **STABLE** | 0.0059 | 7.60x | **0.0056** | **7.20x** | 45.9% | **COMPARABLE, NOT PROVEN BETTER** |
| **Prairies** | 46, 47, 48 | 17,743 | 41 | 0.002311 | **STABLE** | 0.0217 | 9.39x | **0.0303** | **13.13x** | 88.9% | **COMPARABLE, NOT PROVEN BETTER** |
| **British_Columbia** | 59 | 8,261 | 17 | 0.002058 | **STABLE** | 0.1261 | 61.26x | **0.2176** | **105.74x** | 91.6% | **COMPARABLE, NOT PROVEN BETTER** |

> [!CAUTION]
> **RELIABILITY WARNING FOR SPARSE REGIONS**:  
> Blocks with fewer than 10 positive test outbreaks (e.g. Atlantic, $N_pos = 5$) yield numerically unstable PR-AUC estimates and bootstrap distributions. Superiority claims must not be based on sparse regional blocks.

---

## 2. Pooled Spatial-Transfer Evaluation (55,594 Out-of-Region Rows)

| Model | Pooled Spatial PR-AUC | Pooled Spatial ROC-AUC | Precision @ 50 | Recall @ 50 | Precision @ 100 | Recall @ 100 |
|---|---|---|---|---|---|---|
| Wild-Bird Logistic Regression | 0.0157 | 0.8446 | 0.0400 | 0.0253 | 0.0200 | 0.0253 |
| **Wild-Bird Monthly-Snapshot GraphSAGE (Ensemble)** | **0.0439** | **0.8935** | **0.1200** | **0.0759** | **0.0700** | **0.0886** |

- **Observed Pooled Spatial PR-AUC Difference**: +0.0282
- **Bootstrap Mean Difference**: +0.0276
- **95% Percentile Confidence Interval**: [-0.0003, +0.0821]
- **Statistical Decision**: **COMPARABLE, NOT PROVEN BETTER**

---

## 3. Distinction Between Temporal vs Geographic Generalization

- **Temporal Generalization (Previous Rolling-Origin Experiment)**: Evaluated models on future unseen years across all regions combined (GraphSAGE Pooled PR-AUC = `0.0628` vs LR = `0.0540`, CI = `[+0.0006, +0.0176]`, Statistically Better).
- **Geographic Generalization (This Regional Experiment)**: Evaluated models on 2025 data inside regions completely unseen during training. GraphSAGE achieved a pooled spatial-transfer PR-AUC of `0.0439` vs LR of `0.0157`.

---

## 4. Methodological Assurance

> [!IMPORTANT]
> All regional GraphSAGE evaluations used the **valid Monthly-Snapshot GraphSAGE architecture** with strict regional edge isolation (0 cross-region edges). Zero superseded static CSD-level GNN outputs were used.

