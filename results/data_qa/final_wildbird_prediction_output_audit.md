# Final Wild-Bird Model Prediction Output Audit Report

**Date**: 2026-08-03  
**Audit Purpose**: Verify dataset integrity, test temporal prediction variation, reproduce baseline metrics, resolve the LR PR-AUC discrepancy (`0.0730` vs `0.0670`), and check cross-model alignment.

---

## 1. Schema and Key Integrity

| Dataset Name | File Path | Total Rows | Unique CSDs | Months | Positives | Prevalence | Key Check |
|---|---|---|---|---|---|---|---|
| Corrected Wild-Bird GraphSAGE | `wildbird_graphsage_knn3_corrected_test_predictions.csv` | 55,594 | 5,054 | 1-11 | 79 | 0.001421 | PASS |
| Wild-Bird Logistic Regression | `wildbird_lr_strict_test_predictions.csv` | 55,594 | 5,054 | 1-11 | 79 | 0.001421 | PASS |
| Strict Env-Lag LR Baseline | `environment_with_lags_v1_strict_test_predictions.csv` | 55,594 | 5,054 | 1-11 | 79 | 0.001421 | PASS |

- Key `(csd_uid, year, month)` has **zero duplicates** across all prediction files.
- All rows strictly belong to `year == 2025`.

---

## 2. Monthly Prediction Variation Analysis

| Model | Total Unique Scores | Score Std Dev (Median) | Score Std Dev (Mean) | Constant CSDs Count | Constant CSDs % |
|---|---|---|---|---|---|
| Corrected Wild-Bird GraphSAGE | 5,054 | 0.000000 | 0.000000 | 5,054 | 100.0% |
| Wild-Bird Logistic Regression | 55,594 | 0.056293 | 0.066566 | 0 | 0.0% |

### Critical Finding: Temporal Collapse in GraphSAGE
> [!CAUTION]
> **GraphSAGE predictions are constant across all months for every CSD.**  
> Out of 5,054 CSDs, **5,054 (100.0%)** have exactly 1 unique predicted probability across all 11 test months in 2025.  
> 
> **Cause**: In `scripts/train_wildbird_graphsage_knn3_corrected.py` (lines 183-189), the graph node feature matrix `X_all` of shape `(5054, 62)` is constructed by iterating through all monthly rows in the dataset and overwriting `X_all[idx]`. As a result, `X_all[idx]` retains only the final month's feature vector. Passing `X_all` into `model(x_tensor, edge_index)` produces a single static embedding per CSD node, which is broadcast across all monthly test rows. Monthly variations in climate, Dynamic World land cover, outbreak history, and wild-bird HPAI pressure are completely neglected.

---

## 3. Metric Reproduction

| Model | Source File | Stored PR-AUC | Reproduced PR-AUC | Stored ROC-AUC | Reproduced ROC-AUC | Exact Match? |
|---|---|---|---|---|---|---|
| Corrected GraphSAGE Ensemble | `wildbird_graphsage_knn3_corrected_multiseed_metrics.csv` | 0.099950 | 0.099950 | 0.959476 | 0.959476 | **YES** |
| Wild-Bird Logistic Regression | `wildbird_lr_strict_metrics.csv` | 0.073042 | 0.066970 | 0.913606 | 0.916709 | **NO** (See Section 4) |
| Strict Env-Lag LR Baseline | `environment_with_lags_v1_strict_test_predictions.csv` | N/A | 0.065369 | N/A | 0.915754 | N/A |

---

## 4. Resolution of Wild-Bird LR Discrepancy (0.0730 vs 0.0670)

The discrepancy between the two previously reported Wild-Bird LR PR-AUC values (`0.0730` vs `0.0670`) has been fully resolved:

1. **`0.0730` PR-AUC** (`train_wildbird_models_strict.py` -> `wildbird_lr_strict_metrics.csv`):
   - Trained on **ALL 2022-2024 data** (including 2024 Q4).
   - Scaler fit on 2022-2024.
   - Unweighted default `LogisticRegression(max_iter=1000)`.

2. **`0.0670` PR-AUC** (`generate_wildbird_lr_predictions.py` -> `wildbird_lr_strict_test_predictions.csv`):
   - Trained on **2022-01 through 2024-09 ONLY** (excluding 2024 Q4, reserving it for validation).
   - Scaler fit on 2022-01 through 2024-09.
   - Class-weighted `LogisticRegression(class_weight={0:1.0, 1:pos_weight}, solver='liblinear')`.

### Canonical Model Decision
`results/baselines/wildbird_lr_strict_test_predictions.csv` (**PR-AUC = 0.0670**) is selected as the **canonical prediction file** for future comparisons because it respects the leak-free 2022-01..2024-09 training split and uses explicit positive class weighting.

---

## 5. Cross-Model Alignment

- **Row Count**: 55,594 CSD-month test rows matched perfectly across LR and GraphSAGE prediction files.
- **Target Alignment**: **0 target mismatches** out of 55,594 rows.
- **CSD Population**: 5,054 unique CSDs across 11 months (2025).

---

## Audit Decision

> [!CAUTION]
> ### DECISION: **INVALID OR UNCERTAIN: do not use current model comparisons for manuscript claims until the listed issue is fixed.**
> 
> **Reason**: While schema, keys, and cross-model target alignment are valid, the corrected GraphSAGE model suffers from complete temporal collapse (100% of CSDs have static, constant predictions across all 11 test months due to node feature overwriting during graph construction). Until GraphSAGE is updated to handle dynamic spatio-temporal node features per month, the `PR-AUC = 0.0999` result cannot be claimed as a valid spatio-temporal GNN baseline for manuscript conclusions.

