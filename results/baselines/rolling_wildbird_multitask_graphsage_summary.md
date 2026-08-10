# Multi-Task GraphSAGE: Wild-Bird Auxiliary Supervision Experiment

**Purpose**: Test whether a shared GraphSAGE encoder simultaneously trained on poultry HPAI prediction
and wild-bird HPAI presence (auxiliary) improves primary poultry-outbreak prediction.

---

## 1. Lambda Selection per Fold (Primary Poultry Validation PR-AUC)

| Fold | Test Year | Selected Lambda | Multitask Val PR-AUC | Single-Task Val PR-AUC (lambda=0) | Improvement | Selected Epochs |
|---|---|---|---|---|---|---|
| 1 | 2023 | `lambda=0.00` | 0.2058 | 0.2058 | +0.0000 | [17, 27, 14] |
| 2 | 2024 | `lambda=0.10` | 0.1454 | 0.1411 | +0.0043 | [21, 36, 1] |
| 3 | 2025 | `lambda=0.05` | 0.2639 | 0.2536 | +0.0103 | [26, 1, 1] |

> [!NOTE]
> Lambda selection used **only the primary poultry validation PR-AUC**. No test-year labels were used.

---

## 2. Per-Fold Test Results (Ensemble)

| Fold | Test Year | Test Rows | Pos | Prev | Selected Lambda | PR-AUC | Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2023 | 60,648 | 72 | 0.001187 | `lambda=0.00` | 0.0630 | 53.05x | 0.8676 | 0.1000 | 0.0694 | 0.0900 | 0.1250 |
| 2 | 2024 | 60,648 | 40 | 0.000660 | `lambda=0.10` | 0.1135 | 172.04x | 0.9251 | 0.1200 | 0.1500 | 0.0800 | 0.2000 |
| 3 | 2025 | 55,594 | 79 | 0.001421 | `lambda=0.05` | 0.0810 | 56.97x | 0.9220 | 0.1800 | 0.1139 | 0.1200 | 0.1519 |

---

## 3. Pooled Comparison (176,890 rows, 2023-2025)

| Model | Pooled PR-AUC | Lift | Pooled ROC-AUC | P@50 | R@50 | P@100 | R@100 |
|---|---|---|---|---|---|---|---|
| **Multitask GraphSAGE (Ensemble)** | **0.0599** | 55.44x | **0.8827** | 0.2200 | 0.0576 | 0.1400 | 0.0733 |
| Original wBCE GraphSAGE (Ensemble) | 0.0628 | 58.16x | 0.8971 | 0.2400 | 0.0628 | 0.1600 | 0.0838 |
| Logistic Regression | 0.0540 | 49.99x | 0.8967 | 0.2600 | 0.0681 | 0.1500 | 0.0785 |
| Random Forest | 0.0401 | 37.14x | 0.9274 | 0.1600 | 0.0419 | 0.1200 | 0.0628 |
| XGBoost | 0.0540 | 49.97x | 0.8661 | 0.2000 | 0.0524 | 0.1100 | 0.0576 |

---

## 4. Bootstrap Decisions (Multitask GraphSAGE vs Comparators)

| Comparison | Multitask PR-AUC | Reference PR-AUC | Observed Diff | Bootstrap Mean | 95% CI | Win Rate | Decision |
|---|---|---|---|---|---|---|---|
| Multitask GraphSAGE vs original wBCE GraphSAGE | 0.0599 | 0.0628 | -0.0029 | -0.0022 | [-0.0101, +0.0064] | 26.5% | **COMPARABLE, NOT PROVEN BETTER** |
| Multitask GraphSAGE vs Logistic Regression | 0.0599 | 0.0540 | +0.0059 | +0.0062 | [+0.0005, +0.0134] | 98.4% | **STATISTICALLY BETTER** |
| Multitask GraphSAGE vs Random Forest | 0.0599 | 0.0401 | +0.0198 | +0.0200 | [-0.0053, +0.0539] | 89.0% | **COMPARABLE, NOT PROVEN BETTER** |
| Multitask GraphSAGE vs XGBoost | 0.0599 | 0.0540 | +0.0059 | +0.0074 | [-0.0105, +0.0267] | 79.0% | **COMPARABLE, NOT PROVEN BETTER** |

---

## 5. Final Decision

**Wild-bird auxiliary supervision did not improve poultry-outbreak prediction.** Multitask pooled PR-AUC=0.0599 vs original wBCE GraphSAGE PR-AUC=0.0628 (obs diff=-0.0029, 95% CI=[-0.0101,+0.0064]). The single-task wBCE GraphSAGE remains the best GNN configuration.

> [!IMPORTANT]
> **Leakage Prevention Assurance**:
> 1. `wildbird_hpai_present_next_month` was used **only as an auxiliary training/validation label**, never as a predictor feature.
> 2. All auxiliary labels for month M reference M+1 wild-bird detections, consistent with the primary target definition.
> 3. Scaling, class weights, and CCS aggregation used training-fold data only.
> 4. No test-year labels (poultry or wild-bird) were used to select lambda or the early-stopping epoch.
> 5. Monthly CSD-level predictions passed the constant-prediction QA check across all folds.
> 6. All comparisons use the same 62 audited features and CSD-cluster bootstrap resampling.
> 7. No superseded static CSD-level GraphSAGE results were used.

