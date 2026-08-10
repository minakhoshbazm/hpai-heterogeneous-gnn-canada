# Loss Function Ablation: Wild-Bird Monthly-Snapshot GraphSAGE

**Purpose**: Test whether imbalance-aware focal or class-balanced loss improves the valid GraphSAGE.  
**Constraint**: Loss selected by validation PR-AUC only. No test-year data used for selection.

---

## 1. Validation-Selected Loss Settings per Fold

| Fold | Test Year | Selected Loss | Mean 3-Seed Val PR-AUC | Weighted-BCE Val PR-AUC | Improvement over wBCE | wBCE Still Best? |
|---|---|---|---|---|---|---|
| 1 | 2023 | `cb_focal_g1.0_b0.999` | 0.2013 | 0.2003 | +0.0010 | NO |
| 2 | 2024 | `focal_g1.0_a0.75` | 0.1716 | 0.1491 | +0.0225 | NO |
| 3 | 2025 | `focal_g2.0_a0.5` | 0.2818 | 0.2596 | +0.0222 | NO |

> [!NOTE]
> Loss selection was done **exclusively using the fold development-validation period**. No test-year predictions informed any selection decision.

---

## 2. Pooled Out-of-Sample Comparison (176,890 rows, 2023-2025)

| Model | Pooled PR-AUC | Lift | Pooled ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|
| **Loss-Ablation GraphSAGE (Best Loss)** | **0.0275** | 25.43x | **0.8935** | 0.1400 | 0.0366 | 0.1000 | 0.0524 | 0.0550 | 0.0576 |
| Original weighted-BCE GraphSAGE (Ensemble) | 0.0628 | 58.16x | 0.8971 | 0.2400 | 0.0628 | 0.1600 | 0.0838 | 0.1350 | 0.1414 |
| Logistic Regression | 0.0540 | 49.99x | 0.8967 | 0.2600 | 0.0681 | 0.1500 | 0.0785 | 0.0900 | 0.0942 |
| Random Forest | 0.0401 | 37.14x | 0.9274 | 0.1600 | 0.0419 | 0.1200 | 0.0628 | 0.0800 | 0.0838 |
| XGBoost | 0.0540 | 49.97x | 0.8661 | 0.2000 | 0.0524 | 0.1100 | 0.0576 | 0.0650 | 0.0681 |

---

## 3. Bootstrap Decisions (Loss-Ablation GraphSAGE vs Each Model)

| Comparison | Ablation PR-AUC | Reference PR-AUC | Observed Diff | Bootstrap Mean | 95% CI | Win Rate | Decision |
|---|---|---|---|---|---|---|---|
| Loss-Ablation GraphSAGE vs original weighted-BCE GraphSAGE | 0.0275 | 0.0628 | -0.0353 | -0.0339 | [-0.0765, -0.0040] | 0.2% | **STATISTICALLY WORSE** |
| Loss-Ablation GraphSAGE vs Logistic Regression | 0.0275 | 0.0540 | -0.0265 | -0.0255 | [-0.0675, +0.0023] | 10.1% | **COMPARABLE, NOT PROVEN BETTER** |
| Loss-Ablation GraphSAGE vs Random Forest | 0.0275 | 0.0401 | -0.0126 | -0.0117 | [-0.0285, -0.0011] | 1.4% | **STATISTICALLY WORSE** |
| Loss-Ablation GraphSAGE vs XGBoost | 0.0275 | 0.0540 | -0.0265 | -0.0244 | [-0.0619, +0.0019] | 5.7% | **COMPARABLE, NOT PROVEN BETTER** |

---

## 4. Final Decision

**Weighted BCE remains the final loss choice.** Alternative loss configurations did not improve pooled test PR-AUC despite selection on validation data.

> [!IMPORTANT]
> **METHODOLOGICAL ASSURANCE**:
> 1. All GraphSAGE variants used the valid **monthly-snapshot construction**: one graph per `(year, month)`, actual time-varying features per CSD-month, kNN-3 within-snapshot edges, shared GNN weights.
> 2. No test-year rows informed loss selection in any fold.
> 3. No superseded static CSD-level GNN predictions were used.
> 4. All comparisons use the same 62 audited features, identical temporal splits, and CSD-cluster bootstrap resampling.

