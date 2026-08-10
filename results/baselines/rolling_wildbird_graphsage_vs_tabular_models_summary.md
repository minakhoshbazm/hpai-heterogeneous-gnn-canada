# Rolling Temporal Comparison: Tree Baselines vs Monthly-Snapshot GraphSAGE

**Experiment**: Strict 3-Fold Rolling-Origin Temporal Validation (Test Years: 2023, 2024, 2025)  
**Models evaluated**:
1. Wild-Bird Logistic Regression (loaded from valid rolling predictions — not retrained here)
2. Random Forest (trained here on identical splits and 62-feature set)
3. XGBoost (trained here on identical splits and 62-feature set)
4. Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE Ensemble (loaded from valid rolling predictions — not retrained here)

---

## 1. Per-Fold Comparison Table

| Fold | Test Year | Test Rows | Pos | Prev | Model | PR-AUC | Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 | Val PR-AUC | Selected Params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2023 | 60,648 | 72 | 0.001187 | Logistic Regression | 0.0675 | 56.82x | 0.8704 | 0.1400 | 0.0972 | 0.1000 | 0.1389 | 0.0600 | 0.1667 | — | — |
| 2 | 2024 | 60,648 | 40 | 0.000660 | Logistic Regression | 0.0873 | 132.41x | 0.9202 | 0.0800 | 0.1000 | 0.0500 | 0.1250 | 0.0350 | 0.1750 | — | — |
| 3 | 2025 | 55,594 | 79 | 0.001421 | Logistic Regression | 0.0679 | 47.77x | 0.9125 | 0.1000 | 0.0633 | 0.0800 | 0.1013 | 0.0650 | 0.1646 | — | — |
| 1 | 2023 | 60,648 | 72 | 0.001187 | Random Forest | 0.0399 | 33.60x | 0.9190 | 0.0800 | 0.0556 | 0.0800 | 0.1111 | 0.0900 | 0.2500 | 0.0516 | `{'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 20, 'max_features': 'sqrt'}` |
| 1 | 2023 | 60,648 | 72 | 0.001187 | XGBoost | 0.0213 | 17.96x | 0.8956 | 0.0000 | 0.0000 | 0.0300 | 0.0417 | 0.0450 | 0.1250 | 0.0472 | `{'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 10, 'subsample': 0.8, 'n_rounds': 18}` |
| 2 | 2024 | 60,648 | 40 | 0.000660 | Random Forest | 0.1183 | 179.43x | 0.9318 | 0.0800 | 0.1000 | 0.0700 | 0.1750 | 0.0450 | 0.2250 | 0.2031 | `{'n_estimators': 500, 'max_depth': 12, 'min_samples_leaf': 5, 'max_features': 'sqrt'}` |
| 2 | 2024 | 60,648 | 40 | 0.000660 | XGBoost | 0.1952 | 296.02x | 0.9272 | 0.1600 | 0.2000 | 0.0800 | 0.2000 | 0.0500 | 0.2500 | 0.2216 | `{'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 10, 'subsample': 0.9, 'n_rounds': 120}` |
| 3 | 2025 | 55,594 | 79 | 0.001421 | Random Forest | 0.0351 | 24.72x | 0.9397 | 0.0600 | 0.0380 | 0.0500 | 0.0633 | 0.0450 | 0.1139 | 0.2497 | `{'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 10, 'max_features': 'sqrt'}` |
| 3 | 2025 | 55,594 | 79 | 0.001421 | XGBoost | 0.0375 | 26.37x | 0.9327 | 0.0600 | 0.0380 | 0.0600 | 0.0759 | 0.0450 | 0.1139 | 0.2737 | `{'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 10, 'subsample': 0.8, 'n_rounds': 146}` |
| 1 | 2023 | 60,648 | 72 | 0.001187 | Monthly-Snapshot GraphSAGE (Ensemble) | 0.0456 | 38.37x | 0.8800 | 0.1200 | 0.0833 | 0.0900 | 0.1250 | 0.0650 | 0.1806 | — | — |
| 2 | 2024 | 60,648 | 40 | 0.000660 | Monthly-Snapshot GraphSAGE (Ensemble) | 0.0914 | 138.54x | 0.9388 | 0.2000 | 0.2500 | 0.1300 | 0.3250 | 0.0650 | 0.3250 | — | — |
| 3 | 2025 | 55,594 | 79 | 0.001421 | Monthly-Snapshot GraphSAGE (Ensemble) | 0.0738 | 51.93x | 0.9332 | 0.1600 | 0.1013 | 0.1100 | 0.1392 | 0.0700 | 0.1772 | — | — |

---

## 2. Pooled Out-of-Sample Comparison (176,890 Test Rows, 2023–2025)

| Model | Pooled PR-AUC | Lift | Pooled ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.0540 | 49.99x | 0.8967 | 0.2600 | 0.0681 | 0.1500 | 0.0785 | 0.0900 | 0.0942 |
| Random Forest | 0.0401 | 37.14x | 0.9274 | 0.1600 | 0.0419 | 0.1200 | 0.0628 | 0.0800 | 0.0838 |
| XGBoost | 0.0540 | 49.97x | 0.8661 | 0.2000 | 0.0524 | 0.1100 | 0.0576 | 0.0650 | 0.0681 |
| **Monthly-Snapshot GraphSAGE (Ensemble)** | **0.0628** | 58.16x | **0.8971** | 0.2400 | 0.0628 | 0.1600 | 0.0838 | 0.1350 | 0.1414 |

---

## 3. Bootstrap Decisions: GraphSAGE vs Each Tabular Baseline

| Comparison | Observed Pooled PR-AUC Diff | Bootstrap Mean | 95% CI | GraphSAGE Win Rate | Decision |
|---|---|---|---|---|---|
| GraphSAGE vs LR | +0.0088 | +0.0083 | [+0.0006, +0.0176] | 98.5% | **GRAPHSAGE STATISTICALLY BETTER** |
| GraphSAGE vs RF | +0.0227 | +0.0223 | [-0.0027, +0.0535] | 94.5% | **COMPARABLE, NOT PROVEN BETTER** |
| GraphSAGE vs XGB | +0.0088 | +0.0100 | [-0.0081, +0.0291] | 86.3% | **COMPARABLE, NOT PROVEN BETTER** |

---

## 4. Strongest Valid Model & Remaining Uncertainty

**Strongest model by pooled PR-AUC**: **Monthly-Snapshot GraphSAGE** (`0.0628`)

- **GraphSAGE vs LR**: GRAPHSAGE STATISTICALLY BETTER (CI: [+0.0006, +0.0176], Win Rate: 98.5%)
- **GraphSAGE vs RF**: COMPARABLE, NOT PROVEN BETTER (CI: [-0.0027, +0.0535], Win Rate: 94.5%)
- **GraphSAGE vs XGB**: COMPARABLE, NOT PROVEN BETTER (CI: [-0.0081, +0.0291], Win Rate: 86.3%)

> [!IMPORTANT]
> **METHODOLOGICAL ASSURANCE**:
> 1. All GraphSAGE predictions were loaded from the **valid monthly-snapshot rolling output** (`wildbird_snapshot_gs_ensemble_prob`). No superseded static CSD-level GNN results were used.
> 2. Random Forest and XGBoost were trained on identical 62-feature sets, rolling temporal splits, and imputation-from-training-data constraints as the GNN.
> 3. Feature imputation used training-set medians only; no test-year information influenced any model selection or retraining step.

