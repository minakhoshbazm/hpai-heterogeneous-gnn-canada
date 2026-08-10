# Strict Rolling-Origin Temporal Comparison: Wild-Bird Snapshot GraphSAGE vs LR

**Evaluation Protocol**: Strict Rolling-Origin Temporal Validation (2023, 2024, 2025 Test Years)  
**Models Compared**:
1. Wild-Bird Logistic Regression Baseline
2. Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE (3-Seed Ensemble)

---

## 1. Per-Fold Comparison Table

| Fold | Test Year | Test Rows | Positives | Prevalence | Model | PR-AUC | PR-AUC Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2023 | 60,648 | 72 | 0.001187 | Wild-Bird Logistic Regression | **0.0675** | 56.82x | 0.8704 | 0.1400 | 0.0972 | 0.1000 | 0.1389 | 0.0600 | 0.1667 |
| 1 | 2023 | 60,648 | 72 | 0.001187 | Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE (Ensemble) | **0.0456** | 38.37x | 0.8800 | 0.1200 | 0.0833 | 0.0900 | 0.1250 | 0.0650 | 0.1806 |
| 2 | 2024 | 60,648 | 40 | 0.000660 | Wild-Bird Logistic Regression | **0.0873** | 132.41x | 0.9202 | 0.0800 | 0.1000 | 0.0500 | 0.1250 | 0.0350 | 0.1750 |
| 2 | 2024 | 60,648 | 40 | 0.000660 | Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE (Ensemble) | **0.0914** | 138.54x | 0.9388 | 0.2000 | 0.2500 | 0.1300 | 0.3250 | 0.0650 | 0.3250 |
| 3 | 2025 | 55,594 | 79 | 0.001421 | Wild-Bird Logistic Regression | **0.0679** | 47.77x | 0.9125 | 0.1000 | 0.0633 | 0.0800 | 0.1013 | 0.0650 | 0.1646 |
| 3 | 2025 | 55,594 | 79 | 0.001421 | Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE (Ensemble) | **0.0738** | 51.93x | 0.9332 | 0.1600 | 0.1013 | 0.1100 | 0.1392 | 0.0700 | 0.1772 |

---

## 2. Out-of-Sample Pooled Evaluation (2023–2025 Combined: 176,890 Rows)

| Model | Pooled PR-AUC | Pooled ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|
| Wild-Bird Logistic Regression Baseline | 0.0540 | 0.8967 | 0.2600 | 0.0681 | 0.1500 | 0.0785 | 0.0900 | 0.0942 |
| **Wild-Bird Monthly-Snapshot GraphSAGE (Ensemble)** | **0.0628** | **0.8971** | **0.2400** | **0.0628** | **0.1600** | **0.0838** | **0.1350** | **0.1414** |

---

## 3. Pooled 1,000 CSD-Cluster Bootstrap Analysis

- **Observed Pooled PR-AUC Difference**: +0.0088
- **Bootstrap Mean Difference**: +0.0083
- **95% Percentile Confidence Interval**: [+0.0006, +0.0176]
- **GraphSAGE Win Rate**: 98.5%
- **Statistical Decision**: **STATISTICALLY BETTER**

---

## 4. Methodological & Temporal QA Statement

> [!IMPORTANT]
> **TEMPORAL INTEGRITY & SUPERSEDED MODEL ASSURANCE**:  
> 1. Every GraphSAGE prediction in this experiment was generated at the **true CSD-month snapshot level** using distinct monthly feature matrices and shared spatial kNN-3 topology.  
> 2. Zero static CSD-level feature matrices (`X_all`) were used, and zero predictions were broadcast across time. Within-CSD monthly score standard deviations were positive across all folds.  
> 3. All prior static CSD-level GraphSAGE results remain **SUPERSEDED** and were not loaded or referenced as comparators.

