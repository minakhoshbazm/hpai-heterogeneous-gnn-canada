# Pooled Rolling-Origin Model Comparison (2023, 2024, 2025 Test Folds)

## Executive Summary

To evaluate model performance across multiple historical outbreak periods and maximize statistical power, we conducted a **pooled rolling-origin validation** combining predictions from three consecutive temporal test folds:
- **Fold 1 (2023 Test)**: Train 2022 $\rightarrow$ Test 2023 ($N = 60,648$, $N_{\text{pos}} = 72$)
- **Fold 2 (2024 Test)**: Train 2022–2023 $\rightarrow$ Test 2024 ($N = 60,648$, $N_{\text{pos}} = 40$)
- **Fold 3 (2025 Test)**: Train 2022–2024 $\rightarrow$ Test 2025 ($N = 55,594$, $N_{\text{pos}} = 79$)
- **Pooled Total**: $N = 176,890$ CSD-month observations, $N_{\text{pos}} = 191$ positive outbreak months.

---

## 1. Pooled & Per-Fold Performance Metrics

| Model Name | Scope / Fold | PR-AUC | ROC-AUC | Precision@50 | Recall@50 | Precision@100 | Recall@100 |
|---|---|---|---|---|---|---|---|
| **kNN-3 GraphSAGE Ensemble** | **POOLED (2023-2025)** | **0.0558** | **0.9130** | **0.2000** | **0.0524** | **0.1400** | **0.0733** |
| **Relational GNN Ensemble** | **POOLED (2023-2025)** | **0.0552** | **0.9079** | 0.2200 | 0.0576 | 0.1200 | 0.0628 |
| **Strict Env + Lag LR** | **POOLED (2023-2025)** | **0.0519** | **0.8986** | 0.2600 | 0.0681 | 0.1400 | 0.0733 |
| **Strict MLP (No Graph)** | **POOLED (2023-2025)** | **0.0568** | **0.8828** | 0.2400 | 0.0628 | 0.1500 | 0.0785 |

---

## 2. Pooled Cluster-Bootstrap Uncertainty Analysis (1,000 Resamples)

| Pairwise Comparison | Pooled PR-AUC Diff | 95% Percentile Confidence Interval | Resample Fraction (GNN Better) | Formal Decision Conclusion |
|---|---|---|---|---|
| **Time-Safe Relational GNN vs Strict Env + Lag LR** | **+0.0033** | **[-0.0066, +0.0136]** | 0.7210 (72.1%) | **comparable, not proven better** |
| **Time-Safe Relational GNN vs kNN-3 GraphSAGE** | **-0.0006** | **[-0.0075, +0.0074]** | 0.2820 (28.2%) | **comparable, not proven better** |
| **kNN-3 GraphSAGE vs Strict Env + Lag LR** | **+0.0039** | **[-0.0035, +0.0120]** | 0.8590 (85.9%) | **comparable, not proven better** |

---

## 3. Per-Fold Breakdown

### Fold 1 (2023 Test)

| Model Name | PR-AUC | ROC-AUC | Precision@50 | Recall@50 | Precision@100 | Recall@100 |
|---|---|---|---|---|---|---|
| **Strict Environment + Lag Logistic Regression** | 0.0649 | 0.8731 | 0.1400 | 0.0972 | 0.0900 | 0.1250 |
| **Strict MLP (No Graph)** | 0.0564 | 0.8945 | 0.1600 | 0.1111 | 0.1200 | 0.1667 |
| **kNN-3 GraphSAGE Ensemble** | 0.0475 | 0.9068 | 0.0600 | 0.0417 | 0.0600 | 0.0833 |
| **Time-Safe Relational GNN Ensemble** | 0.0515 | 0.9157 | 0.1200 | 0.0833 | 0.0900 | 0.1250 |

### Fold 2 (2024 Test)

| Model Name | PR-AUC | ROC-AUC | Precision@50 | Recall@50 | Precision@100 | Recall@100 |
|---|---|---|---|---|---|---|
| **Strict Environment + Lag Logistic Regression** | 0.0925 | 0.9197 | 0.0800 | 0.1000 | 0.0500 | 0.1250 |
| **Strict MLP (No Graph)** | 0.1373 | 0.9017 | 0.1400 | 0.1750 | 0.1100 | 0.2750 |
| **kNN-3 GraphSAGE Ensemble** | 0.1635 | 0.9375 | 0.1400 | 0.1750 | 0.0900 | 0.2250 |
| **Time-Safe Relational GNN Ensemble** | 0.1486 | 0.9418 | 0.1600 | 0.2000 | 0.1000 | 0.2500 |

### Fold 3 (2025 Test)

| Model Name | PR-AUC | ROC-AUC | Precision@50 | Recall@50 | Precision@100 | Recall@100 |
|---|---|---|---|---|---|---|
| **Strict Environment + Lag Logistic Regression** | 0.0657 | 0.9158 | 0.1200 | 0.0759 | 0.0700 | 0.0886 |
| **Strict MLP (No Graph)** | 0.0513 | 0.9073 | 0.1400 | 0.0886 | 0.0900 | 0.1139 |
| **kNN-3 GraphSAGE Ensemble** | 0.0791 | 0.9263 | 0.1600 | 0.1013 | 0.1300 | 0.1646 |
| **Time-Safe Relational GNN Ensemble** | 0.0698 | 0.9126 | 0.1600 | 0.1013 | 0.1100 | 0.1392 |

