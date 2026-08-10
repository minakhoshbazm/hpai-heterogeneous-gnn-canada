# Relational GNN vs Baselines Cluster-Bootstrap Uncertainty Analysis

## Executive Summary

To evaluate statistical uncertainty and determine whether the multi-relation Relational GNN (RGCN) ensemble outperforms baseline models, we conducted **1,000 cluster-bootstrap iterations** resampling CSD units (retaining all 12 monthly observations per sampled CSD) on the 2025 held-out test set.

## Summary Results Table

| Baseline Model | Relational GNN PR-AUC | Baseline PR-AUC | Observed Diff | 95% Bootstrap CI | Resample Fraction (RGCN Better) | Conclusion |
|---|---|---|---|---|---|---|
| **Strict Env + Lag Logistic Regression** | 0.0698 | 0.0654 | +0.0045 | [-0.0240, +0.0338] | 67.6% (0.6760) | **comparable, not proven better** |
| **kNN-3 GraphSAGE Ensemble** | 0.0698 | 0.0791 | -0.0092 | [-0.0303, +0.0076] | 14.9% (0.1490) | **comparable, not proven better** |

## Statistical Decision Interpretations

### 1. Relational GNN vs. Strict Environment + Lag Logistic Regression
- **Observed Difference**: +0.0045 (0.0698 vs 0.0654)
- **95% Confidence Interval**: [-0.0240, +0.0338]
- **Conclusion**: The 95% confidence interval contains zero; therefore, the Relational GNN is comparable, not proven better than strict Logistic Regression.
- **Resample Fraction**: In 67.6% of bootstrap resamples, the Relational GNN achieved a higher PR-AUC than Logistic Regression.

### 2. Relational GNN vs. kNN-3 GraphSAGE Ensemble
- **Observed Difference**: -0.0092 (0.0698 vs 0.0791)
- **95% Confidence Interval**: [-0.0303, +0.0076]
- **Conclusion**: The 95% confidence interval contains zero; therefore, the Relational GNN is comparable, not proven better than the kNN-3 GraphSAGE ensemble.
- **Resample Fraction**: In 14.9% of bootstrap resamples, the Relational GNN achieved a higher PR-AUC than kNN-3 GraphSAGE.

---
*Report generated automatically by `scripts/compare_relational_gnn_vs_baselines_bootstrap.py`.*
