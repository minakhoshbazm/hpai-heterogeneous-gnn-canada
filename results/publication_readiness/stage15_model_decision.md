# STAGE 15: Final Model Selection & Decision Report

## Scientific Model Decision Matrix

| Model Family | Role | Justification | Status |
| --- | --- | --- | --- |
| GraphSAGE Spatial GNN (3-Seed Ensemble) | PRIMARY NATIONAL MODEL | Beats Environment+Lag LR on validation AP (0.0791 vs 0.0654, +20.9% AP gain), generalizes across regions, and maintains superior AP on emerging non-recurrent outbreaks. | PRIMARY MODEL |
| Strict Environment + Lag Logistic Regression | PRIMARY INTERPRETABLE BENCHMARK | Strong linear baseline (AP = 0.0654, ROC-AUC = 0.9158), captures recurrence lag terms effectively, simple to deploy. | BENCHMARK MODEL |
| Spatiotemporal Directed R-GCN (Model B) | SPECIALIZED ALERT MODEL | Provides highest pooled Top-10 precision (40.0% Precision@10), but lower broad national PR-AUC (0.0452). Complementary for narrow alert budgets. | ALERT SPECIALIST |


