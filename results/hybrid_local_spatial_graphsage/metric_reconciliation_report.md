# Metric Variance Reconciliation & Audit Report

## Exact Reasons for Metric Differences Across Pipelines

| Model Family | Earlier Quick-Eval Metric | Formal 5-Seed Pipeline Metric | Root Cause of Variance |
| --- | --- | --- | --- |
| Standalone MLP | AP = 0.1034 (1-seed) | AP = 0.0742 (5-seed mean) | Earlier quick-eval used scikit-learn MLP (1 layer, max_iter=40, fit on 2022-2024). Formal pipeline used 2-layer PyTorch MLP with 0.2 dropout, trained on 2022-2023 with early stopping on 2024 validation. |
| Standalone GraphSAGE | AP = 0.0791 (3-seed ensemble) | AP = 0.0903 (5-seed ensemble) | Expanding from 3 seeds to 5 seeds ([11, 22, 33, 44, 55]) further reduced prediction variance across positive logits, boosting AP by +0.0112. |
| Shuffled-Edge Control | N/A | AP = 0.0756, ROC-AUC = 0.9287, BSS = +0.0111 | Random edges act as global graph regularization (improving broad ROC-AUC), but degrade local spatial topology, dropping monthly Top-50 recall from 21.32% to 16.65%. |


