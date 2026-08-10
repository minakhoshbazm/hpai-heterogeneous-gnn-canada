# Corrected Wild‑bird kNN‑3 GraphSAGE Model

**Training period**: 2022‑2024 (2024‑Q4 validation)
**Test period**: 2025
**Seeds**: [11, 22, 33]
**Feature count**: 62

## Per‑seed metrics
- Seed 11: PR‑AUC 0.0465, ROC‑AUC 0.9221, P@50 0.1600, R@50 0.1013
- Seed 22: PR‑AUC 0.0545, ROC‑AUC 0.9192, P@50 0.1800, R@50 0.1139
- Seed 33: PR‑AUC 0.0509, ROC‑AUC 0.9179, P@50 0.1600, R@50 0.1013

## Ensemble metrics
- PR‑AUC 0.0591
- ROC‑AUC 0.9281
- P@50 0.1600, R@50 0.1013

## Prediction sanity checks (last seed)
- Min prob: 0.000000
- Max prob: 1.000000
- Mean prob: 0.269046
- Std dev: 0.193209
- Unique values: 55496

## Bootstrap comparison vs baseline GraphSAGE (strict)
- Mean diff: -0.0002
- 95 % CI: [-0.0175, +0.0167]
- Resample >0 %: 50.4
- Decision: **COMPARABLE, NOT PROVEN BETTER**

## Bootstrap comparison vs wild‑bird LR (strict)
- Mean diff: -0.0055
- 95 % CI: [-0.0703, +0.0393]
- Resample >0 %: 46.6
- Decision: **COMPARABLE, NOT PROVEN BETTER**

## Artifacts
- Metrics CSV: [wildbird_graphsage_knn3_corrected_multiseed_metrics.csv](file:///D:/gdrive/models/hpai-heterogeneous-gnn-canada/results/baselines/wildbird_graphsage_knn3_corrected_multiseed_metrics.csv)
- Test predictions CSV: [wildbird_graphsage_knn3_corrected_test_predictions.csv](file:///D:/gdrive/models/hpai-heterogeneous-gnn-canada/results/baselines/wildbird_graphsage_knn3_corrected_test_predictions.csv)
- Summary markdown (this file): [wildbird_graphsage_knn3_corrected_summary.md](file:///D:/gdrive/models/hpai-heterogeneous-gnn-canada/results/baselines/wildbird_graphsage_knn3_corrected_summary.md)
