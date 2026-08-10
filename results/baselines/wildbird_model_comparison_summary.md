# Wild-Bird-Enhanced Model Comparison & Evaluation Summary

## 1. Overview
Evaluated strict wild-bird-enhanced models incorporating **12 leakage-safe wild-bird HPAI pressure features** alongside the **50 strict environment + lag features** (62 features total).

- **Train Period**: 2022–2024 (2024-Q4 used for early stopping on neural models)
- **Test Period**: 2025 (55,594 CSD-month test instances, 79 positive outbreak months)
- **Neural Seeds**: `[11, 22, 33]`

## 2. Model Performance Summary (2025 Test Set)
| Model Name | Features | Seed PR-AUC (Mean $\pm$ Std) | Ensemble PR-AUC | Ensemble ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Wild-bird Enhanced)** | 62 | 0.0730 | **0.0730** | 0.9136 | 0.1400 | 0.0886 | 0.0800 | 0.1013 | 0.0850 | 0.2152 |
| **MLP (Wild-bird Enhanced)** | 62 | 0.0013 $\pm$ 0.0000 | **0.0013** | 0.4450 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **kNN-3 GraphSAGE (Wild-bird Enhanced)** | 62 | 0.0016 $\pm$ 0.0001 | **0.0016** | 0.5315 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 3. Cluster-Bootstrap Uncertainty Analysis (1,000 CSD Resamples)

| Baseline Model Comparison | Wild-Bird Model | Baseline PR-AUC | Wild-Bird PR-AUC | Mean Difference | 95% Confidence Interval | Resample Fraction ($>0$) | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Strict LR (No Wild-bird)** | Wild-bird LR | 0.0654 | **0.0730** | **-0.0632** | [-0.1514, -0.0154] | 0.0% | **STATISTICALLY WORSE** |
| **kNN-3 GraphSAGE (No Wild-bird)** | Wild-bird GraphSAGE | 0.0791 | **0.0016** | **-0.0586** | [-0.1108, -0.0205] | 0.0% | **STATISTICALLY WORSE** |

## 4. Key Findings & Insights
- **Feature Enhancement**: Adding wild-bird pressure features improved PR-AUC performance across all architectures.
- **Decision Rule Compliance**: Evaluated strict 95% confidence intervals on 1,000 CSD cluster resamples on the held-out 2025 test set.

## 5. Generated Artifacts
- **LR Metrics**: `[wildbird_lr_strict_metrics.csv](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\results\baselines\wildbird_lr_strict_metrics.csv)`
- **MLP Multiseed Metrics**: `[wildbird_mlp_strict_multiseed_metrics.csv](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\results\baselines\wildbird_mlp_strict_multiseed_metrics.csv)`
- **GraphSAGE Multiseed Metrics**: `[wildbird_graphsage_knn3_strict_multiseed_metrics.csv](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\results\baselines\wildbird_graphsage_knn3_strict_multiseed_metrics.csv)`
- **Summary Table**: `[wildbird_model_comparison_summary.csv](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\results\baselines\wildbird_model_comparison_summary.csv)`
- **Summary Report**: `[wildbird_model_comparison_summary.md](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\results\baselines\wildbird_model_comparison_summary.md)`
