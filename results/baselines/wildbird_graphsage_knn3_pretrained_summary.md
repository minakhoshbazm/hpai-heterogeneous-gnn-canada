# Leakage-Safe Self-Supervised Pretrained Wild-Bird kNN-3 GraphSAGE Summary

**Pretraining Task**: Masked Node-Feature Reconstruction (20% masking per pass)  
**Supervised Task**: HPAI Outbreak Next Month Prediction  
**Training Period**: 2022-01 through 2024-09  
**Validation Period**: 2024-10 through 2024-12  
**Test Period**: 2025 (Held-out single evaluation)  
**Feature Count**: 62  

## Multi-Seed Results

| Seed | Pretrain Epoch | Finetune Epoch | Val PR-AUC | Test PR-AUC | Test ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 142 | 9 | 0.2410 | 0.0453 | 0.9308 | 0.1400 | 0.0886 | 0.1000 | 0.1266 | 0.0650 | 0.1646 |
| 22 | 61 | 8 | 0.2486 | 0.0485 | 0.9307 | 0.1400 | 0.0886 | 0.1000 | 0.1266 | 0.0650 | 0.1646 |
| 33 | 102 | 5 | 0.2754 | 0.0532 | 0.9270 | 0.1600 | 0.1013 | 0.1000 | 0.1266 | 0.0600 | 0.1519 |
| **Ensemble** | 101 | 7 | 0.2550 | **0.0513** | **0.9331** | **0.1600** | **0.1013** | **0.1000** | **0.1266** | **0.0650** | **0.1646** |

## Prediction Sanity Checks (Ensemble)
- Min Prob: 0.000000
- Max Prob: 1.000000
- Mean Prob: 0.202178
- Std Dev: 0.231400
- Unique Predictions: 55552

## Bootstrap Comparison (1,000 Resamples)

### vs. Corrected Wild-Bird kNN-3 GraphSAGE (No Pretraining)
- Baseline PR-AUC: 0.0591
- Pretrained PR-AUC: 0.0513
- Observed Mean Diff: -0.0067
- 95% Percentile CI: [-0.0164, +0.0006]
- Pretrained Win Rate: 4.3%
- Decision: **COMPARABLE, NOT PROVEN BETTER**

### vs. Wild-Bird Logistic Regression
- Baseline PR-AUC: 0.0670
- Pretrained PR-AUC: 0.0513
- Observed Mean Diff: -0.0122
- 95% Percentile CI: [-0.0785, +0.0303]
- Pretrained Win Rate: 41.0%
- Decision: **COMPARABLE, NOT PROVEN BETTER**

## Model Comparison

| Model | Test PR-AUC | Test ROC-AUC | P@50 | R@50 |
|---|---|---|---|---|
| Wild-Bird Logistic Regression | 0.0670 | 0.9167 | 0.0800 | 0.0506 |
| Corrected Wild-Bird kNN-3 GraphSAGE (No Pretraining) | 0.0591 | 0.9281 | 0.1600 | 0.1013 |
| **Wild-Bird kNN-3 GraphSAGE (With Pretraining)** | **0.0513** | **0.9331** | **0.1600** | **0.1013** |

