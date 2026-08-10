# Rolling Supervised Transfer Learning: Wild-Bird Pretraining → Poultry Fine-Tuning

**Architecture**: kNN-3 Monthly-Snapshot GraphSAGE (same graph as valid rolling baseline)
**Pretraining Task**: Wild-bird HPAI presence prediction (next month) — 7,234 positives
**Fine-tuning Task**: Poultry HPAI outbreak prediction (next month) — 381 positives
**Wild-bird pretraining signal is 19.0x denser than poultry signal.**

## Rolling Fold Configurations
- Fold 1: Final-Train 2022, Test 2023
- Fold 2: Final-Train 2022-2023, Test 2024
- Fold 3: Final-Train 2022-2024, Test 2025

## Per-Fold Test Performance (Ensemble of 3 Seeds)

| Fold | Test Year | Test Rows | Positives | PR-AUC | Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2023 | 60,648 | 72 | **0.0565** | 47.6x | 0.8721 | 0.200 | 0.139 | 0.110 | 0.153 | 0.065 | 0.181 |
| 2 | 2024 | 60,648 | 40 | **0.1591** | 241.2x | 0.9263 | 0.220 | 0.275 | 0.110 | 0.275 | 0.065 | 0.325 |
| 3 | 2025 | 55,594 | 79 | **0.0648** | 45.6x | 0.9241 | 0.180 | 0.114 | 0.120 | 0.152 | 0.070 | 0.177 |

## Pooled Out-of-Sample Results (2023-2025 Combined, 176,890 rows, 191 positives)

| Model | Pooled PR-AUC | Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|
| **Supervised Transfer GraphSAGE (Ensemble)** | **0.0720** | 66.7x | 0.9054 | 0.300 | 0.079 | 0.210 | 0.110 | 0.160 | 0.168 |

## Bootstrap Comparison (1,000 CSD Cluster Resamples)

| Comparator | Transfer PR-AUC | Baseline PR-AUC | Observed Diff | 95% CI | Win Rate | Decision |
|---|---|---|---|---|---|---|
| Monthly-Snapshot GraphSAGE (Rolling) | 0.0720 | 0.0628 | +0.0092 | [-0.0057, +0.0288] | 83.9% | **COMPARABLE, NOT PROVEN BETTER** |
| Wild-Bird Logistic Regression (Rolling) | 0.0720 | 0.0540 | +0.0180 | [+0.0014, +0.0399] | 98.6% | **STATISTICALLY BETTER** |

## Methodological Assurances
1. Scaler fit on training-fold data only; applied identically to val/test.
2. Wild-bird pretraining target constructed from raw CFIA wild-bird records using validated CCS crosswalk.
3. Dev-val used exclusively for early stopping; test year never accessed during development.
4. Optimal pretrain/finetune epochs determined from dev-val, applied to final retrain on full training fold.
5. Bootstrap uses CSD-cluster resampling to preserve spatial-temporal correlation.
6. No test-year labels (poultry or wild-bird) influenced any model selection decision.
