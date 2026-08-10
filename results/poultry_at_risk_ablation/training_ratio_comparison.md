# Negative-Sampling Ablation: Spatiotemporal Directed RGCN Transfer

## Training Ratio Comparison (2025 Test Set, 3-Seed Ensemble)

| Config | Ratio | PR-AUC | ±Std | ROC-AUC | Random Baseline | Lift | P@10 | P@25 | P@50 | P@100 | R@50 | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **all_weighted** | all | **0.0031** | ±0.0119 | 0.6305 | 0.0014 | 2.2x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.3930 |
| **ratio_3** | 3 | **0.0255** | ±0.0108 | 0.7545 | 0.0014 | 18.0x | 0.200 | 0.120 | 0.100 | 0.070 | 0.063 | 0.1639 |
| **ratio_10** | 10 | **0.0042** | ±0.0120 | 0.6529 | 0.0014 | 3.0x | 0.000 | 0.000 | 0.000 | 0.010 | 0.000 | 0.3925 |
| **ratio_25** | 25 | **0.0040** | ±0.0120 | 0.6449 | 0.0014 | 2.8x | 0.000 | 0.000 | 0.000 | 0.010 | 0.000 | 0.3921 |
| **ratio_50** | 50 | **0.0033** | ±0.0119 | 0.6351 | 0.0014 | 2.3x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.3917 |

## Pairwise Bootstrap vs all_weighted (1,000 month-block resamples)

| Config | Observed Diff | 95% CI | Win Rate | Decision |
|---|---|---|---|---|
| ratio_3 | +0.0224 | [-0.0003, +0.0441] | 94.2% | **COMPARABLE, NOT PROVEN BETTER** |
| ratio_10 | +0.0011 | [-0.0003, +0.0031] | 88.9% | **COMPARABLE, NOT PROVEN BETTER** |
| ratio_25 | +0.0009 | [-0.0003, +0.0025] | 88.8% | **COMPARABLE, NOT PROVEN BETTER** |
| ratio_50 | +0.0002 | [-0.0002, +0.0006] | 77.9% | **COMPARABLE, NOT PROVEN BETTER** |

---

## Technical Notes
- Phase 1 (wild-bird pretraining) is identical across all configs.
- Negative sampling applies only to Phase 2 (poultry fine-tuning).
- Negatives sampled per month: 1/3 random + 1/3 spatial (kNN-3 neighbors of positives) + 1/3 environmental (water-habitat similarity).
- IPW correction applied to sampled negatives: weight = N_neg_total / N_neg_sampled.
- Negatives resampled every epoch (stochastic training).
- Full graph context preserved in forward pass; only loss mask changes.
- Evaluation is on the complete natural-prevalence locked 2025 test set (55,594 rows, 79 positives).
- No test-year data influenced any training, sampling, or model selection decision.
