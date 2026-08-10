# Negative-Sampling Ablation: Spatiotemporal Directed RGCN Transfer

## Training Ratio Comparison (2025 Test Set, 3-Seed Ensemble)

| Config | Ratio | PR-AUC | ±Std | ROC-AUC | Random Baseline | Lift | P@10 | P@25 | P@50 | P@100 | R@50 | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **all_weighted** | all | **0.0357** | ±0.0111 | 0.6384 | 0.0014 | 25.1x | 0.300 | 0.160 | 0.080 | 0.050 | 0.051 | 0.1063 |
| **ratio_3** | 3 | **0.0101** | ±0.0107 | 0.7528 | 0.0014 | 7.1x | 0.000 | 0.040 | 0.020 | 0.040 | 0.013 | 0.0936 |
| **ratio_10** | 10 | **0.0249** | ±0.0101 | 0.7399 | 0.0014 | 17.5x | 0.300 | 0.160 | 0.080 | 0.050 | 0.051 | 0.1396 |
| **ratio_25** | 25 | **0.0452** | ±0.0100 | 0.6636 | 0.0014 | 31.8x | 0.400 | 0.200 | 0.140 | 0.090 | 0.089 | 0.1665 |
| **ratio_50** | 50 | **0.0417** | ±0.0099 | 0.7192 | 0.0014 | 29.4x | 0.300 | 0.280 | 0.160 | 0.090 | 0.101 | 0.1335 |

## Pairwise Bootstrap vs all_weighted (1,000 month-block resamples)

| Config | Observed Diff | 95% CI | Win Rate | Decision |
|---|---|---|---|---|
| ratio_3 | -0.0256 | [-0.0830, +0.0026] | 33.1% | **COMPARABLE, NOT PROVEN BETTER** |
| ratio_10 | -0.0108 | [-0.0253, +0.0095] | 34.0% | **COMPARABLE, NOT PROVEN BETTER** |
| ratio_25 | +0.0095 | [-0.0001, +0.0274] | 96.5% | **COMPARABLE, NOT PROVEN BETTER** |
| ratio_50 | +0.0061 | [-0.0059, +0.0305] | 89.2% | **COMPARABLE, NOT PROVEN BETTER** |

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
