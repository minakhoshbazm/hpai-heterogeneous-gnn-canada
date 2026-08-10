# STAGE 7: Probability Calibration Report

## Calibration Benchmark Table (2025 Test Set)

| Model Variant | Average Precision (AP) | ROC-AUC | Brier Score | Ranking Changed? |
| --- | --- | --- | --- | --- |
| Uncalibrated Raw Model Output | 0.065369 | 0.915754 | 0.074643 | No |
| Platt-Calibrated Probabilities | 0.065369 | 0.915754 | 0.001396 | No (Monotonic) |


## Key Calibration Audit Findings
- **Monotonic Preserving**: Platt scaling applies a monotonic sigmoid transformation, leaving ranking metrics (AP = 0.0654, ROC-AUC = 0.9158) exactly unchanged.
- **Brier Score Reduction**: Platt calibration adjusts probability scaling to reflect real 2025 prevalence (0.142%), reducing raw Brier Score error.
