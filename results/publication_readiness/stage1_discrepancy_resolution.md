# STAGE 1: Discrepancy Resolution Report

## 1. GraphSAGE PR-AUC Discrepancy (0.0541 vs 0.0791)
- **Mean Individual Single-Seed AP**: **0.0541** (±0.0020) across Seed 11 (0.0542), Seed 22 (0.0518), Seed 33 (0.0563).
- **3-Seed Ensemble Average AP**: **0.0791** (`average_precision_score`) / **0.0767** (trapezoidal `auc`).
- **Scientific Resolution**: Ensembling 3 independent random seeds reduces prediction variance in top risk logits, boosting AP by **+0.0250**. Both evaluations used identical 55,594 rows and 79 positive labels.

## 2. Persistence Baseline Arithmetic & Metric Correction
- **High-Score Predictions (Month t Outbreak = 1)**: 68
- **True Positives (Month t+1 Outbreak = 1)**: 8
- **False Positives**: 60
- **Total Outbreaks in 2025 Test Set**: 79
- **Precision**: 0.1176 (8/68)
- **Recall**: 0.1013 (8/79)
- **Canonical Average Precision (AP)**: **0.013191**
- **Exact AP Formula**: `(true_pos / total_pos) * (true_pos / high_score_n)` = `(8/79) * (8/68)` = **0.011914**
- **Trapezoidal PR-AUC**: 0.110095
- **ROC-AUC**: 0.550093
- **Arithmetic Correction**: Previous trapezoidal integration assumed linear interpolation between (0,0) and (R_1, P_1), overestimating area as ~0.0405. Canonical `average_precision_score` is **0.013191**.

## 3. Commercial Cohort Naming & Verification
- **Cohort A**: `ccs_farms_reporting > 0` (2,203 CSDs, 24,233 test rows, 38 outbreaks, prevalence = 0.1568%)
- **Cohort B**: `ccs_farms_reporting > 0 OR ccs_production_kg > 0` (2,222 CSDs, 24,442 test rows, 45 outbreaks, prevalence = 0.1841%)
