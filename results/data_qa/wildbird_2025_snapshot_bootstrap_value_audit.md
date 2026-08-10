# Audit of 2025 Wild-Bird Snapshot Bootstrap Values

**Date**: 2026-08-03  
**Purpose**: Disambiguate direct observed point difference vs. 1,000 CSD-cluster bootstrap mean difference.

---

## 1. Value Disambiguation Summary

| Metric / Estimation Method | Value | Description |
|---|---|---|
| **GraphSAGE Test PR-AUC** | **0.073793** (`0.0738`) | Direct out-of-sample PR-AUC on full 2025 test dataset |
| **Retrained LR Test PR-AUC** | **0.067886** (`0.0679`) | Direct out-of-sample PR-AUC on full 2025 test dataset |
| **Direct Observed Point Difference** | **+0.005906** (`+0.0059`) | Exact mathematical difference (`0.073793 - 0.067886`) |
| **Bootstrap Mean Difference** | **+0.009490** (`+0.0095`) | Expected mean difference across 1,000 CSD-cluster resamples |
| **Bootstrap 95% Percentile CI** | **[-0.0142, +0.0434]** | 2.5th to 97.5th percentile of resampled differences |
| **GraphSAGE Win Rate** | **75.3%** | Percentage of bootstrap resamples where GraphSAGE > LR |

---

## 2. Explanation of Difference between Point Difference and Bootstrap Mean

- **Direct Observed Point Difference (`+0.0059`)**: The exact difference between the two models evaluated on the complete 2025 test set (55,594 CSD-month rows).
- **Bootstrap Mean Difference (`+0.0095`)**: The mean of the 1,000 differences calculated on CSD-cluster resampled datasets (with replacement). 
- **Reason for Divergence**: Precision-Recall AUC is a non-linear, rank-ordered metric. Resampling CSD clusters with replacement creates variation in local positive counts and ranking density across resamples. Because PR-AUC is non-linear, the expected value of resampled PR-AUC differences is not strictly identical to the point estimate on the un-resampled dataset, resulting in a standard, minor non-linear estimator shift.
- **Scientific Significance**: The confidence interval (`[-0.0142, +0.0434]`) includes zero in both cases. Therefore, regardless of whether point difference (`+0.0059`) or bootstrap mean (`+0.0095`) is reported, the statistical decision remains unchanged: **`COMPARABLE, NOT PROVEN BETTER`**.

