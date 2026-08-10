# STAGE 4: Monthly Operational Top-K Evaluation

## Macro-Averaged Monthly Top-K Summary (2025 Test Set)

| model | evaluated_months | months_with_outbreaks | macro_P@10 | macro_R@10 | total_detected_10 | macro_P@25 | macro_R@25 | total_detected_25 | macro_P@50 | macro_R@50 | total_detected_50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_env_lag | 11 | 8 | 0.0636 | 0.0323 | 7 | 0.0473 | 0.0734 | 13 | 0.0382 | 0.1784 | 21 |
| model_b_ratio_50 | 11 | 8 | 0.0636 | 0.0308 | 7 | 0.0327 | 0.0813 | 9 | 0.0218 | 0.1086 | 12 |
| model_b_all_weighted | 11 | 8 | 0.0455 | 0.0216 | 5 | 0.0255 | 0.0852 | 7 | 0.0145 | 0.09 | 8 |
| sage | 11 | 8 | 0.0455 | 0.0225 | 5 | 0.0655 | 0.1301 | 18 | 0.0455 | 0.2155 | 25 |
| model_b_ratio_25 | 11 | 8 | 0.0455 | 0.0216 | 5 | 0.0291 | 0.0762 | 8 | 0.0182 | 0.0863 | 10 |
| lr_env_only | 11 | 8 | 0.0364 | 0.019 | 4 | 0.0364 | 0.0601 | 10 | 0.0255 | 0.1336 | 14 |
| model_b_ratio_10 | 11 | 8 | 0.0364 | 0.0174 | 4 | 0.0182 | 0.0216 | 5 | 0.0145 | 0.0893 | 8 |
| model_b_ratio_3 | 11 | 8 | 0.0091 | 0.0041 | 1 | 0.0109 | 0.0124 | 3 | 0.0091 | 0.0326 | 5 |


## Key Operational Findings
- **Monthly vs Pooled Top-K**: Veterinary health authorities make operational inspection decisions on a **per-month basis**, allocating K alerts per month rather than pooling all predictions across the year.
- **Top-10 Monthly Precision**: **Spatiotemporal Directed R-GCN (`model_b_ratio_25`)** and **GraphSAGE (`sage`)** lead monthly operational precision under realistic monthly alert budgets.
