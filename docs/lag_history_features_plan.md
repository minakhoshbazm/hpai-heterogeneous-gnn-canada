# Lag/History Features Plan for V1 Modeling

## Purpose
Plan leakage-safe historical outbreak features that can improve the environment-only models and later GNNs.

## Motivation
The time-persistence baseline performed strongly in top-K ranking, showing that historical outbreak recurrence/location is highly informative.

## Candidate lag/history features
- `same_csd_outbreak_last_1_month`
- `same_csd_outbreak_last_3_months`
- `same_csd_outbreak_last_6_months`
- `same_csd_outbreak_last_12_months`
- `same_csd_outbreak_count_past_12_months`
- `province_outbreak_count_last_1_month`
- `province_outbreak_count_last_3_months`
- `neighboring_csd_outbreak_count_last_1_month`
- `neighboring_csd_outbreak_count_last_3_months`

## Leakage-safety rule
For a row representing CSD-month **M**, lag/history features may only use outbreak information from months **strictly before or equal to M**. They must never incorporate the target month (M + 1) or any future months.

## Validation checklist
- No future labels are used in any lag feature.
- Lag features are correctly shifted (e.g., `last_3_months` uses data from M-1, M-2, M-3).
- For the earliest months where lag data is unavailable, values should be set to zero or NaN and then safely imputed.
- Compare model performance with and without lag/history features to quantify benefit.

## Expected output
The final enriched table will be saved as:
```
data/processed/modeling/model_table_v1_environment_with_lags.csv
```
