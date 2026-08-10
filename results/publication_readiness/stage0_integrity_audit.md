# STAGE 0: Reproducibility & Data Integrity Audit

## Dataset Fingerprint Table

| Split | row_count | pos_count | unique_csds | min_month | max_month | prevalence | id_checksum | label_checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full Dataset (2022-2025) | 237538 | 381 | 5054 | 2022-01 | 2025-12 | 0.0016039538936927986 | 1a793d5090 | 68bc2c07f1 |
| Train-Tune (2022-2023) | 121296 | 262 | 5054 | 2022-01 | 2023-12 | 0.002160005276348767 | b07a1d2c9d | ce77738cbd |
| Val-Tune (2024) | 60648 | 40 | 5054 | 2024-01 | 2024-12 | 0.0006595435958316845 | 70b169e471 | f175e87178 |
| Full Train (2022-2024) | 181944 | 302 | 5054 | 2022-01 | 2024-12 | 0.0016598513828430725 | 12dc31702e | 800f65e301 |
| Test 2025 Retrospective | 55594 | 79 | 5054 | 2025-01 | 2025-11 | 0.0014210166564737202 | deef40d862 | 52f97eedee |


## Verification Checks
- **Unit of Analysis**: Census Subdivision (CSD) x Calendar Month (t).
- **Target Definition**: `outbreak_next_month` in {0, 1} (1-month-ahead forecast $t+1$).
- **Duplicate Check**: Exactly 0 duplicate CSD-year-month rows verified across all splits.
- **Feature Exclusion**: Identifiers `csd_uid`, `csd_name`, `province_code`, `ccs_uid`, `year`, `month`, `outbreak_next_month`, `outbreak_current_month`, `wildbird_hpai_present_next_month`, and data quality flags are strictly excluded from features.
