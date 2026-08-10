# Stratified Pseudo-Absence Sampling Audit

**Source table**: model_table_v1_environment_with_lags_wildbird.csv

**Training years**: [2022, 2023, 2024]

**Total training positives**: 302

**Total training negatives available**: 181,642

**Note**: Negatives are sampled proportionally to province historical outbreak count; no leakage from 2025 test set.

---

### Ratio 3:1
- Total rows: 1,208
- Positives: 302
- Negatives: 906
- File: `pseudo_absence_stratified_ratio_3.csv`

### Ratio 5:1
- Total rows: 1,812
- Positives: 302
- Negatives: 1510
- File: `pseudo_absence_stratified_ratio_5.csv`

### Ratio 10:1
- Total rows: 3,322
- Positives: 302
- Negatives: 3020
- File: `pseudo_absence_stratified_ratio_10.csv`

