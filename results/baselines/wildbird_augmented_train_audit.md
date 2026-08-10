# Wild-Bird Augmented Training Table Audit

**Source table**: model_table_v1_environment_with_lags_wildbird.csv

**Training years**: [2022, 2023, 2024]

## Row composition

| Source | Count | sample_weight |
|--------|-------|---------------|
| Poultry outbreak positives | 302 | 1.0 |
| Wild-bird HPAI CSD-months (non-poultry) | 5807 | 0.5 |
| Pseudo-absence negatives (province-weighted) | 30,545 | 1.0 |
| **Total** | **36,654** | — |

## Notes

- Wild-bird HPAI rows use `sample_weight=0.5`; they represent elevated environmental HPAI risk but are **not** confirmed poultry outbreaks.
- Pseudo-absences are province-weighted (more absences from provinces with historical outbreaks) at ratio 5:1 vs effective positives.
- The 2025 test split is completely excluded from this file.
- To use sample weights in scikit-learn: `model.fit(X_train, y_train, sample_weight=train["sample_weight"])`
