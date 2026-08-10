# Negative-Cohort Pruning: Cohort Definition Audit

**Input table**: `data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv`
**Total CSDs**: 5,054
**Total rows**: 237,538
**Ever-positive CSDs**: 239

## 1. Leakage Guard: Cohort Covariate Staticness

| Column | Max Distinct Values per CSD | Result |
|---|---|---|
| `glw2020_chicken_density_mean` | 1 | PASS |
| `ccs_broilers_roasters_cornish_production_kg` | 1 | PASS |
| `ccs_turkey_production_kg` | 1 | PASS |

A value of 1 confirms the covariate is constant across all months for every CSD, so cohort
membership is fixed before any temporal split and cannot encode label or future information.

## 2. Cohort Composition

| Cohort | CSDs | % Retained | Rows | Positives | Prevalence | Ever-Positive CSDs Kept | Lost |
|---|---|---|---|---|---|---|---|
| `full` | 5,054 | 100.0% | 237,538 | 381 | 0.001604 | 239/239 | 0 |
| `poultry_any` | 4,472 | 88.5% | 210,184 | 377 | 0.001794 | 235/239 | 4 |
| `chicken_gt0` | 4,471 | 88.5% | 210,137 | 377 | 0.001794 | 235/239 | 4 |
| `chicken_gt1` | 1,933 | 38.2% | 90,851 | 312 | 0.003434 | 180/239 | 59 |
| `chicken_gt10` | 1,005 | 19.9% | 47,235 | 248 | 0.005250 | 130/239 | 109 |

## 3. Feature Set

62 features asserted on every cohort and fold (identical exclusion set to
`run_rolling_wildbird_tree_baselines_vs_graphsage.py`).

## 4. Reproduction Check

published rolling LR pooled PR-AUC = 0.0540; this script's `full` cohort LR = 0.0540 (delta +0.00000)
