# Negative-Cohort Pruning Ablation (Rolling-Origin, Wild-Bird CSD-Month)

**Question**: How much of the low absolute PR-AUC (~0.05-0.06) on this task is caused by
*denominator dilution* from CSDs that carry no poultry, versus a genuine signal ceiling?

**Protocol**: Strict 3-fold rolling-origin temporal validation (test years 2023, 2024, 2025),
62-feature set, identical splits to every other rolling evaluation in this project.

**Leakage guard**: cohorts are defined only from covariates verified constant per CSD across
all months (`glw2020_chicken_density_mean`, `ccs_broilers_roasters_cornish_production_kg`, `ccs_turkey_production_kg`). No outbreak label and no
time-varying covariate participates in cohort definition.

---

## 1. Cohort Composition

| Cohort | CSDs | % Retained | Rows | Positives | Prevalence | Ever-Positive CSDs Kept | Lost |
|---|---|---|---|---|---|---|---|
| `full` | 5,054 | 100.0% | 237,538 | 381 | 0.001604 | 239/239 | 0 |
| `poultry_any` | 4,472 | 88.5% | 210,184 | 377 | 0.001794 | 235/239 | 4 |
| `chicken_gt0` | 4,471 | 88.5% | 210,137 | 377 | 0.001794 | 235/239 | 4 |
| `chicken_gt1` | 1,933 | 38.2% | 90,851 | 312 | 0.003434 | 180/239 | 59 |
| `chicken_gt10` | 1,005 | 19.9% | 47,235 | 248 | 0.005250 | 130/239 | 109 |

---

## 2. Two Evaluation Arms

- **`restricted`** — train on cohort, evaluate on cohort test rows only. Shows how much of the
  low PR-AUC is mechanical. **PR-AUC is not comparable across cohorts here** (the evaluation
  prevalence changes); compare **lift over prevalence** instead.
- **`full_eval`** — train on cohort, evaluate on the *entire* test set with out-of-cohort rows
  forced to the bottom of the ranking. Positives living outside the cohort are unrecoverable,
  so the cost of pruning is charged honestly. This arm shares one fixed evaluation set with
  the `full` baseline, so PR-AUC **is** directly comparable — this is the arm the bootstrap uses.

---

## 3. Pooled Results (2023-2025)

### Logistic Regression

| Cohort | Arm | Eval Rows | Pos | Prevalence | PR-AUC | Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full` | restricted | 176,890 | 191 | 0.001080 | 0.0540 | 50.0x | 0.8967 | 0.2600 | 0.0681 | 0.1500 | 0.0785 | 0.0900 | 0.0942 |
| `poultry_any` | restricted | 156,520 | 190 | 0.001214 | 0.0535 | 44.1x | 0.8857 | 0.2600 | 0.0684 | 0.1500 | 0.0789 | 0.0950 | 0.1000 |
| `chicken_gt0` | restricted | 156,485 | 190 | 0.001214 | 0.0535 | 44.1x | 0.8856 | 0.2600 | 0.0684 | 0.1500 | 0.0789 | 0.0950 | 0.1000 |
| `chicken_gt1` | restricted | 67,655 | 157 | 0.002321 | 0.0635 | 27.4x | 0.8509 | 0.2600 | 0.0828 | 0.1600 | 0.1019 | 0.0950 | 0.1210 |
| `chicken_gt10` | restricted | 35,175 | 125 | 0.003554 | 0.0492 | 13.8x | 0.8019 | 0.1800 | 0.0720 | 0.1300 | 0.1040 | 0.0850 | 0.1360 |
| `full` | full_eval | 176,890 | 191 | 0.001080 | 0.0540 | 50.0x | 0.8967 | 0.2600 | 0.0681 | 0.1500 | 0.0785 | 0.0900 | 0.0942 |
| `poultry_any` | full_eval | 176,890 | 191 | 0.001080 | 0.0533 | 49.3x | 0.8944 | 0.2600 | 0.0681 | 0.1500 | 0.0785 | 0.0950 | 0.0995 |
| `chicken_gt0` | full_eval | 176,890 | 191 | 0.001080 | 0.0533 | 49.3x | 0.8944 | 0.2600 | 0.0681 | 0.1500 | 0.0785 | 0.0950 | 0.0995 |
| `chicken_gt1` | full_eval | 176,890 | 191 | 0.001080 | 0.0525 | 48.6x | 0.8323 | 0.2600 | 0.0681 | 0.1600 | 0.0838 | 0.0950 | 0.0995 |
| `chicken_gt10` | full_eval | 176,890 | 191 | 0.001080 | 0.0327 | 30.3x | 0.7780 | 0.1800 | 0.0471 | 0.1300 | 0.0681 | 0.0850 | 0.0890 |

### XGBoost

| Cohort | Arm | Eval Rows | Pos | Prevalence | PR-AUC | Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full` | restricted | 176,890 | 191 | 0.001080 | 0.0540 | 50.0x | 0.8661 | 0.2000 | 0.0524 | 0.1100 | 0.0576 | 0.0650 | 0.0681 |
| `poultry_any` | restricted | 156,520 | 190 | 0.001214 | 0.0620 | 51.1x | 0.8803 | 0.2000 | 0.0526 | 0.1100 | 0.0579 | 0.0700 | 0.0737 |
| `chicken_gt0` | restricted | 156,485 | 190 | 0.001214 | 0.0489 | 40.3x | 0.8660 | 0.1800 | 0.0474 | 0.1000 | 0.0526 | 0.0600 | 0.0632 |
| `chicken_gt1` | restricted | 67,655 | 157 | 0.002321 | 0.0505 | 21.8x | 0.8204 | 0.2000 | 0.0637 | 0.1100 | 0.0701 | 0.0650 | 0.0828 |
| `chicken_gt10` | restricted | 35,175 | 125 | 0.003554 | 0.0957 | 26.9x | 0.8562 | 0.2000 | 0.0800 | 0.1500 | 0.1200 | 0.1100 | 0.1760 |
| `full` | full_eval | 176,890 | 191 | 0.001080 | 0.0540 | 50.0x | 0.8661 | 0.2000 | 0.0524 | 0.1100 | 0.0576 | 0.0650 | 0.0681 |
| `poultry_any` | full_eval | 176,890 | 191 | 0.001080 | 0.0617 | 57.1x | 0.8899 | 0.2000 | 0.0524 | 0.1100 | 0.0576 | 0.0700 | 0.0733 |
| `chicken_gt0` | full_eval | 176,890 | 191 | 0.001080 | 0.0487 | 45.1x | 0.8773 | 0.1800 | 0.0471 | 0.1000 | 0.0524 | 0.0600 | 0.0628 |
| `chicken_gt1` | full_eval | 176,890 | 191 | 0.001080 | 0.0417 | 38.6x | 0.8131 | 0.2000 | 0.0524 | 0.1100 | 0.0576 | 0.0650 | 0.0681 |
| `chicken_gt10` | full_eval | 176,890 | 191 | 0.001080 | 0.0631 | 58.4x | 0.7635 | 0.2000 | 0.0524 | 0.1500 | 0.0785 | 0.1100 | 0.1152 |

---

## 4. CSD-Cluster Bootstrap (1000 resamples, `full_eval` arm)

Each pruned cohort is compared against the `full` cohort on the identical pooled evaluation set.

| Model | Comparison | Observed PR-AUC Diff | Bootstrap Mean | 95% CI | Win Rate | Decision |
|---|---|---|---|---|---|---|
| Logistic Regression | poultry_any vs full | -0.0007 | -0.0016 | [-0.0055, +0.0002] | 5.1% | **COMPARABLE, NOT PROVEN BETTER** |
| Logistic Regression | chicken_gt0 vs full | -0.0007 | -0.0016 | [-0.0055, +0.0002] | 5.1% | **COMPARABLE, NOT PROVEN BETTER** |
| Logistic Regression | chicken_gt1 vs full | -0.0015 | -0.0025 | [-0.0111, +0.0028] | 17.5% | **COMPARABLE, NOT PROVEN BETTER** |
| Logistic Regression | chicken_gt10 vs full | -0.0213 | -0.0181 | [-0.0484, +0.0022] | 5.8% | **COMPARABLE, NOT PROVEN BETTER** |
| XGBoost | poultry_any vs full | +0.0077 | +0.0074 | [-0.0065, +0.0284] | 74.6% | **COMPARABLE, NOT PROVEN BETTER** |
| XGBoost | chicken_gt0 vs full | -0.0053 | -0.0039 | [-0.0128, +0.0061] | 19.9% | **COMPARABLE, NOT PROVEN BETTER** |
| XGBoost | chicken_gt1 vs full | -0.0122 | -0.0103 | [-0.0239, +0.0025] | 7.3% | **COMPARABLE, NOT PROVEN BETTER** |
| XGBoost | chicken_gt10 vs full | +0.0091 | +0.0092 | [-0.0003, +0.0229] | 96.7% | **COMPARABLE, NOT PROVEN BETTER** |

---

## 5. Reproduction Check

published rolling LR pooled PR-AUC = 0.0540; this script's `full` cohort LR = 0.0540 (delta +0.00000)

---

## 6. Per-Fold Detail

See `results/baselines/rolling_wildbird_negative_pruning_fold_metrics.csv`. Cohort/fold combinations with fewer than 10 test positives were
skipped rather than reported, consistent with the project's existing reliability threshold.

> [!IMPORTANT]
> **METHODOLOGICAL ASSURANCE**
> 1. Cohort membership derives only from static, label-independent covariates asserted
>    constant within each CSD; no outbreak label or future covariate is involved.
> 2. Training, hyperparameter selection, and imputation statistics are computed inside each
>    cohort from training rows only; no test-year information influences any fitted quantity.
> 3. The `full_eval` arm charges pruning for every positive it discards, so improvements
>    there are not an artifact of a shrinking evaluation set.
> 4. PR-AUC across `restricted` arms is deliberately NOT compared directly, since evaluation
>    prevalence differs by construction.
