# Negative-Cohort Pruning Result

## Purpose

Absolute PR-AUC on the wild-bird CSD-month task sits around 0.05-0.06, which reads as "low" in
isolation. This experiment tested a specific explanation for that: **denominator dilution**. The
national table evaluates 5,054 CSDs every month, including subdivisions with no poultry at all —
places no surveillance program would ever visit. If those rows were inflating the negative class
without contributing difficulty, restricting the candidate pool should raise real performance.

## Design

Cohorts were defined only from covariates verified constant per CSD across all months
(`glw2020_chicken_density_mean`, `ccs_broilers_roasters_cornish_production_kg`,
`ccs_turkey_production_kg`). Because membership is fixed before any temporal split and never
touches the label, pruning cannot leak outbreak or future information.

| Cohort | CSDs | % Retained | Prevalence | Ever-Positive CSDs Kept |
|---|---|---|---|---|
| `full` | 5,054 | 100.0% | 0.001604 | 239/239 |
| `poultry_any` | 4,472 | 88.5% | 0.001794 | 235/239 |
| `chicken_gt0` | 4,471 | 88.5% | 0.001794 | 235/239 |
| `chicken_gt1` | 1,933 | 38.2% | 0.003434 | 180/239 |
| `chicken_gt10` | 1,005 | 19.9% | 0.005250 | 130/239 |

Two arms were required, because the naive version of this test is self-confirming: restricting the
evaluation set raises prevalence, which mechanically raises PR-AUC even if the model has not
improved at all.

- **`restricted`** — train and evaluate inside the cohort. Compared via **lift over prevalence**,
  never via raw PR-AUC across cohorts.
- **`full_eval`** — train inside the cohort, evaluate on all 176,890 test rows with out-of-cohort
  rows forced to the bottom of the ranking. Positives outside the cohort become unrecoverable, so
  pruning is charged for every case it discards. Shares one fixed evaluation set with the
  baseline, so PR-AUC **is** comparable. This is the arm the bootstrap tests.

Protocol was the strict 3-fold rolling origin (test years 2023/2024/2025), 62-feature set,
identical splits to every other rolling evaluation in the project.

## Reproduction Check

The `full` cohort reproduced the published pooled rolling numbers exactly:

| Model | This experiment | Published (`rolling_wildbird_graphsage_vs_tabular_models_pooled_metrics.csv`) |
|---|---|---|
| Logistic Regression | 0.053979 | 0.053979 |
| XGBoost | 0.053955 | 0.053955 |

Any cohort difference is therefore attributable to the cohort alone, not to pipeline drift.

## Result: The Dilution Hypothesis Is Refuted

**Logistic Regression, pooled 2023-2025:**

| Cohort | `restricted` PR-AUC | `restricted` Lift | `full_eval` PR-AUC |
|---|---|---|---|
| `full` | 0.0540 | 50.0x | 0.0540 |
| `poultry_any` | 0.0535 | 44.1x | 0.0533 |
| `chicken_gt1` | **0.0635** | **27.4x** | 0.0525 |
| `chicken_gt10` | 0.0492 | 13.8x | 0.0327 |

Pruning to `chicken_gt1` *appears* to help — restricted PR-AUC rises 0.0540 to 0.0635. But lift
**falls from 50x to 27x**. The entire apparent gain is the mechanical prevalence increase; measured
against its own base rate the model performs materially *worse* on the poultry-dense subset. On the
fixed evaluation set the effect is a small loss (0.0525), and aggressive pruning is a large one
(0.0327).

**Bootstrap (1,000 CSD-cluster resamples, `full_eval` arm):** all eight comparisons returned
*comparable, not proven better*. Every 95% CI contains zero. Logistic Regression point estimates
were negative in all four cohorts.

XGBoost point estimates were non-monotonic in cohort strictness (`poultry_any` +0.0077,
`chicken_gt0` -0.0053, `chicken_gt1` -0.0122, `chicken_gt10` +0.0091). A real cohort effect would
not reverse sign twice as the filter tightens; this is fold noise. XGBoost is high-variance here —
its per-fold `full` PR-AUC ranges 0.021 / 0.195 / 0.038 — so its two positive point estimates
should not be read as support for pruning.

## Interpretation

Zero-poultry CSDs are **not** diluting noise the models struggle with. They are easy negatives the
models already rank at the bottom, and deleting them removes free correct rankings while discarding
recoverable positives. The models have effectively solved "is this a poultry region"; the residual
difficulty is "*which* poultry region, *this* month" — and that is where a PR-AUC of 0.054 comes from.

This reframes the low absolute numbers. They are not a denominator artifact. They reflect (a) a
genuine signal ceiling on monthly within-region timing, and (b) a very small positive count — 40 to
79 positives per test year, so a single outbreak moves recall by roughly 1.5%, which is why every
pairwise bootstrap in this project straddles zero.

## Decision

1. **Do not prune the candidate pool.** Keep the full national CSD table as the modeling and
   evaluation frame. The pruning hypothesis was tested and did not hold.
2. **Report lift over prevalence alongside PR-AUC** in all future reporting. A PR-AUC of 0.054
   against a base rate of 0.00108 is 50x lift; the raw number alone invites the wrong conclusion.
3. **Never report uncalibrated probabilities.** The reference LR uses
   `class_weight={0:1, 1:~850}`, which makes its outputs ranking scores, not probabilities
   (ECE 0.177). The Platt-calibrated variant is correct (Brier 0.0015, ECE 0.00034) and its
   apparently "low" outputs are the honest ~0.1% monthly risk. Ranking is identical either way.
4. **Next lever is the positive count, not the negative pool.** Coarsening the unit of analysis
   (CCS-month or CD-month) raises prevalence several-fold by aggregating positives rather than by
   deleting negatives, and would stabilise the regional splits currently flagged unreliable
   (Atlantic, N_pos = 5). That is the recommended next experiment.

## Artifacts

- `scripts/run_rolling_wildbird_negative_pruning_ablation.py`
- `results/baselines/rolling_wildbird_negative_pruning_fold_metrics.csv`
- `results/baselines/rolling_wildbird_negative_pruning_pooled_metrics.csv`
- `results/baselines/rolling_wildbird_negative_pruning_bootstrap.csv`
- `results/baselines/rolling_wildbird_negative_pruning_predictions.csv`
- `results/baselines/rolling_wildbird_negative_pruning_summary.md`
- `results/data_qa/rolling_wildbird_negative_pruning_cohort_audit.md`
