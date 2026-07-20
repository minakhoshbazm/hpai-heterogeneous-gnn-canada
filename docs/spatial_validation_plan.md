# Spatial Validation Plan

## Purpose

The current evaluation strategy is temporal: models are trained on earlier years (2022-2024) and tested on 2025. While this successfully tests future temporal prediction, it does not fully evaluate whether the model generalizes to unseen geographical regions. The purpose of this spatial validation plan is to define how we will test our models' capacity to predict outbreak risk in held-out provinces or regional groups.

## Why Spatial Validation Matters

Evaluating models across geographical boundaries is crucial for several reasons:
- **Spatial Clustering**: HPAI risk factors, such as poultry density, wild bird routes, and climate conditions, are strongly clustered in specific regions.
- **Overfitting to Regions**: Models may memorize region-specific coordinates or localized proxy patterns instead of learning generalizable relationships.
- **Geographic Generalization**: A model showing high predictive power in known areas might collapse in performance when deployed to a province that was entirely absent from the training set.
- **Scientific Validity**: Robust spatial validation is required before claiming that our GNN or tabular baselines can generalize geographically.

## Candidate Designs

1. **Leave-one-province-out validation**:
   - Hold out one province at a time as a test set.
   - Train the model on the remaining provinces.
   - Evaluate predictive performance on the held-out province.
2. **Regional block validation**:
   - Group provinces into larger geographic zones (e.g., Atlantic Canada, Quebec/Ontario, Prairies, British Columbia).
   - Hold out one regional block at a time to evaluate spatial generalization on a larger, more stable test distribution.
3. **Spatial-temporal validation**:
   - Train on historical data (e.g., 2022-2024) excluding a specific region, then test on that held-out region in a future year (e.g., 2025).
   - This provides the strictest and most realistic evaluation of real-world generalization.

## Important Challenge

- **Class Imbalance and Sparsity**: HPAI outbreaks are highly rare events.
- **Sparsity in Held-Out Regions**: Some provinces or regional blocks may contain very few (or zero) positive cases in a given year. Evaluating metrics like PR-AUC on regions with zero positive targets is mathematically impossible or highly unstable.
- **Pre-analysis Requirement**: Before selecting the final spatial split structure, we must inspect outbreak counts by province and year to ensure sufficient positive counts exist in each validation split.

## Recommended Next Step

1. **Outbreak Count Audit**: Write a script to count positive outbreak records (`outbreak_next_month == 1`) grouped by province and year.
2. **Group Selection**: Identify which provinces or regional blocks have enough positive cases to support stable validation metrics.
3. **Framework Choice**: Implement either leave-one-province-out or regional block validation based on the audit.
4. **Execution**: Evaluate the simplest baseline model (logistic regression) first, then compare it against the GNN graph models.

## Current Decision

Temporal validation (train on 2022-2024, test on 2025) remains the primary evaluation strategy for the current modeling stage. Spatial validation will be implemented as a robustness check during the final reporting phase.
