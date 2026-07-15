# Pseudo-Absence Sampling Plan

## Purpose

Outbreak positive cases (occurrences of HPAI) are extremely rare in our dataset (positive rate is roughly 0.16%). If we train models on all non-outbreak data points, the overwhelming imbalance causes the model to predict the majority class (non-outbreaks) almost exclusively. To address this, pseudo-absence sampling will keep all positive outbreak rows and sample a smaller, more informative, and representative subset of negative rows (pseudo-absences) for training.

## Key Rule

Pseudo-absence sampling is strictly used for training only. The 2025 test set must remain untouched, fully imbalanced, and represent the actual historical distribution of outbreaks. This ensures that the model evaluation metrics reflect realistic predictive capability in deployment conditions.

## Sampling Criteria

Candidate negative training rows (pseudo-absences) will be selected using the following criteria:
1. **Target Status**: The CSD-month must have no outbreak in the target month (`outbreak_next_month == 0`).
2. **Seasonality Matching**: Negatives should be selected from the same month of the year as the positive outbreaks (e.g., matching April positives with April negatives) to preserve seasonal patterns.
3. **Regional Structure Matching**: Negatives should be selected from the same province as the positive examples where possible, preserving provincial risk and administrative structures.
4. **Plausible Exposure**: CSDs with no recorded poultry production or poultry-related proxy values are usually less informative for poultry outbreak prediction, although this may reflect incomplete proxy coverage rather than true absence of poultry. We will prioritize sampling from negative rows that have poultry production, poultry farms, or chicken/duck density proxy values greater than zero.
5. **Exposure-weighted Probability**: The probability of sampling a negative node will be weighted proportionally to its poultry exposure density, ensuring that negative examples are selected from regions where an outbreak was highly plausible but did not occur.

## Candidate Ratios

We will test three negative:positive sampling ratios in training:
- **5:1** (5 negative rows for every 1 positive row)
- **10:1** (Recommended setting - balances exposure diversity and class ratio)
- **20:1** (Sensitivity check)

## Leakage Safety

Sampling must be independent of future information. The sampling algorithm will only use static covariates (poultry density, area), current location (CSD, province), month indices, and past outbreak history features. It will never look ahead to future target labels or use any information from the 2025 test set when generating the training samples.

## Planned Experiments

We will run experiments comparing:
1. Current baselines trained on the fully imbalanced training set.
2. Logistic Regression (with lag/history features) trained on the pseudo-absence sampled dataset.
3. GraphSAGE Spatial GNN model trained on the pseudo-absence sampled dataset.

All models will be evaluated and compared on the same fully imbalanced 2025 test set.

## Expected Outcome

By focusing training on plausible negative regions, the model's capacity to identify subtle risk boundaries should improve. This is expected to yield higher PR-AUC and more reliable risk rankings on the 2025 test set without compromising evaluation realism.
