# Pseudo-Absence Logistic Regression Result

## Purpose

Pseudo-absence sampling was tested because outbreak positives are rare and the training data is highly imbalanced. The goal was to see whether logistic regression improves when trained on all positive rows and a smaller, more meaningful sample of negative rows.

## Experiment

The experiment used training data from 2022-2024 and evaluated all models on the full unchanged 2025 test set. The test set was not sampled or rebalanced. The model was logistic regression using environmental and lag/history features. Three negative:positive training ratios were tested: 5:1, 10:1, and 20:1.

## Results

| Training setup | PR-AUC | ROC-AUC | Precision@50 | Recall@50 |
| -------------- | ------ | ------- | ------------ | --------- |
| Full imbalanced training | 0.0635 | 0.9003 | 0.120 | 0.0759 |
| Pseudo-absence 5:1 | 0.0213 | 0.8863 | 0.080 | 0.0506 |
| Pseudo-absence 10:1 | 0.0305 | 0.8837 | 0.100 | 0.0633 |
| Pseudo-absence 20:1 | 0.0454 | 0.8850 | 0.100 | 0.0633 |

## Interpretation

Pseudo-absence sampling did not improve logistic regression. The best pseudo-absence setting was 20:1, but it still underperformed the full imbalanced training baseline. This suggests that logistic regression benefits from seeing the full range of negative examples. Pseudo-absence sampling may still be useful for other model types, but it should not replace the current logistic regression baseline.

## Decision

For logistic regression, keep the full imbalanced training setup as the preferred baseline.
