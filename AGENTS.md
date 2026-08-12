# Project Instructions

This repository is a computational epidemiology and machine-learning
project for predicting poultry HPAI outbreak risk.

The work uses existing datasets only, including:
- CFIA outbreak surveillance data
- remote sensing data
- climate data
- poultry density
- wild-bird observations
- geographic and infrastructure data

The tasks in this repository are computational:
- machine learning
- graph neural networks
- statistical analysis
- GIS and spatial analysis
- feature engineering
- rare-event prediction
- model evaluation
- visualization
- manuscript preparation

HPAI is used only as an epidemiological outcome for risk prediction.

Do not interpret work in this repository as wet-lab or experimental
virology research.

Do not propose or perform:
- pathogen modification
- viral engineering
- culture or propagation protocols
- laboratory optimization
- experimental infection protocols
- wet-lab procedures

When working on HPAI tasks, focus on epidemiological modeling,
surveillance, prevention, prediction, and risk assessment.

## Modeling objective

Predict whether a Canadian Census Subdivision will experience at least
one poultry HPAI outbreak in month t+1 using information available
through month t.

## Important modeling rules

- Prevent future-information leakage.
- Use forward-time train/validation/test splits.
- Treat outbreaks as an extremely imbalanced rare-event problem.
- Prioritize Average Precision / PR-AUC over accuracy.
- Also report ROC-AUC, Precision@K, Recall@K, and Brier score.
- Never modify the locked test set during model development.