# STAGE 3: Frozen Analysis Protocol

## Data Partitioning & Decision Rules
- **2022–2023**: Model development & pre-training.
- **2024**: Model selection, hyperparameter tuning, sampling-ratio selection, and calibration training (Zero test leakage).
- **2025**: Retrospective out-of-time evaluation.

## Pre-Defined Statistical Outcomes
- **Primary Outcome**: Average Precision (`average_precision_score`) on the Primary Commercial Poultry Cohort.
- **Primary Operational Outcomes**: Monthly Precision@K and Recall@K for K in {10, 25, 50}.
- **Secondary Outcomes**: ROC-AUC, Brier Skill Score, Log Loss, Calibration Intercept/Slope.
- **Tie Handling Policy**: In Top-K calculations, ties in predicted probability are broken deterministically by sorting on (`y_prob` descending, `csd_uid` ascending).
