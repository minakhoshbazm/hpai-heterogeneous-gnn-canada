#!/usr/bin/env python3
"""
scripts/generate_wildbird_lr_predictions.py

Generates wild‑bird‑enhanced Logistic Regression test predictions
and saves them to results/baselines/wildbird_lr_strict_test_predictions.csv.
This file is required for the GraphSAGE bootstrap comparison.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

# Paths
INPUT_MODEL_TABLE = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
OUT_PRED = "results/baselines/wildbird_lr_strict_test_predictions.csv"

# Exclude columns (same as other scripts)
EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
}

# Load data
df = pd.read_csv(INPUT_MODEL_TABLE)

# Feature selection (numeric, not excluded, no neighbor/ebird/gbif columns)
feature_cols = []
for c in df.columns:
    if c in EXCLUDE_EXPLICIT:
        continue
    if any(sub in c.lower() for sub in ["ebird", "gbif", "neighbor", "pressure"]):
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        feature_cols.append(c)
feature_cols = sorted(feature_cols)
assert len(feature_cols) == 62, f"Expected 62 features, got {len(feature_cols)}"

# Train/val/test splits (same as other scripts)
df_train = df[(df['year'] <= 2023) | ((df['year'] == 2024) & (df['month'] <= 9))].copy()
df_val = df[(df['year'] == 2024) & (df['month'] >= 10)].copy()
df_test = df[df['year'] == 2025].copy()

# Scale (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[feature_cols].fillna(0.0))
X_val = scaler.transform(df_val[feature_cols].fillna(0.0))
X_test = scaler.transform(df_test[feature_cols].fillna(0.0))

y_train = df_train['outbreak_next_month'].values

y_val = df_val['outbreak_next_month'].values

y_test = df_test['outbreak_next_month'].values

# Class weighting
pos = y_train.sum()
neg = len(y_train) - pos
pos_weight = neg / pos if pos > 0 else 1.0

clf = LogisticRegression(max_iter=1000, class_weight={0:1.0, 1:pos_weight}, solver='liblinear')
clf.fit(X_train, y_train)

# Validation PR‑AUC just for sanity (not used further)
val_prob = clf.predict_proba(X_val)[:, 1]
val_pr_auc = average_precision_score(y_val, val_prob)
print(f"Validation PR-AUC: {val_pr_auc:.4f}")

# Test predictions
test_prob = clf.predict_proba(X_test)[:, 1]

# Save predictions with identifiers
out_df = df_test[['csd_uid', 'year', 'month']].copy()
out_df['actual'] = y_test
out_df['strict_env_lag_logistic_prob'] = test_prob
os.makedirs(os.path.dirname(OUT_PRED), exist_ok=True)
out_df.to_csv(OUT_PRED, index=False)
print(f"Saved LR test predictions to {OUT_PRED}")
