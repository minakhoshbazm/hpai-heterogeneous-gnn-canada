#!/usr/bin/env python3
"""
scripts/ablate_data_quality_features_strict_lr.py

Part 1: Data-Quality Artifact Feature Ablation for Strict Logistic Regression.

Purpose:
Evaluates whether removing 9 data-quality / extraction metadata artifact features
from the strict 50-feature set impacts Logistic Regression baseline performance on the 2025 test set.

Artifact features removed (9 total):
- climate_fallback_distance_km
- climate_missing_flag
- dw_image_count
- dw_missing_flag
- dw_missing_flag_final
- dw_missing_flag_original
- glw2015_duck_valid_pixel_count
- glw2020_chicken_valid_pixel_count
- small_area_flag

Inputs:
  data/processed/modeling/model_table_v1_environment_with_lags.csv

Outputs:
  results/baselines/strict_lr_no_data_quality_artifacts_metrics.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

INPUT_PATH = "data/processed/modeling/model_table_v1_environment_with_lags.csv"
METRICS_OUT = "results/baselines/strict_lr_no_data_quality_artifacts_metrics.csv"

EXCLUDE_EXPLICIT = {
    'csd_uid', 'csd_name', 'province_code', 'province_name', 'ccs_uid',
    'year', 'month', 'outbreak_next_month',
    'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
    'glw2015_duck_extraction_method', 'road_data_coverage',
    'outbreak_current_month'
}

DATA_QUALITY_ARTIFACT_FEATURES = [
    'climate_fallback_distance_km',
    'climate_missing_flag',
    'dw_image_count',
    'dw_missing_flag',
    'dw_missing_flag_final',
    'dw_missing_flag_original',
    'glw2015_duck_valid_pixel_count',
    'glw2020_chicken_valid_pixel_count',
    'small_area_flag'
]

def topk_metrics(y_true, probs, k_list, total_positives):
    sorted_idx = np.argsort(probs)[::-1]
    sorted_y = y_true[sorted_idx]
    results = {}
    for k in k_list:
        top_k = sorted_y[:k]
        pos_in_top = int(top_k.sum())
        precision = pos_in_top / k
        recall = pos_in_top / total_positives if total_positives > 0 else 0.0
        results[f'precision_{k}'] = precision
        results[f'recall_{k}'] = recall
    return results

def main():
    print("==========================================================================================")
    print("      PART 1: DATA-QUALITY ARTIFACT FEATURE ABLATION (STRICT LOGISTIC REGRESSION)")
    print("==========================================================================================")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype={'csd_uid': str})
    print(f"Loaded dataset with shape: {df.shape}")

    # Full 50-feature set
    feature_cols_50 = sorted([c for c in df.columns if c not in EXCLUDE_EXPLICIT and pd.api.types.is_numeric_dtype(df[c])])
    assert len(feature_cols_50) == 50, f"Expected 50 features, got {len(feature_cols_50)}"

    # Ablated 41-feature set
    feature_cols_41 = sorted([c for c in feature_cols_50 if c not in DATA_QUALITY_ARTIFACT_FEATURES])
    assert len(feature_cols_41) == 41, f"Expected 41 features, got {len(feature_cols_41)}"

    print(f"\nOriginal feature count: {len(feature_cols_50)}")
    print(f"Removed artifact features ({len(DATA_QUALITY_ARTIFACT_FEATURES)}):", DATA_QUALITY_ARTIFACT_FEATURES)
    print(f"Ablated feature count:  {len(feature_cols_41)}")

    # Temporal split
    train_df = df[df['year'].isin([2022, 2023, 2024])].copy()
    test_df = df[df['year'] == 2025].copy()

    train_rows = len(train_df)
    test_rows = len(test_df)
    train_pos = int(train_df['outbreak_next_month'].sum())
    test_pos = int(test_df['outbreak_next_month'].sum())

    print(f"\nDataset Split Summary:")
    print(f"  Train split (2022-2024): {train_rows:,} rows, {train_pos} positives")
    print(f"  Test split (2025):       {test_rows:,} rows, {test_pos} positives")

    y_train = train_df['outbreak_next_month'].values.astype(int)
    y_test = test_df['outbreak_next_month'].values.astype(int)

    # 1. Fit Original 50-feature Model
    scaler_50 = StandardScaler()
    X_train_50 = scaler_50.fit_transform(train_df[feature_cols_50].values.astype(np.float64))
    X_test_50 = scaler_50.transform(test_df[feature_cols_50].values.astype(np.float64))

    clf_50 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf_50.fit(X_train_50, y_train)
    probs_50 = clf_50.predict_proba(X_test_50)[:, 1]

    ap_50 = average_precision_score(y_test, probs_50)
    roc_50 = roc_auc_score(y_test, probs_50)
    k_50 = topk_metrics(y_test, probs_50, [50, 100, 200], test_pos)

    # 2. Fit Ablated 41-feature Model
    scaler_41 = StandardScaler()
    X_train_41 = scaler_41.fit_transform(train_df[feature_cols_41].values.astype(np.float64))
    X_test_41 = scaler_41.transform(test_df[feature_cols_41].values.astype(np.float64))

    clf_41 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf_41.fit(X_train_41, y_train)
    probs_41 = clf_41.predict_proba(X_test_41)[:, 1]

    ap_41 = average_precision_score(y_test, probs_41)
    roc_41 = roc_auc_score(y_test, probs_41)
    k_41 = topk_metrics(y_test, probs_41, [50, 100, 200], test_pos)

    # Summary table
    records = [
        {
            'variant': 'Original Strict LR (50 features)',
            'feature_count': 50,
            'pr_auc': round(ap_50, 6),
            'pr_auc_diff': 0.0,
            'roc_auc': round(roc_50, 6),
            'roc_auc_diff': 0.0,
            'precision_50': round(k_50['precision_50'], 4),
            'recall_50': round(k_50['recall_50'], 4),
            'precision_100': round(k_50['precision_100'], 4),
            'recall_100': round(k_50['recall_100'], 4),
            'precision_200': round(k_50['precision_200'], 4),
            'recall_200': round(k_50['recall_200'], 4)
        },
        {
            'variant': 'Ablated Strict LR (41 features - No Artifacts)',
            'feature_count': 41,
            'pr_auc': round(ap_41, 6),
            'pr_auc_diff': round(ap_41 - ap_50, 6),
            'roc_auc': round(roc_41, 6),
            'roc_auc_diff': round(roc_41 - roc_50, 6),
            'precision_50': round(k_41['precision_50'], 4),
            'recall_50': round(k_41['recall_50'], 4),
            'precision_100': round(k_41['precision_100'], 4),
            'recall_100': round(k_41['recall_100'], 4),
            'precision_200': round(k_41['precision_200'], 4),
            'recall_200': round(k_41['recall_200'], 4)
        }
    ]

    metrics_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    metrics_df.to_csv(METRICS_OUT, index=False)

    print("\n==========================================================================================")
    print("         DATA-QUALITY ARTIFACT ABLATION RESULTS SUMMARY")
    print("==========================================================================================")
    print(metrics_df[['variant', 'feature_count', 'pr_auc', 'pr_auc_diff', 'roc_auc', 'roc_auc_diff', 'precision_50', 'recall_50']].to_string(index=False))
    print("==========================================================================================")

    print(f"\nSaved ablation metrics to: {METRICS_OUT}")

if __name__ == '__main__':
    main()
