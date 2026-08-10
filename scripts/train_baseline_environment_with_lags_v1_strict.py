"""
Train Corrected Strict Environment + Lag Logistic Regression National Baseline.

Purpose:
Fits weighted Logistic Regression on the national temporal split (2022-2024 Train, 2025 Test)
using strictly ENV_LAG_FEATURES (50 features total: 43 environment + 7 lag/history features).
Explicitly excludes ccs_uid, any feature containing uid, and raw outbreak_current_month.

Inputs:
  data/processed/modeling/model_table_v1_environment_with_lags.csv

Outputs:
  results/baselines/environment_with_lags_v1_strict_metrics.csv
  results/baselines/environment_with_lags_v1_strict_test_predictions.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

INPUT_PATH = "data/processed/modeling/model_table_v1_environment_with_lags.csv"
METRICS_OUT = "results/baselines/environment_with_lags_v1_strict_metrics.csv"
PREDS_OUT = "results/baselines/environment_with_lags_v1_strict_test_predictions.csv"

EXCLUDE_EXPLICIT = {
    'csd_uid', 'csd_name', 'province_code', 'province_name', 'ccs_uid',
    'year', 'month', 'outbreak_next_month',
    'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
    'glw2015_duck_extraction_method', 'road_data_coverage',
    'outbreak_current_month'
}

APPROVED_LAG_FEATURES = [
    'same_csd_outbreak_current_month',
    'same_csd_outbreak_prev_1_month',
    'same_csd_outbreak_count_past_3_months',
    'same_csd_outbreak_count_past_6_months',
    'same_csd_outbreak_count_past_12_months',
    'province_outbreak_count_current_month',
    'province_outbreak_count_past_3_months'
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
    print("  Train Strict Environment + Lag Logistic Regression National Baseline")
    print("==========================================================================================")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype={'csd_uid': str})
    print(f"Loaded dataset with shape: {df.shape}")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_EXPLICIT and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = sorted(feature_cols)

    # ------------------ STRICT ASSERTIONS ------------------
    print("\nExecuting Strict Feature Audit Assertions for National Env + Lag LR...")
    print(f"Selected feature count: {len(feature_cols)} (Expected: 50)")
    print("Features selected:", feature_cols)

    assert len(feature_cols) == 50, f"ERROR: Feature count is {len(feature_cols)}, expected exactly 50!"
    assert 'ccs_uid' not in feature_cols, "ERROR: ccs_uid is present in feature list!"
    uid_features = [c for c in feature_cols if 'uid' in c.lower()]
    assert len(uid_features) == 0, f"ERROR: Features containing 'uid' detected: {uid_features}"
    assert 'outbreak_current_month' not in feature_cols, "ERROR: Raw outbreak_current_month detected in feature list!"

    for lag in APPROVED_LAG_FEATURES:
        assert lag in feature_cols, f"ERROR: Approved lag feature {lag} missing from feature list!"

    print("SUCCESS: National Environment + Lag LR feature list strictly audited (50 features: 43 env + 7 lags).")

    # Temporal split
    train_df = df[df['year'].isin([2022, 2023, 2024])].copy()
    test_df = df[df['year'] == 2025].copy()

    train_rows = len(train_df)
    test_rows = len(test_df)
    train_pos = int(train_df['outbreak_next_month'].sum())
    test_pos = int(test_df['outbreak_next_month'].sum())

    print(f"\nDataset Split Summary:")
    print(f"  Train split: {train_rows:,} rows, {train_pos} positives ({train_pos / train_rows * 100:.4f}%)")
    print(f"  Test split:  {test_rows:,} rows, {test_pos} positives ({test_pos / test_rows * 100:.4f}%)")

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df['outbreak_next_month'].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df['outbreak_next_month'].values.astype(int)

    # Scale features using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    probs = clf.predict_proba(X_test_scaled)[:, 1]

    ap = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    k_metrics = topk_metrics(y_test, probs, [50, 100, 200], test_pos)

    metrics_df = pd.DataFrame([{
        'model': 'LogisticRegression (Strict Env + Lag 50-feat)',
        'pr_auc': ap,
        'roc_auc': roc_auc,
        'precision_50': k_metrics['precision_50'],
        'recall_50': k_metrics['recall_50'],
        'precision_100': k_metrics['precision_100'],
        'recall_100': k_metrics['recall_100'],
        'precision_200': k_metrics['precision_200'],
        'recall_200': k_metrics['recall_200'],
        'feature_count': len(feature_cols)
    }])

    preds_df = pd.DataFrame({
        'csd_uid': test_df['csd_uid'].values,
        'csd_name': test_df['csd_name'].values,
        'province_code': test_df['province_code'].values,
        'year': test_df['year'].values,
        'month': test_df['month'].values,
        'actual': y_test,
        'strict_env_lag_logistic_prob': probs
    })

    # Save outputs
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_OUT), exist_ok=True)
    metrics_df.to_csv(METRICS_OUT, index=False)
    preds_df.to_csv(PREDS_OUT, index=False)

    print("\n================ STRICT NATIONAL ENV + LAG LOGISTIC REGRESSION METRICS ================")
    print(metrics_df.to_string(index=False))
    print("=======================================================================================")

    print(f"\nSaved strict national metrics to: {METRICS_OUT}")
    print(f"Saved strict national predictions to: {PREDS_OUT}")

if __name__ == '__main__':
    main()
