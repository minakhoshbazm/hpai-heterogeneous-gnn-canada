"""
Train Retrospective eBird Sensitivity Logistic Regression Baseline.

Purpose:
Evaluates retrospective eBird sensitivity by fitting a weighted Logistic Regression model
on the retrospective eBird-enhanced dataset (data/processed/modeling/model_table_v1_environment_with_lags_ebird_retrospective.csv).

Notice:
This is a retrospective sensitivity experiment only (using 2023 eBird Status static patterns repeated across 2022-2025).
It does NOT represent a leakage-safe real-time forecasting pipeline.

Inputs:
  data/processed/modeling/model_table_v1_environment_with_lags_ebird_retrospective.csv

Outputs:
  results/baselines/ebird_retrospective_logistic_metrics.csv
  results/baselines/ebird_retrospective_logistic_test_predictions.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

INPUT_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags_ebird_retrospective.csv'
METRICS_OUT = 'results/baselines/ebird_retrospective_logistic_metrics.csv'
PREDS_OUT = 'results/baselines/ebird_retrospective_logistic_test_predictions.csv'

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
    print("  Train Retrospective eBird Sensitivity Logistic Regression Baseline")
    print("  (Retrospective eBird sensitivity dataset - 2023 eBird Status patterns repeated 2022-2025)")
    print("==========================================================================================")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype={'csd_uid': str})

    # QA Checks
    total_rows = len(df)
    unique_csds = df['csd_uid'].nunique()
    dup_keys = df.duplicated(subset=['csd_uid', 'year', 'month']).sum()
    print(f"QA Check - Total Rows: {total_rows} (Expected: 237,538)")
    print(f"QA Check - Unique CSDs: {unique_csds} (Expected: 5,054)")
    print(f"QA Check - Duplicate Keys: {dup_keys} (Expected: 0)")
    assert total_rows == 237538
    assert unique_csds == 5054
    assert dup_keys == 0

    # Temporal split
    train_df = df[df['year'].isin([2022, 2023, 2024])].copy()
    test_df = df[df['year'] == 2025].copy()

    train_pos = int(train_df['outbreak_next_month'].sum())
    test_pos = int(test_df['outbreak_next_month'].sum())

    print(f"Temporal Split QA:")
    print(f"  Train rows: {len(train_df)} (Expected: 181,944), Positives: {train_pos} (Expected: 302)")
    print(f"  Test rows: {len(test_df)} (Expected: 55,594), Positives: {test_pos} (Expected: 79)")
    assert len(train_df) == 181944
    assert len(test_df) == 55594
    assert train_pos == 302
    assert test_pos == 79

    # Columns to exclude from model input features
    exclude_cols = {
        'csd_uid', 'csd_name', 'province_code', 'province_name',
        'year', 'month', 'outbreak_next_month',
        'ebird_status_product_year', 'ebird_retrospective_only_flag',
        'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
        'glw2015_duck_extraction_method', 'road_data_coverage'
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_features
    print(f"Selected numeric features count: {len(feature_cols)}")

    # Confirm no NaNs in selected numeric feature columns
    nan_count = df[feature_cols].isna().sum().sum()
    print(f"QA Check - Feature NaNs: {nan_count} (Expected: 0)")
    assert nan_count == 0

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df['outbreak_next_month'].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df['outbreak_next_month'].values.astype(int)

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Weighted Logistic Regression matching earlier environment + lag baseline
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
    print("Fitting weighted Logistic Regression on retrospective eBird sensitivity dataset...")
    clf.fit(X_train_scaled, y_train)

    probs = clf.predict_proba(X_test_scaled)[:, 1]

    ap = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    k_metrics = topk_metrics(y_test, probs, [50, 100, 200], test_pos)

    metrics_df = pd.DataFrame([{
        'model': 'retrospective eBird sensitivity LogisticRegression',
        'pr_auc': ap,
        'roc_auc': roc_auc,
        'precision_50': k_metrics['precision_50'],
        'recall_50': k_metrics['recall_50'],
        'precision_100': k_metrics['precision_100'],
        'recall_100': k_metrics['recall_100'],
        'precision_200': k_metrics['precision_200'],
        'recall_200': k_metrics['recall_200']
    }])

    preds_df = pd.DataFrame({
        'csd_uid': test_df['csd_uid'].values,
        'csd_name': test_df['csd_name'].values,
        'province_code': test_df['province_code'].values,
        'year': test_df['year'].values,
        'month': test_df['month'].values,
        'actual': y_test,
        'retrospective_ebird_logistic_prob': probs
    })

    # Save outputs
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_OUT), exist_ok=True)
    metrics_df.to_csv(METRICS_OUT, index=False)
    preds_df.to_csv(PREDS_OUT, index=False)

    print("\n================ RETROSPECTIVE EBIRD SENSITIVITY LOGISTIC REGRESSION METRICS ================")
    print(metrics_df.to_string(index=False))
    print("=============================================================================================")
    print(f"\nBaseline Comparison:")
    print(f"  - Non-eBird Environment + Lag Logistic Regression PR-AUC: 0.0635")
    print(f"  - Retrospective eBird Sensitivity Logistic Regression PR-AUC: {ap:.4f}")
    print("\nNOTICE: This result represents retrospective sensitivity evidence only and MUST NOT be used for real-time forecasting claims.")

if __name__ == '__main__':
    main()
