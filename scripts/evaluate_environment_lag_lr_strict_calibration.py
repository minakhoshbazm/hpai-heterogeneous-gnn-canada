"""
Evaluate Probability Calibration for Strict Environment + Lag Logistic Regression.

Purpose:
Fits probability calibration (Platt scaling / Sigmoid) on a dedicated calibration period:
  - Base-model training: 2022-01 through 2024-09 (166,782 rows, 272 positives)
  - Calibration period: 2024-10 through 2024-12 (15,162 rows, 30 positives)
  - Evaluation test period: untouched 2025 rows (55,594 rows, 79 positives)

Features:
  Uses exactly the 50 strict ENV_LAG_FEATURES (43 environment features + 7 approved lag/history features).
  Strictly excludes ccs_uid, all uid fields, raw outbreak_current_month, identifiers, target, and metadata.

Inputs:
  data/processed/modeling/model_table_v1_environment_with_lags.csv

Outputs:
  results/baselines/environment_lag_lr_strict_calibration_metrics.csv
  results/baselines/environment_lag_lr_strict_calibration_predictions.csv
  results/baselines/environment_lag_lr_strict_calibration_bins.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

INPUT_PATH = "data/processed/modeling/model_table_v1_environment_with_lags.csv"
METRICS_OUT = "results/baselines/environment_lag_lr_strict_calibration_metrics.csv"
PREDS_OUT = "results/baselines/environment_lag_lr_strict_calibration_predictions.csv"
BINS_OUT = "results/baselines/environment_lag_lr_strict_calibration_bins.csv"

EXCLUDE_EXPLICIT_ENV_LAG = {
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

def compute_ece(y_true, probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            avg_prob = np.mean(probs[in_bin])
            avg_actual = np.mean(y_true[in_bin])
            abs_diff = np.abs(avg_prob - avg_actual)
            ece += (bin_count / len(y_true)) * abs_diff
            bin_details.append({
                'bin_index': i + 1,
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'count': int(bin_count),
                'mean_predicted_prob': avg_prob,
                'observed_outbreak_rate': avg_actual,
                'abs_error': abs_diff
            })
        else:
            bin_details.append({
                'bin_index': i + 1,
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'count': 0,
                'mean_predicted_prob': np.nan,
                'observed_outbreak_rate': np.nan,
                'abs_error': np.nan
            })
            
    return ece, pd.DataFrame(bin_details)

def main():
    print("==========================================================================================")
    print("  Probability Calibration Evaluation for Strict Environment + Lag Logistic Regression")
    print("==========================================================================================")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype={'csd_uid': str})
    print(f"Loaded dataset with shape: {df.shape}")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_EXPLICIT_ENV_LAG and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = sorted(feature_cols)

    # ------------------ STRICT FEATURE AUDIT ASSERTS ------------------
    print("\nExecuting Strict Feature Audit Assertions...")
    print(f"Selected feature count: {len(feature_cols)} (Expected: 50)")
    print("Features selected:", feature_cols)

    assert len(feature_cols) == 50, f"ERROR: Feature count is {len(feature_cols)}, expected exactly 50!"
    assert 'ccs_uid' not in feature_cols, "ERROR: ccs_uid is present in feature list!"
    uid_features = [c for c in feature_cols if 'uid' in c.lower()]
    assert len(uid_features) == 0, f"ERROR: Features containing 'uid' detected: {uid_features}"
    assert 'outbreak_current_month' not in feature_cols, "ERROR: Raw outbreak_current_month detected in feature list!"

    for lag in APPROVED_LAG_FEATURES:
        assert lag in feature_cols, f"ERROR: Approved lag feature {lag} missing from feature list!"

    print("SUCCESS: Feature list strictly audited (50 features: 43 env + 7 lags, ccs_uid & outbreak_current_month excluded).")

    # Define splits
    train_df = df[df['year'].isin([2022, 2023]) | ((df['year'] == 2024) & (df['month'] <= 9))].copy()
    calib_df = df[(df['year'] == 2024) & (df['month'] >= 10)].copy()
    test_df = df[df['year'] == 2025].copy()

    train_rows, calib_rows, test_rows = len(train_df), len(calib_df), len(test_df)
    train_pos = int(train_df['outbreak_next_month'].sum())
    calib_pos = int(calib_df['outbreak_next_month'].sum())
    test_pos = int(test_df['outbreak_next_month'].sum())

    print(f"\nSplit Summary:")
    print(f"  Base Model Train (2022-01 to 2024-09): {train_rows:,} rows | {train_pos} positives ({train_pos/train_rows*100:.4f}%)")
    print(f"  Calibration      (2024-10 to 2024-12): {calib_rows:,} rows | {calib_pos} positives ({calib_pos/calib_rows*100:.4f}%)")
    print(f"  Untouched Test   (2025 All Months)   : {test_rows:,} rows | {test_pos} positives ({test_pos/test_rows*100:.4f}%)")

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df['outbreak_next_month'].values.astype(int)

    X_calib = calib_df[feature_cols].values.astype(np.float64)
    y_calib = calib_df['outbreak_next_month'].values.astype(int)

    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df['outbreak_next_month'].values.astype(int)

    # 1. Fit StandardScaler strictly on Base Model Train period
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)

    # 2. Fit Base Weighted Logistic Regression model
    print("\n1. Training Base Weighted Logistic Regression on 2022-01 to 2024-09...")
    base_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    base_clf.fit(X_train_scaled, y_train)

    # 3. Fit Platt / Sigmoid Calibration strictly on Calibration period using raw log-odds (decision_function)
    print("2. Fitting Platt (Sigmoid) Calibrator on 2024-10 to 2024-12 Calibration Period...")
    train_decisions = base_clf.decision_function(X_calib_scaled).reshape(-1, 1)
    
    platt_calibrator = LogisticRegression(random_state=42)
    platt_calibrator.fit(train_decisions, y_calib)

    # 4. Predict on Untouched 2025 Test set
    print("3. Evaluating Uncalibrated and Calibrated Models on untouched 2025 Test set...")
    uncal_probs = base_clf.predict_proba(X_test_scaled)[:, 1]
    
    test_decisions = base_clf.decision_function(X_test_scaled).reshape(-1, 1)
    cal_probs = platt_calibrator.predict_proba(test_decisions)[:, 1]

    # Evaluate metrics for both models
    uncal_ap = average_precision_score(y_test, uncal_probs)
    uncal_roc = roc_auc_score(y_test, uncal_probs)
    uncal_brier = brier_score_loss(y_test, uncal_probs)
    uncal_ece, uncal_bins_df = compute_ece(y_test, uncal_probs, n_bins=10)
    uncal_k = topk_metrics(y_test, uncal_probs, [50, 100, 200], test_pos)

    cal_ap = average_precision_score(y_test, cal_probs)
    cal_roc = roc_auc_score(y_test, cal_probs)
    cal_brier = brier_score_loss(y_test, cal_probs)
    cal_ece, cal_bins_df = compute_ece(y_test, cal_probs, n_bins=10)
    cal_k = topk_metrics(y_test, cal_probs, [50, 100, 200], test_pos)

    metrics_list = [
        {
            'model_variant': 'Uncalibrated LogisticRegression',
            'pr_auc': uncal_ap,
            'roc_auc': uncal_roc,
            'brier_score': uncal_brier,
            'expected_calibration_error': uncal_ece,
            'precision_50': uncal_k['precision_50'],
            'recall_50': uncal_k['recall_50'],
            'precision_100': uncal_k['precision_100'],
            'recall_100': uncal_k['recall_100'],
            'precision_200': uncal_k['precision_200'],
            'recall_200': uncal_k['recall_200']
        },
        {
            'model_variant': 'Platt Calibrated LogisticRegression',
            'pr_auc': cal_ap,
            'roc_auc': cal_roc,
            'brier_score': cal_brier,
            'expected_calibration_error': cal_ece,
            'precision_50': cal_k['precision_50'],
            'recall_50': cal_k['recall_50'],
            'precision_100': cal_k['precision_100'],
            'recall_100': cal_k['recall_100'],
            'precision_200': cal_k['precision_200'],
            'recall_200': cal_k['recall_200']
        }
    ]

    metrics_df = pd.DataFrame(metrics_list)

    preds_df = pd.DataFrame({
        'csd_uid': test_df['csd_uid'].values,
        'csd_name': test_df['csd_name'].values,
        'province_code': test_df['province_code'].values,
        'year': test_df['year'].values,
        'month': test_df['month'].values,
        'actual': y_test,
        'uncalibrated_probability': uncal_probs,
        'platt_calibrated_probability': cal_probs
    })

    uncal_bins_df['model_variant'] = 'Uncalibrated LogisticRegression'
    cal_bins_df['model_variant'] = 'Platt Calibrated LogisticRegression'
    combined_bins_df = pd.concat([uncal_bins_df, cal_bins_df], ignore_index=True)

    # Save outputs
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(BINS_OUT), exist_ok=True)

    metrics_df.to_csv(METRICS_OUT, index=False)
    preds_df.to_csv(PREDS_OUT, index=False)
    combined_bins_df.to_csv(BINS_OUT, index=False)

    print("\n==========================================================================================")
    print("         2025 UNTOUCHED TEST SET METRICS: UNCALIBRATED VS PLATT CALIBRATED")
    print("==========================================================================================")
    print(metrics_df.to_string(index=False))

    print("\n------------------------------------------------------------------------------------------")
    print("                       PLATT CALIBRATION RISK BINS SUMMARY (2025)")
    print("------------------------------------------------------------------------------------------")
    print(cal_bins_df[['bin_index', 'bin_lower', 'bin_upper', 'count', 'mean_predicted_prob', 'observed_outbreak_rate', 'abs_error']].to_string(index=False))

    print("\nCALIBRATION INTERPRETATION NOTE:")
    print("  - Probability calibration (Platt scaling) aligns output risk scores directly with empirical outbreak probabilities.")
    print(f"  - Brier Score improved from {uncal_brier:.6f} (uncalibrated) to {cal_brier:.6f} (calibrated).")
    print(f"  - Expected Calibration Error (ECE) reduced from {uncal_ece:.6f} to {cal_ece:.6f}.")
    print("  - Calibration does NOT significantly change PR-AUC (Uncalibrated: {:.6f} vs Calibrated: {:.6f}) because PR-AUC measures rank-ordering performance, which is preserved under monotonic calibration transformations.".format(uncal_ap, cal_ap))

    print(f"\nSaved metrics to: {METRICS_OUT}")
    print(f"Saved predictions to: {PREDS_OUT}")
    print(f"Saved calibration bins table to: {BINS_OUT}")

if __name__ == '__main__':
    main()
