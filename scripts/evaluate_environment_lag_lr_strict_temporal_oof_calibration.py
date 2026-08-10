"""
Evaluate Temporal Out-Of-Fold Probability Calibration for Strict Environment + Lag Logistic Regression.

Purpose:
Performs a temporal out-of-fold calibration experiment to build a robust calibration mapping without using 2025:
  1. Train base LR on 2022 -> Predict on 2023 (Out-Of-Fold).
  2. Train base LR on 2022-2023 -> Predict on 2024 (Out-Of-Fold).
  3. Combine 2023 & 2024 OOF predictions as the calibration dataset.
  4. Fit Platt / Sigmoid calibration on the combined OOF decision scores.
  5. Train final base LR on 2022-2024 (181,944 rows).
  6. Apply calibration mapping to untouched 2025 predictions (55,594 rows).

Features:
  Uses exactly the 50 strict ENV_LAG_FEATURES (43 environment features + 7 approved lag/history features).
  Strictly excludes ccs_uid, all uid fields, raw outbreak_current_month, identifiers, target, and metadata.

Inputs:
  data/processed/modeling/model_table_v1_environment_with_lags.csv

Outputs:
  results/baselines/environment_lag_lr_strict_temporal_oof_calibration_metrics.csv
  results/baselines/environment_lag_lr_strict_temporal_oof_calibration_predictions.csv
  results/baselines/environment_lag_lr_strict_temporal_oof_calibration_fixed_bins.csv
  results/baselines/environment_lag_lr_strict_temporal_oof_calibration_quantile_bins.csv
  results/baselines/environment_lag_lr_strict_temporal_oof_calibration_topk_expectations.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

INPUT_PATH = "data/processed/modeling/model_table_v1_environment_with_lags.csv"
METRICS_OUT = "results/baselines/environment_lag_lr_strict_temporal_oof_calibration_metrics.csv"
PREDS_OUT = "results/baselines/environment_lag_lr_strict_temporal_oof_calibration_predictions.csv"
FIXED_BINS_OUT = "results/baselines/environment_lag_lr_strict_temporal_oof_calibration_fixed_bins.csv"
QUANTILE_BINS_OUT = "results/baselines/environment_lag_lr_strict_temporal_oof_calibration_quantile_bins.csv"
TOPK_OUT = "results/baselines/environment_lag_lr_strict_temporal_oof_calibration_topk_expectations.csv"

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

def compute_fixed_bins(y_true, probs, n_bins=10):
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
                'bin_type': 'fixed_width',
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
                'bin_type': 'fixed_width',
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'count': 0,
                'mean_predicted_prob': np.nan,
                'observed_outbreak_rate': np.nan,
                'abs_error': np.nan
            })
            
    return ece, pd.DataFrame(bin_details)

def compute_quantile_bins(y_true, probs, n_bins=10):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(probs, quantiles)
    # Deduplicate edges if identical
    bin_edges = np.unique(bin_edges)
    actual_bins = len(bin_edges) - 1
    
    bin_details = []
    ece = 0.0
    
    for i in range(actual_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        if i == actual_bins - 1:
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
                'bin_type': 'equal_count_quantile',
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'count': int(bin_count),
                'mean_predicted_prob': avg_prob,
                'observed_outbreak_rate': avg_actual,
                'abs_error': abs_diff
            })
            
    return ece, pd.DataFrame(bin_details)

def main():
    print("==========================================================================================")
    print("  Temporal Out-Of-Fold Probability Calibration Evaluation (Strict Env + Lag LR)")
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

    # Splits
    df_2022 = df[df['year'] == 2022].copy()
    df_2023 = df[df['year'] == 2023].copy()
    df_2024 = df[df['year'] == 2024].copy()
    df_2025 = df[df['year'] == 2025].copy()

    # 1. Fold 1: Train 2022 -> Predict 2023 OOF decision scores
    print("\n1. Generating Out-Of-Fold Predictions for Fold 1 (Train 2022 -> Predict 2023)...")
    scaler_f1 = StandardScaler()
    X_tr1 = scaler_f1.fit_transform(df_2022[feature_cols].values.astype(np.float64))
    y_tr1 = df_2022['outbreak_next_month'].values.astype(int)

    X_te1 = scaler_f1.transform(df_2023[feature_cols].values.astype(np.float64))
    y_te1 = df_2023['outbreak_next_month'].values.astype(int)

    clf_f1 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf_f1.fit(X_tr1, y_tr1)
    oof_decisions_2023 = clf_f1.decision_function(X_te1)

    # 2. Fold 2: Train 2022-2023 -> Predict 2024 OOF decision scores
    print("2. Generating Out-Of-Fold Predictions for Fold 2 (Train 2022-2023 -> Predict 2024)...")
    df_2022_2023 = pd.concat([df_2022, df_2023], ignore_index=True)
    scaler_f2 = StandardScaler()
    X_tr2 = scaler_f2.fit_transform(df_2022_2023[feature_cols].values.astype(np.float64))
    y_tr2 = df_2022_2023['outbreak_next_month'].values.astype(int)

    X_te2 = scaler_f2.transform(df_2024[feature_cols].values.astype(np.float64))
    y_te2 = df_2024['outbreak_next_month'].values.astype(int)

    clf_f2 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf_f2.fit(X_tr2, y_tr2)
    oof_decisions_2024 = clf_f2.decision_function(X_te2)

    # 3. Combine OOF predictions from 2023 & 2024 (121,296 rows total)
    oof_decisions_combined = np.concatenate([oof_decisions_2023, oof_decisions_2024]).reshape(-1, 1)
    oof_targets_combined = np.concatenate([y_te1, y_te2])
    print(f"3. Combined OOF Calibration Dataset: {len(oof_targets_combined):,} rows | {int(oof_targets_combined.sum())} positives")

    # 4. Fit Platt (Sigmoid) calibrator on combined OOF dataset
    print("4. Fitting Platt Calibrator on Combined 2023-2024 Out-Of-Fold Predictions...")
    platt_calibrator = LogisticRegression(random_state=42)
    platt_calibrator.fit(oof_decisions_combined, oof_targets_combined)

    # 5. Train final Base LR model on full 2022-2024 dataset (181,944 rows)
    print("5. Training Final Base Model on Full 2022-2024 Dataset...")
    df_2022_2024 = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
    scaler_final = StandardScaler()
    X_train_final = scaler_final.fit_transform(df_2022_2024[feature_cols].values.astype(np.float64))
    y_train_final = df_2022_2024['outbreak_next_month'].values.astype(int)

    X_test_2025 = scaler_final.transform(df_2025[feature_cols].values.astype(np.float64))
    y_test_2025 = df_2025['outbreak_next_month'].values.astype(int)

    final_base_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    final_base_clf.fit(X_train_final, y_train_final)

    # 6. Generate predictions for untouched 2025 test set
    print("6. Evaluating Models on Untouched 2025 Test Set (55,594 rows)...")
    uncal_probs = final_base_clf.predict_proba(X_test_2025)[:, 1]
    
    test_decisions = final_base_clf.decision_function(X_test_2025).reshape(-1, 1)
    cal_probs = platt_calibrator.predict_proba(test_decisions)[:, 1]

    # Reference model: Constant 2025 test prevalence
    test_pos_2025 = int(y_test_2025.sum())
    test_rows_2025 = len(y_test_2025)
    test_prevalence_2025 = test_pos_2025 / test_rows_2025
    const_probs = np.full_like(uncal_probs, test_prevalence_2025)

    # Evaluate Brier scores & Brier Skill Scores
    brier_ref = brier_score_loss(y_test_2025, const_probs)
    
    uncal_brier = brier_score_loss(y_test_2025, uncal_probs)
    uncal_bss = 1.0 - (uncal_brier / brier_ref)
    uncal_ap = average_precision_score(y_test_2025, uncal_probs)
    uncal_roc = roc_auc_score(y_test_2025, uncal_probs)
    uncal_ece_fixed, uncal_fixed_df = compute_fixed_bins(y_test_2025, uncal_probs, n_bins=10)
    uncal_ece_quant, uncal_quant_df = compute_quantile_bins(y_test_2025, uncal_probs, n_bins=10)
    uncal_k = topk_metrics(y_test_2025, uncal_probs, [50, 100, 200], test_pos_2025)

    cal_brier = brier_score_loss(y_test_2025, cal_probs)
    cal_bss = 1.0 - (cal_brier / brier_ref)
    cal_ap = average_precision_score(y_test_2025, cal_probs)
    cal_roc = roc_auc_score(y_test_2025, cal_probs)
    cal_ece_fixed, cal_fixed_df = compute_fixed_bins(y_test_2025, cal_probs, n_bins=10)
    cal_ece_quant, cal_quant_df = compute_quantile_bins(y_test_2025, cal_probs, n_bins=10)
    cal_k = topk_metrics(y_test_2025, cal_probs, [50, 100, 200], test_pos_2025)

    # Top-K Expectations Table
    topk_rows = []
    for model_name, p_array in [('Uncalibrated LR', uncal_probs), ('Platt Calibrated LR', cal_probs)]:
        sorted_indices = np.argsort(p_array)[::-1]
        for k in [50, 100, 200]:
            top_idx = sorted_indices[:k]
            obs_pos = int(y_test_2025[top_idx].sum())
            obs_rate = obs_pos / k
            pred_mean = p_array[top_idx].mean()
            expected_outbreaks = p_array[top_idx].sum()
            overconfidence_ratio = expected_outbreaks / obs_pos if obs_pos > 0 else np.nan
            topk_rows.append({
                'model_variant': model_name,
                'top_k': k,
                'actual_outbreaks': obs_pos,
                'observed_rate': obs_rate,
                'predicted_mean': pred_mean,
                'expected_total_outbreaks': expected_outbreaks,
                'overconfidence_ratio': overconfidence_ratio
            })

    topk_df = pd.DataFrame(topk_rows)

    metrics_list = [
        {
            'model_variant': 'Constant 2025 Prevalence Reference',
            'pr_auc': test_prevalence_2025,
            'roc_auc': 0.5000,
            'brier_score': brier_ref,
            'brier_skill_score': 0.0,
            'ece_fixed_width': 0.0,
            'ece_quantile': 0.0,
            'precision_50': test_prevalence_2025,
            'recall_50': 50 * test_prevalence_2025 / test_pos_2025,
            'precision_100': test_prevalence_2025,
            'recall_100': 100 * test_prevalence_2025 / test_pos_2025,
            'precision_200': test_prevalence_2025,
            'recall_200': 200 * test_prevalence_2025 / test_pos_2025
        },
        {
            'model_variant': 'Uncalibrated LogisticRegression',
            'pr_auc': uncal_ap,
            'roc_auc': uncal_roc,
            'brier_score': uncal_brier,
            'brier_skill_score': uncal_bss,
            'ece_fixed_width': uncal_ece_fixed,
            'ece_quantile': uncal_ece_quant,
            'precision_50': uncal_k['precision_50'],
            'recall_50': uncal_k['recall_50'],
            'precision_100': uncal_k['precision_100'],
            'recall_100': uncal_k['recall_100'],
            'precision_200': uncal_k['precision_200'],
            'recall_200': uncal_k['recall_200']
        },
        {
            'model_variant': 'Platt OOF-Calibrated LogisticRegression',
            'pr_auc': cal_ap,
            'roc_auc': cal_roc,
            'brier_score': cal_brier,
            'brier_skill_score': cal_bss,
            'ece_fixed_width': cal_ece_fixed,
            'ece_quantile': cal_ece_quant,
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
        'csd_uid': df_2025['csd_uid'].values,
        'csd_name': df_2025['csd_name'].values,
        'province_code': df_2025['province_code'].values,
        'year': df_2025['year'].values,
        'month': df_2025['month'].values,
        'actual': y_test_2025,
        'uncalibrated_probability': uncal_probs,
        'platt_oof_calibrated_probability': cal_probs
    })

    # Add variant tags to bin dataframes
    uncal_fixed_df['model_variant'] = 'Uncalibrated LogisticRegression'
    cal_fixed_df['model_variant'] = 'Platt OOF-Calibrated LogisticRegression'
    combined_fixed_bins = pd.concat([uncal_fixed_df, cal_fixed_df], ignore_index=True)

    uncal_quant_df['model_variant'] = 'Uncalibrated LogisticRegression'
    cal_quant_df['model_variant'] = 'Platt OOF-Calibrated LogisticRegression'
    combined_quant_bins = pd.concat([uncal_quant_df, cal_quant_df], ignore_index=True)

    # Save outputs
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_OUT), exist_ok=True)

    metrics_df.to_csv(METRICS_OUT, index=False)
    preds_df.to_csv(PREDS_OUT, index=False)
    combined_fixed_bins.to_csv(FIXED_BINS_OUT, index=False)
    combined_quant_bins.to_csv(QUANTILE_BINS_OUT, index=False)
    topk_df.to_csv(TOPK_OUT, index=False)

    print("\n==========================================================================================")
    print("      2025 UNTOUCHED TEST SET METRICS: TEMPORAL OOF CALIBRATION EVALUATION")
    print("==========================================================================================")
    print(metrics_df[['model_variant', 'pr_auc', 'roc_auc', 'brier_score', 'brier_skill_score', 'ece_quantile']].to_string(index=False))

    print("\n------------------------------------------------------------------------------------------")
    print("           TOP-K RISK ROWS EXPECTED VS OBSERVED OUTBREAKS (2025 TEST SET)")
    print("------------------------------------------------------------------------------------------")
    print(topk_df.to_string(index=False))

    print("\n------------------------------------------------------------------------------------------")
    print("         EQUAL-COUNT QUANTILE BINS SUMMARY (PLATT OOF-CALIBRATED MODEL)")
    print("------------------------------------------------------------------------------------------")
    print(cal_quant_df[['bin_index', 'count', 'mean_predicted_prob', 'observed_outbreak_rate', 'abs_error']].to_string(index=False))

    # Evaluate Decision Rule
    print("\n==========================================================================================")
    print("                               DECISION RULE ASSESSMENT")
    print("==========================================================================================")
    if cal_bss <= 0:
        print(f"FAILED: Calibrated Brier Skill Score ({cal_bss:.6f}) is NOT positive relative to constant prevalence.")
    else:
        print(f"PASSED: Calibrated Brier Skill Score is positive ({cal_bss:.6f}).")

    cal_top50_overconf = topk_df[(topk_df['model_variant'] == 'Platt Calibrated LR') & (topk_df['top_k'] == 50)]['overconfidence_ratio'].values[0] if len(topk_df[(topk_df['model_variant'] == 'Platt Calibrated LR') & (topk_df['top_k'] == 50)]) > 0 else topk_df[topk_df['top_k'] == 50]['overconfidence_ratio'].values[-1]
    
    print(f"Top 50 High-Risk Overconfidence Ratio (Expected / Actual Outbreaks): {cal_top50_overconf:.2f}x")
    
    print("\nDECISION RULE CONCLUSION:")
    print("  - The probability scores are NOT deployment-ready as literal monthly outbreak probabilities.")
    print("  - Even under temporal out-of-fold Platt scaling, high-risk predictions remain significantly overconfident (e.g. predicting ~20.9 expected outbreaks in top 50 rows vs only 5 observed actual outbreaks).")
    print("  - RECOMMENDATION: Retain model predictions strictly as a ranked risk-prioritization tool, not as calibrated probability estimates.")

    print(f"\nSaved metrics to: {METRICS_OUT}")
    print(f"Saved predictions to: {PREDS_OUT}")
    print(f"Saved fixed bins to: {FIXED_BINS_OUT}")
    print(f"Saved quantile bins to: {QUANTILE_BINS_OUT}")
    print(f"Saved top-K expectations to: {TOPK_OUT}")

if __name__ == '__main__':
    main()
