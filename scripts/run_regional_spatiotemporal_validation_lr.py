"""
Run Regional Spatial-Temporal Validation for Environment + Lag Logistic Regression.

Purpose:
Performs leave-one-region-out spatial-temporal validation to evaluate geographic generalizability:
  - Training: 2022-2024 data, excluding all CSDs in the held-out regional block.
  - Testing: 2025 data for CSDs in the held-out regional block only.

Regional Blocks:
  - Atlantic: NL (10), PEI (11), NS (12), NB (13)
  - Quebec_Ontario: QC (24), ON (35)
  - Prairies: MB (46), SK (47), AB (48)
  - British_Columbia: BC (59)

Inputs:
  data/processed/modeling/model_table_v1_environment_with_lags.csv
  results/baselines/environment_with_lags_v1_metrics.csv

Outputs:
  results/baselines/regional_spatiotemporal_lr_metrics.csv
  results/baselines/regional_spatiotemporal_lr_predictions.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

INPUT_PATH = "data/processed/modeling/model_table_v1_environment_with_lags.csv"
ORDINARY_METRICS_PATH = "results/baselines/environment_with_lags_v1_metrics.csv"
METRICS_OUT = "results/baselines/regional_spatiotemporal_lr_metrics.csv"
PREDS_OUT = "results/baselines/regional_spatiotemporal_lr_predictions.csv"

REGIONAL_BLOCKS = {
    'Atlantic': [10, 11, 12, 13],
    'Quebec_Ontario': [24, 35],
    'Prairies': [46, 47, 48],
    'British_Columbia': [59]
}

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
    print("  Regional Spatial-Temporal Validation for Environment + Lag Logistic Regression")
    print("==========================================================================================")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # Read ordinary temporal metrics directly from existing CSV
    ordinary_metrics_df = pd.read_csv(ORDINARY_METRICS_PATH)
    lr_ord = ordinary_metrics_df[ordinary_metrics_df['model'] == 'LogisticRegression'].iloc[0]
    ord_pr_auc = float(lr_ord['pr_auc'])
    ord_roc_auc = float(lr_ord['roc_auc'])
    print(f"Loaded Ordinary Temporal (2025 National) LR Metrics from CSV:")
    print(f"  - PR-AUC:  {ord_pr_auc:.6f}")
    print(f"  - ROC-AUC: {ord_roc_auc:.6f}")

    df = pd.read_csv(INPUT_PATH, dtype={'csd_uid': str})
    print(f"Loaded dataset with shape: {df.shape}")

    # Exclude identifiers, target, and non-numeric columns
    exclude_cols = {
        'csd_uid', 'csd_name', 'province_code', 'province_name',
        'year', 'month', 'outbreak_next_month',
        'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
        'glw2015_duck_extraction_method', 'road_data_coverage'
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_features
    print(f"Selected numeric feature count: {len(feature_cols)}")

    # Ensure no eBird features exist in input dataset
    ebird_cols = [c for c in df.columns if 'ebird' in c.lower()]
    assert len(ebird_cols) == 0, f"eBird features detected in input file: {ebird_cols}"

    all_metrics = []
    all_predictions = []

    for block_name, prov_codes in REGIONAL_BLOCKS.items():
        print(f"\n------------------------------------------------------------------------------------------")
        print(f"Processing Regional Block: {block_name} (Provinces: {prov_codes})")

        # 1. Training data: 2022-2024 rows excluding held-out block CSDs
        train_df = df[(df['year'].isin([2022, 2023, 2024])) & (~df['province_code'].isin(prov_codes))].copy()
        
        # 2. Test data: 2025 rows for held-out block CSDs only
        test_df = df[(df['year'] == 2025) & (df['province_code'].isin(prov_codes))].copy()

        train_rows = len(train_df)
        test_rows = len(test_df)
        train_pos = int(train_df['outbreak_next_month'].sum())
        test_pos = int(test_df['outbreak_next_month'].sum())

        test_positive_rate = test_pos / test_rows if test_rows > 0 else 0.0

        small_sample_note = ""
        if test_pos < 10:
            small_sample_note = f"WARNING: Test positive count ({test_pos}) is very small (< 10); metrics may be unstable."

        print(f"  Train rows: {train_rows:,} | Train Positives: {train_pos}")
        print(f"  Test rows:  {test_rows:,} | Test Positives:  {test_pos} (Prevalence: {test_positive_rate:.6f})")
        if small_sample_note:
            print(f"  {small_sample_note}")

        X_train = train_df[feature_cols].values.astype(np.float64)
        y_train = train_df['outbreak_next_month'].values.astype(int)
        X_test = test_df[feature_cols].values.astype(np.float64)
        y_test = test_df['outbreak_next_month'].values.astype(int)

        # Scale features using training data only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit weighted Logistic Regression
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        probs = clf.predict_proba(X_test_scaled)[:, 1]

        # Evaluate metrics
        ap = average_precision_score(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        k_metrics = topk_metrics(y_test, probs, [50, 100, 200], test_pos)

        pr_auc_lift = ap / test_positive_rate if test_positive_rate > 0 else 0.0

        metrics_rec = {
            'regional_block': block_name,
            'province_codes': str(prov_codes),
            'train_rows': train_rows,
            'train_positives': train_pos,
            'test_rows': test_rows,
            'test_positives': test_pos,
            'test_positive_rate': test_positive_rate,
            'pr_auc': ap,
            'pr_auc_lift': pr_auc_lift,
            'roc_auc': roc_auc,
            'precision_50': k_metrics['precision_50'],
            'recall_50': k_metrics['recall_50'],
            'precision_100': k_metrics['precision_100'],
            'recall_100': k_metrics['recall_100'],
            'precision_200': k_metrics['precision_200'],
            'recall_200': k_metrics['recall_200'],
            'small_sample_warning': small_sample_note
        }
        all_metrics.append(metrics_rec)

        block_preds = pd.DataFrame({
            'regional_block': block_name,
            'csd_uid': test_df['csd_uid'].values,
            'csd_name': test_df['csd_name'].values,
            'province_code': test_df['province_code'].values,
            'year': test_df['year'].values,
            'month': test_df['month'].values,
            'actual': y_test,
            'predicted_probability': probs
        })
        all_predictions.append(block_preds)

    df_metrics = pd.DataFrame(all_metrics)
    df_preds = pd.concat(all_predictions, ignore_index=True)

    # Save outputs
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_OUT), exist_ok=True)
    df_metrics.to_csv(METRICS_OUT, index=False)
    df_preds.to_csv(PREDS_OUT, index=False)

    print("\n==========================================================================================")
    print("           REGIONAL SPATIAL-TEMPORAL LOGISTIC REGRESSION METRICS")
    print("==========================================================================================")
    summary_cols = ['regional_block', 'train_positives', 'test_positives', 'test_positive_rate', 'pr_auc', 'pr_auc_lift', 'roc_auc', 'precision_50', 'recall_50', 'precision_100', 'recall_100', 'precision_200', 'recall_200']
    print(df_metrics[summary_cols].to_string(index=False))

    print("\n------------------------------------------------------------------------------------------")
    print("             COMPARISON WITH ORDINARY TEMPORAL (2025) VALIDATION")
    print("------------------------------------------------------------------------------------------")
    print(f"Ordinary Temporal (2025 National Test) LR PR-AUC:  {ord_pr_auc:.6f}")
    print(f"Ordinary Temporal (2025 National Test) LR ROC-AUC: {ord_roc_auc:.6f}")
    print("\nRegional Block PR-AUC Comparison:")
    for _, row in df_metrics.iterrows():
        diff = row['pr_auc'] - ord_pr_auc
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        note = f" ({row['small_sample_warning']})" if row['small_sample_warning'] else ""
        print(f"  - {row['regional_block']:<18}: PR-AUC = {row['pr_auc']:.4f} (Lift: {row['pr_auc_lift']:.2f}x, vs National: {diff_str}){note}")

    print("\nMETHODOLOGICAL INTERPRETATION NOTE:")
    print("  - Performance varied substantially across held-out regional blocks.")
    print("  - The environment + lag spatial validation test retains lagged outbreak-history information for the held-out region.")
    print("  - It measures operational geographic transfer conditional on available outbreak history, not pure environmental geographic generalization.")

    print(f"\nSaved metrics to: {METRICS_OUT}")
    print(f"Saved predictions to: {PREDS_OUT}")

if __name__ == '__main__':
    main()
