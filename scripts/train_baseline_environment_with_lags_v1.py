#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Train baseline models (Logistic Regression and Random Forest) on the environment+lag features.

Input:  data/processed/modeling/model_table_v1_environment_with_lags.csv
Outputs:
  - results/baselines/environment_with_lags_v1_metrics.csv
  - results/baselines/environment_with_lags_v1_test_predictions.csv

The script performs a temporal split (train 2022-2024, test 2025) and reports PR-AUC, ROC-AUC,
precision/recall at K=50,100,200.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def topk_metrics(y_true, probs, k_list, total_positives):
    """Return precision and recall for each K in k_list.
    y_true and probs are NumPy arrays.
    """
    sorted_idx = np.argsort(probs)[::-1]
    sorted_y = y_true[sorted_idx]
    results = {}
    for k in k_list:
        top_k = sorted_y[:k]
        pos_in_top = int(top_k.sum())
        precision = pos_in_top / k
        recall = pos_in_top / total_positives if total_positives > 0 else 0.0
        results[f'precision_at_{k}'] = precision
        results[f'recall_at_{k}'] = recall
    return results


def train_baselines(input_path, metrics_out, preds_out):
    print("Training baseline models with lag/history features (V1 environment + lags)...")
    print("===================================================================")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load data
    df = pd.read_csv(input_path, dtype={'csd_uid': str})

    # Temporal split
    train_df = df[df['year'].isin([2022, 2023, 2024])].copy()
    test_df = df[df['year'] == 2025].copy()

    # Columns to exclude from features
    exclude_cols = [
        'csd_uid', 'csd_name', 'province_code', 'province_name',
        'year', 'month', 'outbreak_next_month'
    ]
    # Keep all remaining numeric columns (including lag columns)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_features

    # Prepare training data
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df['outbreak_next_month'].astype(int)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df['outbreak_next_month'].astype(int)

    train_pos = int(y_train.sum())
    test_pos = int(y_test.sum())
    print(f"Dataset split summary:\n  Train rows: {len(X_train)}, positives: {train_pos} ({train_pos/len(X_train)*100:.4f}%)\n  Test rows: {len(X_test)}, positives: {test_pos} ({test_pos/len(X_test)*100:.4f}%)")
    print(f"Number of features: {len(feature_cols)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model definitions (lightweight)
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    }

    metrics_list = []
    predictions = {
        'csd_uid': test_df['csd_uid'].values,
        'csd_name': test_df['csd_name'].values,
        'province_code': test_df['province_code'].values,
        'year': test_df['year'].values,
        'month': test_df['month'].values,
        'actual': y_test.values
    }

    for name, clf in models.items():
        print(f"Fitting {name} ...")
        clf.fit(X_train_scaled, y_train)
        probs = clf.predict_proba(X_test_scaled)[:, 1]
        predictions[f"{name}_prob"] = probs

        ap = average_precision_score(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        k_metrics = topk_metrics(y_test.values, probs, [50, 100, 200], test_pos)

        metrics_list.append({
            'model': name,
            'pr_auc': ap,
            'roc_auc': roc_auc,
            'precision_at_50': k_metrics['precision_at_50'],
            'recall_at_50': k_metrics['recall_at_50'],
            'precision_at_100': k_metrics['precision_at_100'],
            'recall_at_100': k_metrics['recall_at_100'],
            'precision_at_200': k_metrics['precision_at_200'],
            'recall_at_200': k_metrics['recall_at_200']
        })

    # Save metrics and predictions
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    os.makedirs(os.path.dirname(preds_out), exist_ok=True)
    pd.DataFrame(metrics_list).to_csv(metrics_out, index=False)
    pd.DataFrame(predictions).to_csv(preds_out, index=False)

    # Print a concise summary
    print("\n=== Baseline Metrics Summary ===")
    print(pd.DataFrame(metrics_list).to_string(index=False))
    print("=== End of Summary ===")

    # Show top 20 risk predictions from RandomForest (higher risk first)
    if 'RandomForest_prob' in predictions:
        print("\nTop 20 Highest Risk Predictions (RandomForest, 2025):")
        pred_df = pd.DataFrame(predictions)
        top20 = pred_df.sort_values(by='RandomForest_prob', ascending=False).head(20)
        for i, row in top20.iterrows():
            print(f"{i+1:2d}. CSD {row['csd_uid']} ({row['csd_name']}), month {int(row['month'])}: prob={row['RandomForest_prob']:.4f}, actual={int(row['actual'])}")

    print("\nTraining completed.")

if __name__ == '__main__':
    default_input = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
    default_metrics = 'results/baselines/environment_with_lags_v1_metrics.csv'
    default_preds = 'results/baselines/environment_with_lags_v1_test_predictions.csv'
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    metrics_out = sys.argv[2] if len(sys.argv) > 2 else default_metrics
    preds_out = sys.argv[3] if len(sys.argv) > 3 else default_preds
    try:
        train_baselines(input_path, metrics_out, preds_out)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
