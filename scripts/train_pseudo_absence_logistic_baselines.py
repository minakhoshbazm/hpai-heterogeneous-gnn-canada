#!/usr/bin/env python3
"""
scripts/train_pseudo_absence_logistic_baselines.py

Trains Logistic Regression baselines using pseudo-absence sampled sets
and evaluates them on the full, imbalanced 2025 test set.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

def evaluate_metrics(y_true, y_scores):
    ap = average_precision_score(y_true, y_scores)
    roc = roc_auc_score(y_true, y_scores)
    
    # Sort indices by prediction scores descending
    sorted_indices = np.argsort(y_scores)[::-1]
    
    metrics = {
        'pr_auc': ap,
        'roc_auc': roc
    }
    
    for k in [50, 100, 200]:
        top_k = sorted_indices[:k]
        hits = y_true[top_k].sum()
        precision = hits / k
        recall = hits / y_true.sum() if y_true.sum() > 0 else 0.0
        metrics[f'precision_{k}'] = precision
        metrics[f'recall_{k}'] = recall
        
    return metrics

def main():
    model_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
    metrics_out = 'results/baselines/pseudo_absence_logistic_metrics.csv'
    preds_out = 'results/baselines/pseudo_absence_logistic_test_predictions.csv'

    print("Loading test data...")
    df_all = pd.read_csv(model_table_path)
    df_test = df_all[df_all['year'] == 2025].copy()
    print(f"Test split rows (2025): {len(df_test)}")

    # Exclude metadata and non-numeric columns
    exclude_cols = {
        'csd_uid', 'csd_name', 'province_code', 'province_name', 
        'year', 'month', 'outbreak_next_month',
        'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
        'glw2015_duck_extraction_method', 'road_data_coverage'
    }
    feature_cols = [c for c in df_test.columns if c not in exclude_cols]
    print(f"Num features: {len(feature_cols)}")

    ratios = [5, 10, 20]
    results = []
    test_preds_df = pd.DataFrame({
        'csd_uid': df_test['csd_uid'],
        'year': df_test['year'],
        'month': df_test['month'],
        'outbreak_next_month': df_test['outbreak_next_month']
    }).reset_index(drop=True)

    y_test = df_test['outbreak_next_month'].values

    for r in ratios:
        train_path = f"data/processed/modeling/pseudo_absence_train_ratio_{r}.csv"
        print(f"\nTraining model for ratio {r}:1 from {train_path}...")
        
        df_train = pd.read_csv(train_path)
        
        # Fit scaler on train split only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[feature_cols].fillna(0.0))
        y_train = df_train['outbreak_next_month'].values
        
        X_test = scaler.transform(df_test[feature_cols].fillna(0.0))
        
        # Fit logistic regression
        clf = LogisticRegression(max_iter=2000, class_weight=None, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predict probability of positive class
        # sklearn classes: [0, 1]
        probs = clf.predict_proba(X_test)[:, 1]
        
        # Evaluate
        m = evaluate_metrics(y_test, probs)
        m['model_name'] = f"Pseudo-Absence {r}:1 LR"
        results.append(m)
        
        test_preds_df[f'prob_ratio_{r}'] = probs
        
        print(f"  PR-AUC:        {m['pr_auc']:.4f}")
        print(f"  ROC-AUC:       {m['roc_auc']:.4f}")
        print(f"  Precision@50:  {m['precision_50']:.3f}")
        print(f"  Recall@50:     {m['recall_50']:.4f}")

    df_metrics = pd.DataFrame(results)
    
    print("\n=========================================================================")
    print("                Pseudo-Absence Logistic Regression Results")
    print("=========================================================================")
    print(df_metrics.to_string(index=False, formatters={
        'pr_auc': '{:,.4f}'.format,
        'roc_auc': '{:,.4f}'.format,
        'precision_50': '{:,.3f}'.format,
        'recall_50': '{:,.4f}'.format,
        'precision_100': '{:,.3f}'.format,
        'recall_100': '{:,.4f}'.format,
        'precision_200': '{:,.3f}'.format,
        'recall_200': '{:,.4f}'.format,
    }))
    print("=========================================================================")

    print("\nComparison against baselines:")
    print("  Env + Lag LogisticRegression: PR-AUC 0.0635, Precision@50 0.120")
    print("  GraphSAGE Spatial GNN:        PR-AUC 0.0953, Precision@50 0.180")
    print("  Time-persistence Baseline:    PR-AUC 0.0405, Precision@50 0.200")

    # Save output files
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    df_metrics.to_csv(metrics_out, index=False)
    test_preds_df.to_csv(preds_out, index=False)
    print(f"\nSaved metrics to: {metrics_out}")
    print(f"Saved predictions to: {preds_out}")

if __name__ == '__main__':
    main()
