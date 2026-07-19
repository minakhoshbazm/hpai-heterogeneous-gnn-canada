#!/usr/bin/env python3
"""
scripts/train_neighbor_pressure_tabular_baselines.py

Trains Logistic Regression and Random Forest models on environment + lags
+ neighbor outbreak pressure features.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
    model_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags_and_neighbor_pressure.csv'
    metrics_out = 'results/baselines/neighbor_pressure_tabular_metrics.csv'
    preds_out = 'results/baselines/neighbor_pressure_tabular_test_predictions.csv'

    print("Loading enriched dataset...")
    df_all = pd.read_csv(model_table_path)
    
    # Train/Test splits
    df_train = df_all[df_all['year'].isin([2022, 2023, 2024])].copy()
    df_test = df_all[df_all['year'] == 2025].copy()

    # Determine numeric feature columns
    exclude_cols = {
        'csd_uid', 'csd_name', 'province_code', 'province_name', 
        'year', 'month', 'outbreak_next_month',
        'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
        'glw2015_duck_extraction_method', 'road_data_coverage'
    }
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    print(f"Num features: {len(feature_cols)}")

    X_train_raw = df_train[feature_cols].fillna(0.0)
    y_train = df_train['outbreak_next_month'].values

    X_test_raw = df_test[feature_cols].fillna(0.0)
    y_test = df_test['outbreak_next_month'].values

    # Scale inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    results = []
    test_preds_df = pd.DataFrame({
        'csd_uid': df_test['csd_uid'],
        'year': df_test['year'],
        'month': df_test['month'],
        'outbreak_next_month': df_test['outbreak_next_month']
    }).reset_index(drop=True)

    # 1. Logistic Regression
    print("\nTraining Logistic Regression with neighbor pressure...")
    clf_lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    clf_lr.fit(X_train_scaled, y_train)
    probs_lr = clf_lr.predict_proba(X_test_scaled)[:, 1]
    
    m_lr = evaluate_metrics(y_test, probs_lr)
    m_lr['model_name'] = 'LR + Neighbor Pressure'
    results.append(m_lr)
    test_preds_df['prob_lr'] = probs_lr
    
    # 2. Random Forest
    print("Training Random Forest with neighbor pressure (this might take a minute)...")
    clf_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )
    clf_rf.fit(X_train_raw, y_train)  # RF handles unscaled features fine
    probs_rf = clf_rf.predict_proba(X_test_raw)[:, 1]
    
    m_rf = evaluate_metrics(y_test, probs_rf)
    m_rf['model_name'] = 'RF + Neighbor Pressure'
    results.append(m_rf)
    test_preds_df['prob_rf'] = probs_rf

    df_metrics = pd.DataFrame(results)

    print("\n=========================================================================")
    print("                Neighbor Outbreak Pressure Tabular Baselines")
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

    print("\nComparison against previous baselines:")
    print("  Env + Lag LogisticRegression:  PR-AUC 0.0635, Precision@50 0.120")
    print("  Env + Lag RandomForest:        PR-AUC 0.0308, Precision@50 0.060")
    print("  GraphSAGE early-stopping mean: PR-AUC 0.0735, Precision@50 0.140")

    # Save output files
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    df_metrics.to_csv(metrics_out, index=False)
    test_preds_df.to_csv(preds_out, index=False)
    print(f"\nSaved metrics to: {metrics_out}")
    print(f"Saved predictions to: {preds_out}")

if __name__ == '__main__':
    main()
