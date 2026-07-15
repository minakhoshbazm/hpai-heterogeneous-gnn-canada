#!/usr/bin/env python3
"""
scripts/summarize_baseline_results_v1.py

Aggregates baseline metrics and prints a sorted comparison table.
"""

import os
import pandas as pd

def main():
    metrics_dir = 'results/baselines'
    out_summary_path = os.path.join(metrics_dir, 'baseline_model_comparison_summary.csv')

    records = []

    # 1. Load Environment-only baseline
    path_env = os.path.join(metrics_dir, 'environment_v1_baseline_metrics.csv')
    if os.path.exists(path_env):
        df = pd.read_csv(path_env)
        # It contains LogisticRegression and RandomForest
        for idx, row in df.iterrows():
            model_name = f"Env-only {row['model']}"
            records.append({
                'model_name': model_name,
                'pr_auc': row.get('ap', row.get('pr_auc', 0.0)),
                'roc_auc': row.get('roc_auc', 0.0),
                'precision_at_50': row.get('precision_at_50', 0.0),
                'recall_at_50': row.get('recall_at_50', 0.0),
                'precision_at_100': row.get('precision_at_100', 0.0),
                'recall_at_100': row.get('recall_at_100', 0.0),
                'precision_at_200': row.get('precision_at_200', 0.0),
                'recall_at_200': row.get('recall_at_200', 0.0)
            })

    # 2. Load Environment with Lags baseline
    path_lags = os.path.join(metrics_dir, 'environment_with_lags_v1_metrics.csv')
    if os.path.exists(path_lags):
        df = pd.read_csv(path_lags)
        for idx, row in df.iterrows():
            model_name = f"Env + Lag {row['model']}"
            records.append({
                'model_name': model_name,
                'pr_auc': row.get('pr_auc', row.get('ap', 0.0)),
                'roc_auc': row.get('roc_auc', 0.0),
                'precision_at_50': row.get('precision_at_50', 0.0),
                'recall_at_50': row.get('recall_at_50', 0.0),
                'precision_at_100': row.get('precision_at_100', 0.0),
                'recall_at_100': row.get('recall_at_100', 0.0),
                'precision_at_200': row.get('precision_at_200', 0.0),
                'recall_at_200': row.get('recall_at_200', 0.0)
            })

    # 3. Load Time-persistence baseline
    path_time = os.path.join(metrics_dir, 'time_persistence_baseline_metrics.csv')
    if os.path.exists(path_time):
        df = pd.read_csv(path_time)
        # Column names are 'Unnamed: 0' (metric) and 'value'
        metric_col = df.columns[0]
        val_col = df.columns[1]
        df_kv = df.set_index(metric_col)[val_col]
        records.append({
            'model_name': 'Time-persistence Baseline',
            'pr_auc': df_kv.get('average_precision', 0.0),
            'roc_auc': df_kv.get('roc_auc', 0.0),
            'precision_at_50': df_kv.get('precision_at_50', 0.0),
            'recall_at_50': df_kv.get('recall_at_50', 0.0),
            'precision_at_100': df_kv.get('precision_at_100', 0.0),
            'recall_at_100': df_kv.get('recall_at_100', 0.0),
            'precision_at_200': df_kv.get('precision_at_200', 0.0),
            'recall_at_200': df_kv.get('recall_at_200', 0.0)
        })

    # 4. Load GraphSAGE baseline
    path_sage = os.path.join(metrics_dir, 'graphsage_v1_metrics.csv')
    if os.path.exists(path_sage):
        df = pd.read_csv(path_sage)
        records.append({
            'model_name': 'GraphSAGE Spatial GNN',
            'pr_auc': df.get('PR-AUC', df.get('pr_auc', [0.0]))[0],
            'roc_auc': df.get('ROC-AUC', df.get('roc_auc', [0.0]))[0],
            'precision_at_50': df.get('Precision@50', [0.0])[0],
            'recall_at_50': df.get('Recall@50', [0.0])[0],
            'precision_at_100': df.get('Precision@100', [0.0])[0],
            'recall_at_100': df.get('Recall@100', [0.0])[0],
            'precision_at_200': df.get('Precision@200', [0.0])[0],
            'recall_at_200': df.get('Recall@200', [0.0])[0]
        })

    df_summary = pd.DataFrame(records)
    # Sort models by PR-AUC descending
    df_summary = df_summary.sort_values(by='pr_auc', ascending=False).reset_index(drop=True)

    print("\n=========================================================================")
    print("                      Baseline Model Comparison Summary")
    print("=========================================================================")
    print(df_summary.to_string(index=False, formatters={
        'pr_auc': '{:,.4f}'.format,
        'roc_auc': '{:,.4f}'.format,
        'precision_at_50': '{:,.3f}'.format,
        'recall_at_50': '{:,.4f}'.format,
        'precision_at_100': '{:,.3f}'.format,
        'recall_at_100': '{:,.4f}'.format,
        'precision_at_200': '{:,.3f}'.format,
        'recall_at_200': '{:,.4f}'.format,
    }))
    print("=========================================================================")

    # Save summary
    df_summary.to_csv(out_summary_path, index=False)
    print(f"\nSaved combined comparison summary to: {out_summary_path}")

    # Deduce best models
    best_pr = df_summary.loc[df_summary['pr_auc'].idxmax()]
    best_p50 = df_summary.loc[df_summary['precision_at_50'].idxmax()]
    best_r200 = df_summary.loc[df_summary['recall_at_200'].idxmax()]

    print("\n--- Conclusions ---")
    print(f"- Best overall model by PR-AUC:       {best_pr['model_name']} ({best_pr['pr_auc']:.4f})")
    print(f"- Best overall model by Precision@50: {best_p50['model_name']} ({best_p50['precision_at_50']:.3f})")
    print(f"- Best overall model by Recall@200:   {best_r200['model_name']} ({best_r200['recall_at_200']:.4f})")

if __name__ == '__main__':
    main()
