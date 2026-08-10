#!/usr/bin/env python3
"""
scripts/compare_relational_gnn_vs_baselines_bootstrap.py

Part 2: Cluster-Bootstrap Uncertainty Analysis

Using the valid time-safe 3-seed Relational GNN ensemble predictions on the 2025 test set:
- Compares against strict Environment + Lag Logistic Regression
- Compares against the kNN-3 GraphSAGE ensemble
- Resamples CSD clusters (retaining all 12 months for each sampled CSD)
- Uses 1,000 bootstrap iterations
- Reports observed PR-AUC difference, 95% percentile confidence interval, and fraction of resamples where Relational GNN > baseline
- Applies decision rule: if 95% CI includes zero, state "comparable, not proven better"
- Does NOT describe resample fraction as a probability

Outputs:
- results/baselines/relational_gnn_vs_baselines_bootstrap_comparison.csv
- results/baselines/relational_gnn_vs_baselines_bootstrap_summary.md
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASELINES_DIR = 'results/baselines'

RGCN_PREDS_PATH = os.path.join(BASELINES_DIR, 'relational_gnn_biological_graph_test_predictions.csv')
LR_PREDS_PATH = os.path.join(BASELINES_DIR, 'environment_with_lags_v1_strict_test_predictions.csv')
KNN3_PREDS_PATH = os.path.join(BASELINES_DIR, 'graphsage_selected_spatial_graph_test_predictions.csv')

CSV_OUT = os.path.join(BASELINES_DIR, 'relational_gnn_vs_baselines_bootstrap_comparison.csv')
MD_OUT = os.path.join(BASELINES_DIR, 'relational_gnn_vs_baselines_bootstrap_summary.md')

def main():
    print("==========================================================================================")
    print("      PART 2: CLUSTER-BOOTSTRAP UNCERTAINTY ANALYSIS (RELATIONAL GNN VS BASELINES)")
    print("==========================================================================================")

    # 1. Load predictions
    df_rgcn = pd.read_csv(RGCN_PREDS_PATH, dtype={'csd_uid': str})
    df_lr = pd.read_csv(LR_PREDS_PATH, dtype={'csd_uid': str})
    df_knn3 = pd.read_csv(KNN3_PREDS_PATH, dtype={'csd_uid': str})

    # Validate alignment
    df_rgcn['key'] = df_rgcn['csd_uid'] + '_' + df_rgcn['year'].astype(str) + '_' + df_rgcn['month'].astype(str)
    df_lr['key'] = df_lr['csd_uid'] + '_' + df_lr['year'].astype(str) + '_' + df_lr['month'].astype(str)
    df_knn3['key'] = df_knn3['csd_uid'] + '_' + df_knn3['year'].astype(str) + '_' + df_knn3['month'].astype(str)

    # Calculate kNN-3 ensemble probability if not present
    knn3_seed_cols = [c for c in df_knn3.columns if 'graphsage_selected_knn_3_seed' in c]
    df_knn3['knn3_ensemble_prob'] = df_knn3[knn3_seed_cols].mean(axis=1)

    # Merge on key
    merged = df_rgcn[['key', 'csd_uid', 'year', 'month', 'actual', 'rgcn_ensemble_prob']].merge(
        df_lr[['key', 'strict_env_lag_logistic_prob']], on='key', how='inner'
    ).merge(
        df_knn3[['key', 'knn3_ensemble_prob']], on='key', how='inner'
    )

    print(f"Total merged 2025 test samples: {len(merged):,} rows across {merged['csd_uid'].nunique():,} CSDs.")
    print(f"Total positive outbreaks in 2025 test set: {int(merged['actual'].sum())}")

    # Full dataset observed metrics
    y_true_full = merged['actual'].values
    rgcn_full_ap = average_precision_score(y_true_full, merged['rgcn_ensemble_prob'].values)
    lr_full_ap = average_precision_score(y_true_full, merged['strict_env_lag_logistic_prob'].values)
    knn3_full_ap = average_precision_score(y_true_full, merged['knn3_ensemble_prob'].values)

    diff_full_vs_lr = rgcn_full_ap - lr_full_ap
    diff_full_vs_knn3 = rgcn_full_ap - knn3_full_ap

    print(f"\nObserved 2025 Test PR-AUCs:")
    print(f"  - Relational GNN Ensemble:   {rgcn_full_ap:.4f}")
    print(f"  - Strict Env + Lag LR:       {lr_full_ap:.4f} (Observed diff: {diff_full_vs_lr:+.4f})")
    print(f"  - kNN-3 GraphSAGE Ensemble:  {knn3_full_ap:.4f} (Observed diff: {diff_full_vs_knn3:+.4f})")

    # 2. Cluster Bootstrap setup
    unique_csds = sorted(merged['csd_uid'].unique())
    num_csds = len(unique_csds)
    
    # Pre-group index map by CSD for fast indexing
    csd_group_dict = {csd: group.index.values for csd, group in merged.groupby('csd_uid')}

    num_iterations = 1000
    seed = 42
    rng = np.random.RandomState(seed)

    diffs_vs_lr = []
    diffs_vs_knn3 = []
    rgcn_aps_boot = []
    lr_aps_boot = []
    knn3_aps_boot = []

    print(f"\nRunning {num_iterations:,} cluster-bootstrap iterations (resampling CSDs)...")
    for i in range(num_iterations):
        sampled_csds = rng.choice(unique_csds, size=num_csds, replace=True)
        sampled_indices = np.concatenate([csd_group_dict[c] for c in sampled_csds])

        boot_df = merged.iloc[sampled_indices]
        y_b = boot_df['actual'].values

        if y_b.sum() == 0:
            continue

        rgcn_b_ap = average_precision_score(y_b, boot_df['rgcn_ensemble_prob'].values)
        lr_b_ap = average_precision_score(y_b, boot_df['strict_env_lag_logistic_prob'].values)
        knn3_b_ap = average_precision_score(y_b, boot_df['knn3_ensemble_prob'].values)

        rgcn_aps_boot.append(rgcn_b_ap)
        lr_aps_boot.append(lr_b_ap)
        knn3_aps_boot.append(knn3_b_ap)

        diffs_vs_lr.append(rgcn_b_ap - lr_b_ap)
        diffs_vs_knn3.append(rgcn_b_ap - knn3_b_ap)

    # 3. Percentile Confidence Intervals & Summary Results
    ci_lower_lr, ci_upper_lr = np.percentile(diffs_vs_lr, [2.5, 97.5])
    ci_lower_knn3, ci_upper_knn3 = np.percentile(diffs_vs_knn3, [2.5, 97.5])

    resample_frac_lr = np.mean(np.array(diffs_vs_lr) > 0)
    resample_frac_knn3 = np.mean(np.array(diffs_vs_knn3) > 0)

    # Decision rules
    if ci_lower_lr <= 0 <= ci_upper_lr:
        decision_lr = "comparable, not proven better"
    elif ci_lower_lr > 0:
        decision_lr = "statistically significant superior performance"
    else:
        decision_lr = "inferior performance"

    if ci_lower_knn3 <= 0 <= ci_upper_knn3:
        decision_knn3 = "comparable, not proven better"
    elif ci_lower_knn3 > 0:
        decision_knn3 = "statistically significant superior performance"
    else:
        decision_knn3 = "inferior performance"

    results = [
        {
            'comparison_baseline': 'Strict Environment + Lag Logistic Regression',
            'relational_gnn_pr_auc': round(rgcn_full_ap, 4),
            'baseline_pr_auc': round(lr_full_ap, 4),
            'observed_pr_auc_diff': round(diff_full_vs_lr, 4),
            'ci_95_lower': round(ci_lower_lr, 4),
            'ci_95_upper': round(ci_upper_lr, 4),
            'resample_fraction_rgcn_better': round(resample_frac_lr, 4),
            'decision_conclusion': decision_lr
        },
        {
            'comparison_baseline': 'kNN-3 GraphSAGE Ensemble',
            'relational_gnn_pr_auc': round(rgcn_full_ap, 4),
            'baseline_pr_auc': round(knn3_full_ap, 4),
            'observed_pr_auc_diff': round(diff_full_vs_knn3, 4),
            'ci_95_lower': round(ci_lower_knn3, 4),
            'ci_95_upper': round(ci_upper_knn3, 4),
            'resample_fraction_rgcn_better': round(resample_frac_knn3, 4),
            'decision_conclusion': decision_knn3
        }
    ]

    df_res = pd.DataFrame(results)
    df_res.to_csv(CSV_OUT, index=False)

    # Markdown summary report
    md_content = f"""# Relational GNN vs Baselines Cluster-Bootstrap Uncertainty Analysis

## Executive Summary

To evaluate statistical uncertainty and determine whether the multi-relation Relational GNN (RGCN) ensemble outperforms baseline models, we conducted **1,000 cluster-bootstrap iterations** resampling CSD units (retaining all 12 monthly observations per sampled CSD) on the 2025 held-out test set.

## Summary Results Table

| Baseline Model | Relational GNN PR-AUC | Baseline PR-AUC | Observed Diff | 95% Bootstrap CI | Resample Fraction (RGCN Better) | Conclusion |
|---|---|---|---|---|---|---|
| **Strict Env + Lag Logistic Regression** | {rgcn_full_ap:.4f} | {lr_full_ap:.4f} | {diff_full_vs_lr:+.4f} | [{ci_lower_lr:+.4f}, {ci_upper_lr:+.4f}] | {resample_frac_lr:.1%} ({resample_frac_lr:.4f}) | **{decision_lr}** |
| **kNN-3 GraphSAGE Ensemble** | {rgcn_full_ap:.4f} | {knn3_full_ap:.4f} | {diff_full_vs_knn3:+.4f} | [{ci_lower_knn3:+.4f}, {ci_upper_knn3:+.4f}] | {resample_frac_knn3:.1%} ({resample_frac_knn3:.4f}) | **{decision_knn3}** |

## Statistical Decision Interpretations

### 1. Relational GNN vs. Strict Environment + Lag Logistic Regression
- **Observed Difference**: {diff_full_vs_lr:+.4f} ({rgcn_full_ap:.4f} vs {lr_full_ap:.4f})
- **95% Confidence Interval**: [{ci_lower_lr:+.4f}, {ci_upper_lr:+.4f}]
- **Conclusion**: {"The 95% confidence interval contains zero; therefore, the Relational GNN is comparable, not proven better than strict Logistic Regression." if "comparable" in decision_lr else "The Relational GNN demonstrates statistically significant superiority over strict Logistic Regression."}
- **Resample Fraction**: In {resample_frac_lr:.1%} of bootstrap resamples, the Relational GNN achieved a higher PR-AUC than Logistic Regression.

### 2. Relational GNN vs. kNN-3 GraphSAGE Ensemble
- **Observed Difference**: {diff_full_vs_knn3:+.4f} ({rgcn_full_ap:.4f} vs {knn3_full_ap:.4f})
- **95% Confidence Interval**: [{ci_lower_knn3:+.4f}, {ci_upper_knn3:+.4f}]
- **Conclusion**: {"The 95% confidence interval contains zero; therefore, the Relational GNN is comparable, not proven better than the kNN-3 GraphSAGE ensemble." if "comparable" in decision_knn3 else "The Relational GNN demonstrates statistically significant superiority over the kNN-3 GraphSAGE ensemble."}
- **Resample Fraction**: In {resample_frac_knn3:.1%} of bootstrap resamples, the Relational GNN achieved a higher PR-AUC than kNN-3 GraphSAGE.

---
*Report generated automatically by `scripts/compare_relational_gnn_vs_baselines_bootstrap.py`.*
"""
    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print("\n==========================================================================================")
    print("                     CLUSTER-BOOTSTRAP UNCERTAINTY RESULTS")
    print("==========================================================================================")
    print(df_res.to_string(index=False))
    print(f"\nSaved CSV results to: {CSV_OUT}")
    print(f"Saved MD report to:  {MD_OUT}")

if __name__ == '__main__':
    main()
