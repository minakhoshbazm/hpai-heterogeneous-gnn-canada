#!/usr/bin/env python3
"""
scripts/audit_2025_wildbird_snapshot_bootstrap_values.py

Audits and clarifies the numerical distinction between:
1. Direct observed point difference: GraphSAGE PR-AUC minus LR PR-AUC (+0.0059)
2. Bootstrap mean difference (+0.0095)
3. Bootstrap 95% percentile confidence interval ([-0.0142, +0.0434])

Inputs:
- results/baselines/wildbird_monthly_snapshot_graphsage_knn3_test_predictions.csv
- results/baselines/wildbird_lr_final_retrained_test_predictions.csv

Outputs (new, never overwritten):
- results/data_qa/wildbird_2025_snapshot_bootstrap_value_audit.csv
- results/data_qa/wildbird_2025_snapshot_bootstrap_value_audit.md
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

GS_PRED_PATH = "results/baselines/wildbird_monthly_snapshot_graphsage_knn3_test_predictions.csv"
LR_PRED_PATH = "results/baselines/wildbird_lr_final_retrained_test_predictions.csv"

OUT_CSV = "results/data_qa/wildbird_2025_snapshot_bootstrap_value_audit.csv"
OUT_MD = "results/data_qa/wildbird_2025_snapshot_bootstrap_value_audit.md"

def main():
    print("=== Auditing 2025 Snapshot Bootstrap Values ===")

    df_gs = pd.read_csv(GS_PRED_PATH)
    df_lr = pd.read_csv(LR_PRED_PATH)

    merge_cols = ["csd_uid", "year", "month"]
    df_merged = df_gs.merge(df_lr, on=merge_cols, suffixes=("_gs", "_lr"))

    y_true = df_merged["actual_gs"].values
    p_gs = df_merged["wildbird_snapshot_gs_ensemble_prob"].values
    p_lr = df_merged["wildbird_lr_final_prob"].values

    # Direct Point Metrics
    gs_pr_auc = average_precision_score(y_true, p_gs)
    lr_pr_auc = average_precision_score(y_true, p_lr)
    direct_diff = gs_pr_auc - lr_pr_auc

    print(f"GraphSAGE Test PR-AUC:  {gs_pr_auc:.6f}")
    print(f"Retrained LR Test PR-AUC: {lr_pr_auc:.6f}")
    print(f"Direct Point Difference:  {direct_diff:+.6f} ({direct_diff:+.4f})")

    # Bootstrap 1,000 Resamples (CSD Cluster Resampling)
    unique_csds = df_merged["csd_uid"].unique()
    n_csds = len(unique_csds)
    csd_groups = df_merged.groupby("csd_uid").indices

    np.random.seed(42)
    N_BOOT = 1000
    boot_diffs = []

    for _ in range(N_BOOT):
        resample_csds = np.random.choice(unique_csds, size=n_csds, replace=True)
        idx = np.concatenate([csd_groups[c] for c in resample_csds])

        y_b = y_true[idx]
        gs_b = p_gs[idx]
        lr_b = p_lr[idx]

        score_gs = average_precision_score(y_b, gs_b)
        score_lr = average_precision_score(y_b, lr_b)
        boot_diffs.append(score_gs - score_lr)

    boot_mean = np.mean(boot_diffs)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    win_rate = np.mean(np.array(boot_diffs) > 0) * 100.0

    print(f"Bootstrap Mean Difference: {boot_mean:+.6f} ({boot_mean:+.4f})")
    print(f"Bootstrap 95% Percentile CI: [{ci_low:+.6f}, {ci_high:+.6f}]")
    print(f"GraphSAGE Win Rate:        {win_rate:.1f}%")

    # Save Audit CSV
    audit_data = [{
        "graphsage_pr_auc": gs_pr_auc,
        "lr_pr_auc": lr_pr_auc,
        "direct_observed_point_diff": direct_diff,
        "bootstrap_mean_diff": boot_mean,
        "ci_lower_95": ci_low,
        "ci_upper_95": ci_high,
        "win_rate_pct": win_rate,
        "explanation": "PR-AUC is a non-linear ranking metric. Cluster resampling with replacement alters the positive-to-negative ratio and ranking density in resampled datasets, introducing expected non-linear estimator bias, making the bootstrap mean difference (+0.0095) slightly higher than the direct point difference (+0.0059)."
    }]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(audit_data).to_csv(OUT_CSV, index=False)
    print(f"Saved audit CSV to {OUT_CSV}")

    # Save Audit Markdown
    md_content = f"""# Audit of 2025 Wild-Bird Snapshot Bootstrap Values

**Date**: 2026-08-03  
**Purpose**: Disambiguate direct observed point difference vs. 1,000 CSD-cluster bootstrap mean difference.

---

## 1. Value Disambiguation Summary

| Metric / Estimation Method | Value | Description |
|---|---|---|
| **GraphSAGE Test PR-AUC** | **{gs_pr_auc:.6f}** (`0.0738`) | Direct out-of-sample PR-AUC on full 2025 test dataset |
| **Retrained LR Test PR-AUC** | **{lr_pr_auc:.6f}** (`0.0679`) | Direct out-of-sample PR-AUC on full 2025 test dataset |
| **Direct Observed Point Difference** | **{direct_diff:+.6f}** (`+0.0059`) | Exact mathematical difference (`0.073793 - 0.067886`) |
| **Bootstrap Mean Difference** | **{boot_mean:+.6f}** (`+0.0095`) | Expected mean difference across 1,000 CSD-cluster resamples |
| **Bootstrap 95% Percentile CI** | **[{ci_low:+.4f}, {ci_high:+.4f}]** | 2.5th to 97.5th percentile of resampled differences |
| **GraphSAGE Win Rate** | **{win_rate:.1f}%** | Percentage of bootstrap resamples where GraphSAGE > LR |

---

## 2. Explanation of Difference between Point Difference and Bootstrap Mean

- **Direct Observed Point Difference (`+0.0059`)**: The exact difference between the two models evaluated on the complete 2025 test set (55,594 CSD-month rows).
- **Bootstrap Mean Difference (`+0.0095`)**: The mean of the 1,000 differences calculated on CSD-cluster resampled datasets (with replacement). 
- **Reason for Divergence**: Precision-Recall AUC is a non-linear, rank-ordered metric. Resampling CSD clusters with replacement creates variation in local positive counts and ranking density across resamples. Because PR-AUC is non-linear, the expected value of resampled PR-AUC differences is not strictly identical to the point estimate on the un-resampled dataset, resulting in a standard, minor non-linear estimator shift.
- **Scientific Significance**: The confidence interval (`[-0.0142, +0.0434]`) includes zero in both cases. Therefore, regardless of whether point difference (`+0.0059`) or bootstrap mean (`+0.0095`) is reported, the statistical decision remains unchanged: **`COMPARABLE, NOT PROVEN BETTER`**.

"""

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Saved audit report to {OUT_MD}")

if __name__ == '__main__':
    main()
