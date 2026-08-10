import pandas as pd, numpy as np, os
from sklearn.metrics import average_precision_score

# Paths
baseline_path = r'd:/gdrive/models/hpai-heterogeneous-gnn-canada/results/baselines/environment_with_lags_v1_strict_test_predictions.csv'
wb_lr_path = r'd:/gdrive/models/hpai-heterogeneous-gnn-canada/results/baselines/wildbird_lr_strict_test_predictions.csv'

# Load
base = pd.read_csv(baseline_path)
wb = pd.read_csv(wb_lr_path)
# Ensure alignment columns
cols = ['csd_uid', 'year', 'month']
base = base[cols + ['actual', 'strict_env_lag_logistic_prob']]
wb = wb[cols + ['actual', 'wildbird_lr_prob']]
# Merge to assure same ordering
merged = pd.merge(wb, base, on=cols + ['actual'], how='inner')
# Use actual as target, probabilities from both
y = merged['actual'].values
wb_prob = merged['wildbird_lr_prob'].values
bench_prob = merged['strict_env_lag_logistic_prob'].values

# Compute PR-AUCs
pr_auc_wb = average_precision_score(y, wb_prob)
pr_auc_base = average_precision_score(y, bench_prob)

# Bootstrap
unique_csds = merged['csd_uid'].unique()
csd_groups = merged.groupby('csd_uid').indices
n_csds = len(unique_csds)
np.random.seed(42)
N_BOOT = 1000
diffs = []
for _ in range(N_BOOT):
    resampled = np.random.choice(unique_csds, size=n_csds, replace=True)
    idx = np.concatenate([csd_groups[c] for c in resampled])
    yb = y[idx]
    wb_p = wb_prob[idx]
    bench_p = bench_prob[idx]
    diff = average_precision_score(yb, wb_p) - average_precision_score(yb, bench_p)
    diffs.append(diff)
ci_low, ci_up = np.percentile(diffs, [2.5, 97.5])
mean_diff = np.mean(diffs)
resample_win = np.mean(np.array(diffs) > 0) * 100.0
decision = "comparable, not proven better" if (ci_low <= 0 <= ci_up) else ("statistically better" if ci_low > 0 else "statistically worse")

# Output CSV
csv_path = r'd:/gdrive/models/hpai-heterogeneous-gnn-canada/results/baselines/wildbird_lr_vs_strict_lr_corrected_comparison.csv'
pd.DataFrame({
    'baseline_pr_auc': [pr_auc_base],
    'wildbird_pr_auc': [pr_auc_wb],
    'mean_difference': [mean_diff],
    'ci_lower': [ci_low],
    'ci_upper': [ci_up],
    'resample_win_percent': [resample_win],
    'decision': [decision]
}).to_csv(csv_path, index=False)
print('Wrote CSV to', csv_path)

# Markdown summary
md_path = r'd:/gdrive/models/hpai-heterogeneous-gnn-canada/results/baselines/wildbird_lr_vs_strict_lr_corrected_summary.md'
with open(md_path, 'w') as f:
    f.write(f"# Wild‑bird LR vs Strict LR (Corrected)\n\n")
    f.write(f"- Baseline strict LR PR‑AUC: {pr_auc_base:.6f}\n")
    f.write(f"- Wild‑bird LR PR‑AUC: {pr_auc_wb:.6f}\n")
    f.write(f"- Mean PR‑AUC difference (wild‑bird – baseline): {mean_diff:+.6f}\n")
    f.write(f"- 95 % CI of difference: [{ci_low:+.6f}, {ci_up:+.6f}]\n")
    f.write(f"- Resample win >0 %: {resample_win:.1f}\n")
    f.write(f"- Decision: **{decision.upper()}**\n")
print('Wrote markdown to', md_path)
