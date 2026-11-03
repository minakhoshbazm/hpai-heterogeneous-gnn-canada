import os
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT_DIR = '.'
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'modeling', 'model_table_v1_environment_with_lags_wildbird.csv')
PREDS_DIR = os.path.join(ROOT_DIR, 'results', 'final_manuscript', 'canonical_predictions')

df_all = pd.read_csv(DATA_PATH, dtype={'csd_uid': str})
df_test = df_all[df_all['year'] == 2025].copy()

farm_cols = ['ccs_broilers_roasters_cornish_farms_reporting', 'ccs_broilers_roasters_cornish_production_kg', 'ccs_turkey_farms_reporting', 'ccs_turkey_production_kg']
df_all['farm_sum'] = df_all[farm_cols].fillna(0).sum(axis=1)
at_risk_csds = set(df_all.groupby('csd_uid')['farm_sum'].max().loc[lambda x: x > 0].index)
eligible_mask = df_test['csd_uid'].isin(at_risk_csds) | (df_test['outbreak_next_month'] == 1)

df_test_el = df_test[eligible_mask].copy()
df_test_el['csd_uid'] = df_test_el['csd_uid'].astype(str).str.strip()

# Load models
models = {
    'GraphSAGE_62': 'preds_GraphSAGE.csv',
    'MLP_62': 'preds_MLP.csv',
    'GraphSAGE_43_nolag': 'preds_GraphSAGE_nolag.csv',
    'MLP_43_nolag': 'preds_MLP_nolag.csv'
}

prob_col = 'pred_prob_ensemble'
preds = {}

for name, fname in models.items():
    df_m = pd.read_csv(os.path.join(PREDS_DIR, fname), dtype={'csd_uid': str})
    df_m['csd_uid'] = df_m['csd_uid'].astype(str).str.strip()
    m = df_test_el.merge(df_m[['csd_uid', 'year', 'month', prob_col]], on=['csd_uid', 'year', 'month'], how='left')
    preds[name] = m[prob_col].values

y = df_test_el['outbreak_next_month'].values
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
n_pos = len(pos_idx)
n_neg = n_pos * 5

print(f"Test Positives: {n_pos}, Eligible Negatives: {len(neg_idx)}")

# Run 1,000 bootstrap / Monte Carlo iterations of 1:5 sampling
np.random.seed(42)
n_iters = 1000

res = {k: [] for k in models.keys()}
diff_full = []
diff_nolag = []

for i in range(n_iters):
    chosen_neg = np.random.choice(neg_idx, size=n_neg, replace=False)
    sample_idx = np.concatenate([pos_idx, chosen_neg])
    y_s = y[sample_idx]
    
    score_sage_full = average_precision_score(y_s, preds['GraphSAGE_62'][sample_idx])
    score_mlp_full = average_precision_score(y_s, preds['MLP_62'][sample_idx])
    score_sage_nolag = average_precision_score(y_s, preds['GraphSAGE_43_nolag'][sample_idx])
    score_mlp_nolag = average_precision_score(y_s, preds['MLP_43_nolag'][sample_idx])
    
    res['GraphSAGE_62'].append(score_sage_full)
    res['MLP_62'].append(score_mlp_full)
    res['GraphSAGE_43_nolag'].append(score_sage_nolag)
    res['MLP_43_nolag'].append(score_mlp_nolag)
    
    diff_full.append(score_sage_full - score_mlp_full)
    diff_nolag.append(score_sage_nolag - score_mlp_nolag)

diff_full = np.array(diff_full)
diff_nolag = np.array(diff_nolag)

print("\n=== TABLE 3 UNDER 1:5 RATIO (1,000 RESAMPLES) ===")
print("FULL MODEL (62 features):")
print(f"GraphSAGE PR-AUC: {np.mean(res['GraphSAGE_62']):.4f} +/- {np.std(res['GraphSAGE_62']):.4f}")
print(f"Matched MLP PR-AUC: {np.mean(res['MLP_62']):.4f} +/- {np.std(res['MLP_62']):.4f}")
print(f"Delta PR-AUC: +{np.mean(diff_full):.4f}")
print(f"95% CI of Delta: [{np.percentile(diff_full, 2.5):.4f}, {np.percentile(diff_full, 97.5):.4f}]")
print(f"Win Rate: {np.mean(diff_full > 0)*100:.1f}%")

print("\nNO-LAG ABLATION (43 features):")
print(f"GraphSAGE PR-AUC: {np.mean(res['GraphSAGE_43_nolag']):.4f} +/- {np.std(res['GraphSAGE_43_nolag']):.4f}")
print(f"Matched MLP PR-AUC: {np.mean(res['MLP_43_nolag']):.4f} +/- {np.std(res['MLP_43_nolag']):.4f}")
print(f"Delta PR-AUC: +{np.mean(diff_nolag):.4f}")
print(f"95% CI of Delta: [{np.percentile(diff_nolag, 2.5):.4f}, {np.percentile(diff_nolag, 97.5):.4f}]")
print(f"Win Rate: {np.mean(diff_nolag > 0)*100:.1f}%")

# Save table to CSV
df_out = pd.DataFrame([
    {
        'Experiment': 'Full Model',
        'Features': 62,
        'GraphSAGE_PR_AUC_1to5': f"{np.mean(res['GraphSAGE_62']):.4f} +/- {np.std(res['GraphSAGE_62']):.4f}",
        'Matched_MLP_PR_AUC_1to5': f"{np.mean(res['MLP_62']):.4f} +/- {np.std(res['MLP_62']):.4f}",
        'Delta_PR_AUC': f"+{np.mean(diff_full):.4f}",
        'CI_95': f"[{np.percentile(diff_full, 2.5):.4f}, {np.percentile(diff_full, 97.5):.4f}]",
        'Win_Rate': f"{np.mean(diff_full > 0)*100:.1f}%"
    },
    {
        'Experiment': 'No-Lag Ablation',
        'Features': 43,
        'GraphSAGE_PR_AUC_1to5': f"{np.mean(res['GraphSAGE_43_nolag']):.4f} +/- {np.std(res['GraphSAGE_43_nolag']):.4f}",
        'Matched_MLP_PR_AUC_1to5': f"{np.mean(res['MLP_43_nolag']):.4f} +/- {np.std(res['MLP_43_nolag']):.4f}",
        'Delta_PR_AUC': f"+{np.mean(diff_nolag):.4f}",
        'CI_95': f"[{np.percentile(diff_nolag, 2.5):.4f}, {np.percentile(diff_nolag, 97.5):.4f}]",
        'Win_Rate': f"{np.mean(diff_nolag > 0)*100:.1f}%"
    }
])
df_out.to_csv('results/manuscript_draft/tables/Table3_spatial_ablation_1to5.csv', index=False)
df_out.to_csv('results/final_manuscript/tables/Table3_spatial_ablation_1to5.csv', index=False)
print("Saved Table3_spatial_ablation_1to5.csv successfully!")
