import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

ROOT = Path('.')
DATA_PATH = ROOT / "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
PREDS_DIR = ROOT / "results/final_manuscript/canonical_predictions"

# Directories to save the figure
OUT_DIRS = [
    ROOT / "results/final_manuscript",
    ROOT / "results/manuscript_elsevier",
    ROOT / "results/manuscript_elsevier/figures",
    ROOT / "results/manuscript_latex",
    Path("D:/codex_work/figures")
]

# 1. Load Data
df_raw = pd.read_csv(DATA_PATH, dtype={'csd_uid': str})
farm_cols = [
    'ccs_broilers_roasters_cornish_farms_reporting',
    'ccs_broilers_roasters_cornish_production_kg',
    'ccs_turkey_farms_reporting',
    'ccs_turkey_production_kg'
]
df_raw['farm_sum'] = df_raw[farm_cols].fillna(0).sum(axis=1)
at_risk_csds = set(df_raw.groupby('csd_uid')['farm_sum'].max().loc[lambda x: x > 0].index)

df_test = df_raw[df_raw['year'] == 2025].copy()
eligible_mask = df_test['csd_uid'].isin(at_risk_csds) | (df_test['outbreak_next_month'] == 1)
df_test_el = df_test[eligible_mask].copy()
df_test_el['csd_uid'] = df_test_el['csd_uid'].astype(str).str.strip()

# 2. Models Meta (Matching the final table exactly)
models_meta = [
    {
        'key': 'GraphSAGE',
        'display_name': 'GraphSAGE',
        'full_label': 'GraphSAGE (Canonical GNN)',
        'file': 'preds_GraphSAGE.csv',
        'tau': 0.0044,
        'color': '#1d4ed8', # Royal blue
        'ls': '-',
        'lw': 2.8,
        'zorder': 5
    },
    {
        'key': 'Random Forest',
        'display_name': 'Random Forest',
        'full_label': 'Random Forest (Ensemble)',
        'file': 'preds_RF.csv',
        'tau': 0.5063,
        'color': '#b91c1c', # Crimson red
        'ls': '--',
        'lw': 2.4,
        'zorder': 4
    },
    {
        'key': 'MLP',
        'display_name': 'MLP',
        'full_label': 'MLP (Tabular Neural Net)',
        'file': 'preds_MLP.csv',
        'tau': 0.6446,
        'color': '#0d9488', # Deep teal
        'ls': '-.',
        'lw': 2.4,
        'zorder': 3
    },
    {
        'key': 'Logistic Regression',
        'display_name': 'Logistic Reg.',
        'full_label': 'Logistic Regression (Linear)',
        'file': 'preds_LR.csv',
        'tau': 0.7076,
        'color': '#ea580c', # Burnt orange
        'ls': (0, (3, 1, 1, 1)),
        'lw': 2.2,
        'zorder': 2
    },
    {
        'key': 'XGBoost',
        'display_name': 'XGBoost',
        'full_label': 'XGBoost (Gradient Boost)',
        'file': 'preds_XGB.csv',
        'tau': 0.0005,
        'color': '#7c3aed', # Deep violet
        'ls': ':',
        'lw': 2.4,
        'zorder': 1
    }
]

# 3. Load Predictions
preds_data = {}
for m in models_meta:
    df_p = pd.read_csv(PREDS_DIR / m['file'], dtype={'csd_uid': str})
    df_p['csd_uid'] = df_p['csd_uid'].astype(str).str.strip()
    merged = df_test_el.merge(df_p[['csd_uid', 'year', 'month', 'pred_prob_ensemble']], on=['csd_uid', 'year', 'month'], how='left')
    preds_data[m['key']] = merged['pred_prob_ensemble'].values

y_test = df_test_el['outbreak_next_month'].values
pos_idx = np.where(y_test == 1)[0]
neg_idx = np.where(y_test == 0)[0]
N_POS = len(pos_idx) # 79
N_NEG = N_POS * 5   # 395

# 4. 50 Monte Carlo Evaluation Resamples (Calculated directly from predictions)
N_RESAMPLES = 50
np.random.seed(42)

grid_rec = np.linspace(0, 1.0, 300)
grid_fpr = np.linspace(0, 1.0, 300)

results = {
    m['key']: {
        'ap': [], 'roc': [],
        'prec': [], 'rec': [], 'f1': [],
        'pr_curves': [], 'roc_curves': []
    }
    for m in models_meta
}

for i in range(N_RESAMPLES):
    sample_neg = np.random.choice(neg_idx, size=N_NEG, replace=False)
    sample_idx = np.concatenate([pos_idx, sample_neg])
    y_sub = y_test[sample_idx]
    
    for m in models_meta:
        k = m['key']
        scores = preds_data[k][sample_idx]
        
        # AP & ROC-AUC
        results[k]['ap'].append(average_precision_score(y_sub, scores))
        results[k]['roc'].append(roc_auc_score(y_sub, scores))
        
        # PR Curve interpolation directly from empirical resample
        p_c, r_c, _ = precision_recall_curve(y_sub, scores)
        p_interp = np.interp(grid_rec, r_c[::-1], p_c[::-1])
        results[k]['pr_curves'].append(p_interp)
        
        # ROC Curve interpolation directly from empirical resample
        fpr_c, tpr_c, _ = roc_curve(y_sub, scores)
        tpr_interp = np.interp(grid_fpr, fpr_c, tpr_c)
        results[k]['roc_curves'].append(tpr_interp)
        
        # Operational classification metrics at validation threshold
        y_pred = (scores >= m['tau']).astype(int)
        results[k]['prec'].append(precision_score(y_sub, y_pred, zero_division=0))
        results[k]['rec'].append(recall_score(y_sub, y_pred, zero_division=0))
        results[k]['f1'].append(f1_score(y_sub, y_pred, zero_division=0))

# 5. Publication-Grade Plotting
# 5. Publication-Grade Plotting
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10.5,
    'axes.labelsize': 11.5,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12.5,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10.0,
    'ytick.labelsize': 10.0,
    'legend.fontsize': 9.2,
    'figure.titlesize': 13.5,
    'figure.titleweight': 'bold',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.2, 4.3), dpi=300, gridspec_kw={'width_ratios': [1.0, 1.0, 1.28]})
fig.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.25, wspace=0.36)

# ── PANEL A: Prospective Precision-Recall Curves ─────────────────────────────
p0 = N_POS / (N_POS + N_NEG) # 79 / 474 = 0.1667
ax1.axhline(p0, color='#64748b', linestyle=':', linewidth=1.8, label=f'Prevalence ({p0:.3f})')

for m in models_meta:
    k = m['key']
    pr_matrix = np.array(results[k]['pr_curves'])
    mean_pr = np.mean(pr_matrix, axis=0)
    std_pr = np.std(pr_matrix, axis=0)
    
    ap_mean = np.mean(results[k]['ap'])
    
    label = f"{m['display_name']} ({ap_mean:.3f})"
    ax1.plot(grid_rec, mean_pr, label=label, color=m['color'], linestyle=m['ls'], linewidth=m['lw']*0.9, zorder=m['zorder'])
    ax1.fill_between(grid_rec, np.maximum(0, mean_pr - std_pr), np.minimum(1, mean_pr + std_pr),
                     color=m['color'], alpha=0.10, zorder=m['zorder']-0.5)

ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.02])
ax1.set_xlabel('Recall (Sensitivity)', labelpad=6, fontweight='bold', fontsize=11.5)
ax1.set_ylabel('Precision (PPV)', labelpad=6, fontweight='bold', fontsize=11.5)
ax1.set_title("(A) Precision–Recall Curves", pad=10, fontsize=12.5, fontweight='bold')
ax1.tick_params(labelsize=10.0)
ax1.grid(True, linestyle='--', alpha=0.45, color='#cbd5e1')
ax1.legend(loc='lower left', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=9.2)

# ── PANEL B: Receiver Operating Characteristic (ROC) ─────────────────────────
ax2.plot([0, 1], [0, 1], color='#64748b', linestyle=':', linewidth=1.8, label='Chance (0.500)')

for m in models_meta:
    k = m['key']
    roc_matrix = np.array(results[k]['roc_curves'])
    mean_tpr = np.mean(roc_matrix, axis=0)
    std_tpr = np.std(roc_matrix, axis=0)
    
    roc_mean = np.mean(results[k]['roc'])
    
    label = f"{m['display_name']} ({roc_mean:.3f})"
    ax2.plot(grid_fpr, mean_tpr, label=label, color=m['color'], linestyle=m['ls'], linewidth=m['lw']*0.9, zorder=m['zorder'])
    ax2.fill_between(grid_fpr, np.maximum(0, mean_tpr - std_tpr), np.minimum(1, mean_tpr + std_tpr),
                     color=m['color'], alpha=0.10, zorder=m['zorder']-0.5)

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.02])
ax2.set_xlabel('False Positive Rate (1 - Specificity)', labelpad=6, fontweight='bold', fontsize=11.5)
ax2.set_ylabel('True Positive Rate (Sensitivity)', labelpad=6, fontweight='bold', fontsize=11.5)
ax2.set_title("(B) ROC Trajectories", pad=10, fontsize=12.5, fontweight='bold')
ax2.tick_params(labelsize=10.0)
ax2.grid(True, linestyle='--', alpha=0.45, color='#cbd5e1')
ax2.legend(loc='lower right', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=9.2)

# ── PANEL C: Operational Classification Metrics (Precision, Recall, F1) ───────
x = np.arange(len(models_meta))
width = 0.24

prec_means = [np.mean(results[m['key']]['prec']) for m in models_meta]
prec_stds = [np.std(results[m['key']]['prec']) for m in models_meta]

rec_means = [np.mean(results[m['key']]['rec']) for m in models_meta]
rec_stds = [np.std(results[m['key']]['rec']) for m in models_meta]

f1_means = [np.mean(results[m['key']]['f1']) for m in models_meta]
f1_stds = [np.std(results[m['key']]['f1']) for m in models_meta]

color_p = '#0284c7'  # Sky blue
color_r = '#f59e0b'  # Amber
color_f = '#10b981'  # Emerald green

rects_p = ax3.bar(x - width, prec_means, width, yerr=prec_stds, label='Precision',
                  color=color_p, alpha=0.90, edgecolor='#0f172a', linewidth=0.9, capsize=3.0)
rects_r = ax3.bar(x, rec_means, width, yerr=rec_stds, label='Recall',
                  color=color_r, alpha=0.90, edgecolor='#0f172a', linewidth=0.9, capsize=3.0)
rects_f = ax3.bar(x + width, f1_means, width, yerr=f1_stds, label='$F_1$-Score',
                  color=color_f, alpha=0.90, edgecolor='#0f172a', linewidth=0.9, capsize=3.0)

ax3.set_ylabel(r'Metric Value at Validation $\tau^*$', labelpad=6, fontweight='bold', fontsize=11.5)
ax3.set_title(r"(C) Operational Metrics at $\tau^*$", pad=10, fontsize=12.5, fontweight='bold')
ax3.set_xticks(x)
display_names = ['GraphSAGE', 'Random Forest', 'MLP', 'Logistic Reg.', 'XGBoost']
ax3.set_xticklabels(display_names, rotation=22, ha='right', fontweight='bold', fontsize=9.2)
ax3.set_ylim([0.0, 1.15])
ax3.tick_params(labelsize=10.0)
ax3.grid(True, axis='y', linestyle='--', alpha=0.45, color='#cbd5e1')
ax3.legend(loc='upper left', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', ncol=1, fontsize=9.0)

# Save across all targets
for d in OUT_DIRS:
    try:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / 'Figure3_primary_comparison.png', dpi=300, bbox_inches='tight')
        fig.savefig(d / 'Figure3_primary_comparison.pdf', bbox_inches='tight')
        print(f"Saved Figure 3 to {d}")
    except Exception as e:
        print(f"Warning: could not save to {d}: {e}")

plt.close()
print("All figures successfully regenerated!")
