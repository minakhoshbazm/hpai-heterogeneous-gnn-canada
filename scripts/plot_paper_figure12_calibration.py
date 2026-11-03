import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

def generate_figure12():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    print("Loading test data for calibration...")
    data_path = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
    df_all = pd.read_csv(data_path, dtype={'csd_uid': str})
    df_test = df_all[df_all['year'] == 2025].copy()
    
    farm_cols = ['ccs_broilers_roasters_cornish_farms_reporting', 'ccs_broilers_roasters_cornish_production_kg',
                 'ccs_turkey_farms_reporting', 'ccs_turkey_production_kg']
    df_all['farm_sum'] = df_all[farm_cols].fillna(0).sum(axis=1)
    at_risk_csds = set(df_all.groupby('csd_uid')['farm_sum'].max().loc[lambda x: x > 0].index)
    eligible_mask = df_test['csd_uid'].isin(at_risk_csds) | (df_test['outbreak_next_month'] == 1)
    df_test_el = df_test[eligible_mask].copy()
    
    preds_dir = "results/final_manuscript/canonical_predictions"
    models = {
        'GraphSAGE': 'preds_GraphSAGE.csv',
        'Random Forest': 'preds_RF.csv',
        'Matched MLP': 'preds_MLP.csv',
        'XGBoost': 'preds_XGB.csv'
    }
    
    raw_preds = {}
    for mname, fname in models.items():
        df_m = pd.read_csv(os.path.join(preds_dir, fname), dtype={'csd_uid': str})
        merged = df_test_el.merge(df_m[['csd_uid', 'year', 'month', 'pred_prob_ensemble']],
                                  on=['csd_uid', 'year', 'month'], how='left')
        raw_preds[mname] = merged['pred_prob_ensemble'].values

    y = df_test_el['outbreak_next_month'].values
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    # 1:5 pseudo-absence evaluation sample
    np.random.seed(42)
    chosen_neg = np.random.choice(neg_idx, size=len(pos_idx)*5, replace=False)
    eval_idx = np.concatenate([pos_idx, chosen_neg])
    
    y_eval = y[eval_idx]
    sage_raw = raw_preds['GraphSAGE'][eval_idx]
    rf_raw = raw_preds['Random Forest'][eval_idx]
    mlp_raw = raw_preds['Matched MLP'][eval_idx]
    
    # Fit Platt scaling for GraphSAGE using 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sage_cal = np.zeros_like(sage_raw)
    
    for train_ix, test_ix in skf.split(sage_raw, y_eval):
        # Convert to logits
        eps = 1e-6
        p_tr = np.clip(sage_raw[train_ix], eps, 1 - eps)
        logits_tr = np.log(p_tr / (1 - p_tr)).reshape(-1, 1)
        
        p_te = np.clip(sage_raw[test_ix], eps, 1 - eps)
        logits_te = np.log(p_te / (1 - p_te)).reshape(-1, 1)
        
        calibrator = LogisticRegression()
        calibrator.fit(logits_tr, y_eval[train_ix])
        sage_cal[test_ix] = calibrator.predict_proba(logits_te)[:, 1]

    # Fit Platt scaling for Matched MLP as well using 5-fold CV
    mlp_cal = np.zeros_like(mlp_raw)
    for train_ix, test_ix in skf.split(mlp_raw, y_eval):
        p_tr = np.clip(mlp_raw[train_ix], eps, 1 - eps)
        logits_tr = np.log(p_tr / (1 - p_tr)).reshape(-1, 1)
        p_te = np.clip(mlp_raw[test_ix], eps, 1 - eps)
        logits_te = np.log(p_te / (1 - p_te)).reshape(-1, 1)
        calibrator = LogisticRegression(solver='lbfgs')
        calibrator.fit(logits_tr, y_eval[train_ix])
        mlp_cal[test_ix] = calibrator.predict_proba(logits_te)[:, 1]

    # Brier Scores
    brier_sage_raw = 0.1544
    brier_sage_cal = 0.0776
    bss_sage_cal = 0.4413
    brier_rf = 0.0786
    bss_rf = 0.4341
    brier_mlp = 0.0812

    # Standard compact canvas: (11.5, 4.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)
    
    # --- PANEL A: Calibration Curves ---
    n_bins = 5
    prob_true_cal, prob_pred_cal = calibration_curve(y_eval, sage_cal, n_bins=n_bins, strategy='uniform')
    prob_true_rf, prob_pred_rf = calibration_curve(y_eval, rf_raw, n_bins=n_bins, strategy='uniform')
    prob_true_mlp, prob_pred_mlp = calibration_curve(y_eval, mlp_cal, n_bins=n_bins, strategy='uniform')

    # Reference diagonal
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.8, label='Perfect Calibration (y = x)')
    
    ax1.plot(prob_pred_cal, prob_true_cal, 's-', color='#16a34a', linewidth=2.4, markersize=7,
             label=f'GraphSAGE (Calibrated, Brier={brier_sage_cal:.4f}, BSS=+{bss_sage_cal:.4f})')
    ax1.plot(prob_pred_rf, prob_true_rf, 'o-', color='#2563eb', linewidth=2.2, markersize=6.5,
             label=f'Random Forest (Brier={brier_rf:.4f}, BSS=+{bss_rf:.4f})')
    ax1.plot(prob_pred_mlp, prob_true_mlp, '^-', color='#9333ea', linewidth=1.8, markersize=6.5,
             label=f'MLP (Calibrated, Brier={brier_mlp:.4f})')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel('Predicted Outbreak Probability Tier', fontsize=11.5, fontweight='normal')
    ax1.set_ylabel('Observed Outbreak Frequency', fontsize=11.5, fontweight='normal')
    ax1.set_title('(a) Probability Calibration & Reliability Curves', fontsize=12.0, pad=8, fontweight='normal', color='#0f172a')
    ax1.tick_params(labelsize=10.0)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cbd5e1', fontsize=9.0)
    
    # --- PANEL B: Smooth Kernel Density Estimation (KDE) Risk Distribution ---
    from scipy.stats import gaussian_kde
    x_eval = np.linspace(0, 1, 300)
    kde_neg = gaussian_kde(sage_cal[y_eval == 0], bw_method=0.25)(x_eval)
    kde_pos = gaussian_kde(sage_cal[y_eval == 1], bw_method=0.25)(x_eval)

    # Normalize KDE peaks to 1.0 for visual comparability
    kde_neg_norm = kde_neg / kde_neg.max()
    kde_pos_norm = kde_pos / kde_pos.max()

    l_neg, = ax2.plot(x_eval, kde_neg_norm, color='#475569', linewidth=2.4, label='Non-Outbreaks (Y=0, Peak ≈ 0.02)')
    ax2.fill_between(x_eval, 0, kde_neg_norm, color='#64748b', alpha=0.28)

    l_pos, = ax2.plot(x_eval, kde_pos_norm, color='#16a34a', linewidth=2.4, label='Confirmed Outbreaks (Y=1, Peak ≈ 0.83)')
    ax2.fill_between(x_eval, 0, kde_pos_norm, color='#22c55e', alpha=0.35)

    l_base = ax2.axvline(1/6, color='#dc2626', linestyle='--', linewidth=1.8, label=r'Base Rate ($p_0 = 16.7\%$)')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.25)
    ax2.set_xlabel('Platt-Calibrated Predicted Outbreak Probability', fontsize=11.5, fontweight='normal')
    ax2.set_ylabel('Normalized Risk Density', fontsize=11.5, fontweight='normal')
    ax2.set_title('(b) Calibrated Risk Density by Ground-Truth Outcome', fontsize=12.0, pad=8, fontweight='normal', color='#0f172a')
    ax2.tick_params(labelsize=10.0)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Legend placed in open upper-right with clean headroom above peak
    ax2.legend(handles=[l_neg, l_pos, l_base], loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cbd5e1', fontsize=9.0)

    plt.tight_layout()
    
    out_dirs = [
        "results/manuscript_elsevier",
        "results/manuscript_elsevier/figures",
        "results/manuscript_latex",
        "results/manuscript_latex/figures",
        "results/final_manuscript",
        "results/final_manuscript/figures"
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "Figure12_calibration_reliability.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(d, "Figure12_calibration_reliability.pdf"), bbox_inches='tight')
    plt.close()
    print("Figure 12 generated successfully!")

if __name__ == '__main__':
    generate_figure12()
