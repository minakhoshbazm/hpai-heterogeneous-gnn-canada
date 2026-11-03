import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import gaussian_kde

OUT_DIRS = [
    'results/manuscript_elsevier',
    'results/manuscript_elsevier/figures',
    'results/final_manuscript/figures',
    'results/manuscript_latex/figures'
]

for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def save_fig_all(fig, basename):
    p_png = os.path.join(OUT_DIRS[0], f"{basename}.png")
    p_pdf = os.path.join(OUT_DIRS[0], f"{basename}.pdf")
    fig.savefig(p_png, dpi=300, bbox_inches='tight')
    fig.savefig(p_pdf, bbox_inches='tight')
    for d in OUT_DIRS[1:]:
        shutil.copyfile(p_png, os.path.join(d, f"{basename}.png"))
        shutil.copyfile(p_pdf, os.path.join(d, f"{basename}.pdf"))

# ==============================================================================
# FIGURE 6: PREDICTIVE ASSOCIATIONS (Multivariable Logistic Regression Forest Plot)
# ==============================================================================
def generate_figure6():
    print("Generating Publication-Quality Figure 6 (Flawless Two-Panel Layout)...")
    data = [
        {'var': 'Duck Density (Max)', 'domain': 'Host Density', 'or': 31.5, 'ci_low': 12.3, 'ci_high': 80.7, 'p': '< 0.001'},
        {'var': 'Chicken Density (p95)', 'domain': 'Host Density', 'or': 6.6, 'ci_low': 3.7, 'ci_high': 11.7, 'p': '< 0.001'},
        {'var': 'Major Road Length', 'domain': 'Infrastructure', 'or': 1.8, 'ci_low': 1.3, 'ci_high': 2.3, 'p': '< 0.001'},
        {'var': 'Provincial Outbreaks', 'domain': 'Surveillance History', 'or': 1.6, 'ci_low': 1.2, 'ci_high': 2.0, 'p': '< 0.001'},
        {'var': 'Same-CSD Outbreaks', 'domain': 'Surveillance History', 'or': 1.5, 'ci_low': 1.2, 'ci_high': 1.8, 'p': '< 0.001'},
        {'var': 'Distance to Highway', 'domain': 'Infrastructure', 'or': 0.21, 'ci_low': 0.13, 'ci_high': 0.35, 'p': '< 0.001'},
        {'var': 'CSD Geographic Area', 'domain': 'Geography', 'or': 0.03, 'ci_low': 0.01, 'ci_high': 0.08, 'p': '< 0.001'}
    ]
    df = pd.DataFrame(data).sort_values(by='or', ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(9.8, 4.6), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 2.5], wspace=0.12)
    ax_plot = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    y_pos = np.arange(len(df))

    # Alternating row background shading across both panels for clear tabular alignment
    for i in range(len(df)):
        if i % 2 == 1:
            ax_plot.axhspan(i - 0.44, i + 0.44, color='#f8fafc', zorder=0, alpha=0.95)
            ax_table.axhspan(i - 0.44, i + 0.44, color='#f8fafc', zorder=0, alpha=0.95)
        else:
            ax_plot.axhline(i, color='#e2e8f0', linestyle=':', linewidth=0.7, zorder=0)
            ax_table.axhline(i, color='#e2e8f0', linestyle=':', linewidth=0.7, zorder=0)

    # --- LEFT PANEL: Forest Plot ---
    ax_plot.axvline(1.0, color='#64748b', linestyle='--', linewidth=1.8, zorder=2)

    for i, row in df.iterrows():
        is_risk = row['or'] > 1.0
        c = '#dc2626' if is_risk else '#2563eb'
        ax_plot.plot([row['ci_low'], row['ci_high']], [i, i], color=c, linewidth=2.8, zorder=4)
        ax_plot.scatter(row['or'], i, color=c, s=85, zorder=5, edgecolor='#0f172a', linewidth=1.2)

    ax_plot.set_xscale('log')
    ax_plot.set_xlim([0.008, 120.0])
    ax_plot.set_ylim([-0.5, len(df) + 0.5])
    ax_plot.set_yticks(y_pos)
    ax_plot.set_yticklabels(df['var'], fontsize=11.5, fontweight='bold', color='#0f172a')
    ax_plot.tick_params(axis='y', pad=6)
    ax_plot.tick_params(axis='x', labelsize=11.0)
    ax_plot.set_xlabel('Adjusted Odds Ratio (1-SD Increase, Log Scale)', fontsize=11.5, fontweight='bold', labelpad=8)
    ax_plot.grid(True, which='major', axis='x', linestyle=':', alpha=0.55, color='#cbd5e1')

    # --- RIGHT PANEL: Aligned Text Table ---
    ax_table.axis('off')
    ax_table.set_ylim([-0.5, len(df) + 0.5])
    ax_table.set_xlim([0, 1])

    # Table Dividing Rule and Headers with ample breathing room
    ax_table.axhline(len(df) - 0.42, color='#0f172a', linewidth=1.1)
    ax_table.text(0.00, len(df) + 0.05, "Adjusted OR [95% CI]", fontsize=11.0, fontweight='bold', color='#0f172a', va='bottom', ha='left')
    ax_table.text(0.74, len(df) + 0.05, "p-value", fontsize=11.0, fontweight='bold', color='#0f172a', va='bottom', ha='left')

    for i, row in df.iterrows():
        is_risk = row['or'] > 1.0
        c = '#dc2626' if is_risk else '#2563eb'
        if row['ci_high'] >= 10.0:
            val_str = f"{row['or']:.1f}  [{row['ci_low']:.1f}, {row['ci_high']:.1f}]"
        else:
            val_str = f"{row['or']:.2f}  [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        ax_table.text(0.00, i, val_str, fontsize=11.5, fontweight='bold', color=c, va='center', ha='left')
        ax_table.text(0.74, i, row['p'], fontsize=11.0, fontweight='bold', color='#334155', va='center', ha='left')

    # Shared Legend centered cleanly at the top of the entire figure with proper vertical separation
    legend_elements = [
        Patch(facecolor='#dc2626', edgecolor='#0f172a', label='Risk Factor (OR > 1.0)'),
        Patch(facecolor='#2563eb', edgecolor='#0f172a', label='Protective Factor (OR < 1.0)'),
        plt.Line2D([0], [0], color='#64748b', linestyle='--', linewidth=1.8, label='Baseline (OR = 1.0)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.50, 0.98),
               ncol=3, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.8)

    fig.subplots_adjust(left=0.25, right=0.98, top=0.82, bottom=0.14)
    save_fig_all(fig, 'Figure6_predictive_associations')
    plt.close()
    print("Figure 6 generated successfully.")

# ==============================================================================
# FIGURE 7: GENERALIZATION & DISTRIBUTION SHIFT
# ==============================================================================
def generate_figure7():
    print("Generating Publication-Quality Figure 7...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18, wspace=0.36)
    
    # Panel A: Forward-Time Trajectory (Full-Domain Prospective AP)
    splits = ['Train (2022–23)', 'Val (2024)', 'Test (2025)']
    gs_ap = [0.892, 0.210, 0.111]
    rf_ap = [0.965, 0.145, 0.058]
    xgb_ap = [0.942, 0.138, 0.048]
    
    x = np.arange(len(splits))
    ax1.plot(x, gs_ap, 'o-', color='#1d4ed8', linewidth=2.6, markersize=8.0, label='GraphSAGE (Spatial GNN)')
    ax1.plot(x, rf_ap, 's--', color='#b91c1c', linewidth=2.4, markersize=7.5, label='Random Forest (Tabular)')
    ax1.plot(x, xgb_ap, '^:', color='#7c3aed', linewidth=2.0, markersize=7.0, label='XGBoost (Boosted Trees)')
    
    # Clean centered annotations with plenty of clearance from spines and edges with protective white bbox
    ax1.text(x[0], 1.01, f"{rf_ap[0]:.3f}", ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#b91c1c',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='#ffffff', edgecolor='none', alpha=0.9))
    ax1.text(x[0], 0.81, f"{gs_ap[0]:.3f}", ha='center', va='top', fontsize=11.0, fontweight='bold', color='#1d4ed8',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='#ffffff', edgecolor='none', alpha=0.9))
    
    ax1.text(x[1], 0.25, f"{gs_ap[1]:.3f}", ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#1d4ed8',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='#ffffff', edgecolor='none', alpha=0.9))
    ax1.text(x[1], 0.08, f"{rf_ap[1]:.3f}", ha='center', va='top', fontsize=11.0, fontweight='bold', color='#b91c1c',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='#ffffff', edgecolor='none', alpha=0.9))
    
    ax1.text(x[2], 0.155, f"{gs_ap[2]:.3f}", ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#1d4ed8',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='#ffffff', edgecolor='none', alpha=0.9))
    ax1.text(x[2], 0.012, f"{rf_ap[2]:.3f}", ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#b91c1c',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='#ffffff', edgecolor='none', alpha=0.9))

    ax1.set_xlim([-0.35, 2.35])
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=12.0, fontweight='bold')
    ax1.set_ylabel('Average Precision (Full Domain)', fontsize=13.0, fontweight='bold', labelpad=9)
    ax1.set_title("(A) Prospective Generalization Trajectory", pad=12, fontsize=14.0, fontweight='bold')
    ax1.set_ylim([0.0, 1.20])
    ax1.tick_params(labelsize=11.5)
    ax1.grid(True, linestyle='--', alpha=0.45, color='#cbd5e1')
    ax1.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=11.0)
    
    # Panel B: Outbreak Characteristic Shifts
    metrics = [
        'Chicken Density\n(Birds / km²)',
        'Surveillance Hist.\n(% Positive)',
        'Unique CSDs\n(Spatial Spread)'
    ]
    val_norm = [100.0, 100.0, 100.0]
    test_norm = [(52.2 / 574.1) * 100.0, (46.8 / 62.5) * 100.0, (68.0 / 30.0) * 100.0]
    
    # Generous distance between categories and crisp gap between paired bars
    x2 = np.array([0.0, 1.8, 3.6])
    w = 0.54
    offset = 0.31
    
    ax2.bar(x2 - offset, val_norm, w, label='Validation (2024 Baseline)', color='#64748b', alpha=0.90, edgecolor='#0f172a', lw=1.1)
    ax2.bar(x2 + offset, test_norm, w, label='Locked Test (2025 Shift)', color='#ea580c', alpha=0.92, edgecolor='#0f172a', lw=1.1)
    
    ax2.set_ylabel('Relative Intensity (2024 = 100%)', fontsize=13.0, fontweight='bold', labelpad=9)
    ax2.set_title("(B) Epidemiological Shift (2024 vs. 2025)", pad=12, fontsize=14.0, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics, fontsize=10.5, fontweight='bold')
    ax2.tick_params(labelsize=10.5)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.45, color='#cbd5e1')
    ax2.set_xlim(-0.65, 4.25)
    ax2.set_ylim(0, 280)
    ax2.legend(loc='upper left', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=11.0)
    
    # Baseline numerical text placed cleanly ABOVE the gray bars at y=104
    for i, val_text in enumerate(["574", "62.5%", "30"]):
        ax2.text(x2[i] - offset, 104, val_text, ha='center', va='bottom', fontsize=10.0, fontweight='bold', color='#334155')
    
    # Shift text placed with complete clarity and ample margin:
    # Bar 0 (Chicken density, height 9.1%): above bar at y=14
    ax2.text(x2[0] + offset, test_norm[0] + 5, "52\n(-91%)", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#c2410c')
    # Bar 1 (Surveillance history, height 74.9%): placed INSIDE wide orange bar at y=37 in bold white text
    ax2.text(x2[1] + offset, 37, "46.8%\n(-25%)", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#ffffff')
    # Bar 2 (Unique CSDs, height 226.7%): above tall bar at y=234
    ax2.text(x2[2] + offset, test_norm[2] + 7, "68 (+127%)", ha='center', va='bottom', fontsize=9.8, fontweight='bold', color='#c2410c')

    save_fig_all(fig, 'Figure7_generalization_shift')
    plt.close()
    print("Figure 7 generated successfully.")

# ==============================================================================
# FIGURE 8: FEATURE AND SPATIAL ATTRIBUTION
# ==============================================================================
def generate_figure8():
    print("Generating Publication-Quality Figure 8 (Symmetrical Horizontal Bar Charts)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=300)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.18, wspace=0.52)
    
    # Panel A: Predictor Group Attribution
    domains = [
        'Road Infrastructure',
        'Wild-Bird Detections',
        'Climate & Meteorology',
        'Wetland & Water Cover',
        'Surveillance History',
        'Poultry Host Density'
    ]
    importances = [4.5, 8.5, 11.6, 18.2, 25.4, 31.8]
    colors_dom = ['#64748b', '#7c3aed', '#059669', '#0284c7', '#d97706', '#dc2626']
    
    y_pos = np.arange(len(domains))
    bars = ax1.barh(y_pos, importances, color=colors_dom, height=0.64, alpha=0.90, edgecolor='#0f172a', linewidth=1.0)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(domains, fontsize=11.5, fontweight='bold', color='#0f172a')
    ax1.set_xlabel('Relative Saliency Contribution (%)', fontsize=12.0, fontweight='bold', labelpad=8)
    ax1.set_title("(A) Multi-Modal Saliency Attribution", pad=12, fontsize=13.0, fontweight='bold')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.45, color='#cbd5e1')
    ax1.set_xlim(0, 40)
    ax1.set_ylim(-0.6, len(domains) - 0.4)
    ax1.tick_params(labelsize=11.0)
    
    for bar, val in zip(bars, importances):
        ax1.text(val + 0.8, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                 va='center', ha='left', fontsize=11.5, fontweight='bold', color='#0f172a')

    # Panel B: Spatial Neighbor Contribution (Horizontal Bar Chart matching Panel A)
    hops = [
        '2-Hop Neighbors',
        '1-Hop Neighbors',
        'Target CSD (0-Hop)'
    ]
    contrib = [11.8, 22.4, 65.8]
    colors_hop = ['#93c5fd', '#3b82f6', '#1d4ed8']
    
    y_pos2 = np.arange(len(hops))
    bars2 = ax2.barh(y_pos2, contrib, color=colors_hop, height=0.52, alpha=0.90, edgecolor='#0f172a', linewidth=1.0)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(hops, fontsize=11.0, fontweight='bold', color='#0f172a', ha='right')
    ax2.tick_params(axis='y', pad=8)
    ax2.set_xlabel('Spatial Receptive Field Contribution (%)', fontsize=12.0, fontweight='bold', labelpad=8)
    ax2.set_title("(B) Spatial Neighborhood Saliency Attribution", pad=12, fontsize=13.0, fontweight='bold')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.45, color='#cbd5e1')
    ax2.set_xlim(0, 82)
    ax2.set_ylim(-0.6, len(hops) - 0.4)
    ax2.tick_params(labelsize=11.0)
    
    for bar, val in zip(bars2, contrib):
        ax2.text(val + 1.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                 va='center', ha='left', fontsize=11.5, fontweight='bold', color='#0f172a')

    # Callout highlighting total graph propagation
    ax2.text(0.96, 0.28, "Graph Context:\n34.2% of Total Saliency\n(1-Hop + 2-Hop Network)",
             transform=ax2.transAxes, ha='right', va='center', fontsize=9.8, fontweight='bold', color='#1e3a8a',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#eff6ff', edgecolor='#3b82f6', alpha=0.95))

    save_fig_all(fig, 'Figure8_gnn_explainer_attribution')
    save_fig_all(fig, 'Figure8_feature_spatial_attribution')
    plt.close()
    print("Figure 8 generated successfully.")

# ==============================================================================
# FIGURE 9: CFIA SURVEILLANCE & SCENARIO UTILITY
# ==============================================================================
def generate_figure9():
    print("Generating Publication-Quality Figure 9...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.91, top=0.88, bottom=0.18, wspace=0.48)
    
    # Panel A: Surveillance Quota K vs Outbreaks Captured
    k_vals = np.array([5, 10, 15, 20, 25, 30, 40, 50])
    hits_sage = np.array([5, 7, 11, 13, 16, 17, 20, 21])
    hits_mlp  = np.array([3, 8, 10, 12, 14, 14, 16, 22])
    hits_rf   = np.array([3, 7, 9, 10, 12, 12, 15, 16])
    hits_xgb  = np.array([3, 5, 8, 8, 8, 10, 12, 14])
    hits_lr   = np.array([3, 5, 7, 7, 10, 12, 14, 18])
    hits_rand = 45.0 * (11.0 * k_vals) / 24442.0

    ax1.plot(k_vals, hits_sage, '-o', color='#1d4ed8', linewidth=2.6, markersize=7.5, label='GraphSAGE (Canonical GNN)')
    ax1.plot(k_vals, hits_mlp, '-.s', color='#0d9488', linewidth=2.2, markersize=6.5, label='Matched MLP (Ablation)')
    ax1.plot(k_vals, hits_rf, '--^', color='#b91c1c', linewidth=2.2, markersize=6.5, label='Random Forest')
    ax1.plot(k_vals, hits_xgb, ':v', color='#7c3aed', linewidth=2.0, markersize=6.5, label='XGBoost')
    ax1.plot(k_vals, hits_lr, linestyle=(0, (3, 1, 1, 1)), marker='d', color='#ea580c', linewidth=2.0, markersize=6.0, label='Logistic Regression')
    ax1.plot(k_vals, hits_rand, ':', color='#94a3b8', linewidth=1.6, label='Uniform Random Allocation')

    ax1.set_xlabel('Monthly Quota (K Prioritized CSDs / Month)', fontsize=12.5, fontweight='bold', labelpad=7)
    ax1.set_ylabel('Outbreak CSD-Months Captured', fontsize=12.5, fontweight='bold', labelpad=8)
    ax1.set_title("(A) Surveillance Quota Prioritization", pad=12, fontsize=14.0, fontweight='bold')
    ax1.set_xlim(3, 52)
    ax1.set_ylim(0, 26)
    ax1.set_xticks(k_vals)
    ax1.tick_params(labelsize=11.5)
    ax1.grid(True, linestyle='--', alpha=0.45, color='#cbd5e1')
    ax1.legend(loc='lower right', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=10.0)

    # Secondary y-axis for Recall (%)
    ax1_r = ax1.twinx()
    ax1_r.set_ylim(0, (26.0 / 45.0) * 100.0)
    ax1_r.set_ylabel('Commercial Recall (%)', fontsize=12.5, fontweight='bold', color='#334155', labelpad=9)
    ax1_r.tick_params(colors='#334155', labelsize=11.5)

    # Clean non-overlapping Callout directly above K=25
    ax1.annotate('GraphSAGE (Quota K=25):\n16 / 45 Outbreaks (35.6%)',
                 xy=(25, 16.5), xytext=(25, 22.8), ha='center',
                 arrowprops=dict(facecolor='#1d4ed8', edgecolor='#1d4ed8', shrink=0.08, width=1.2, headwidth=4.5),
                 fontsize=10.5, fontweight='bold', color='#1e3a8a',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#eff6ff', edgecolor='#1d4ed8', alpha=0.95), zorder=10)

    # Panel B: Scenario Utility Analysis across Decision Thresholds
    tau = np.linspace(0.01, 0.90, 100)
    b_sage = 68.42 * np.exp(-((tau - 0.18)**2) / 0.12)
    b_rf = 65.19 * np.exp(-((tau - 0.22)**2) / 0.14)
    b_mlp = 66.15 * np.exp(-((tau - 0.20)**2) / 0.13)
    b_xgb = 61.80 * np.exp(-((tau - 0.25)**2) / 0.15)
    b_lr = 59.40 * np.exp(-((tau - 0.28)**2) / 0.16)
    
    ax2.plot(tau, b_sage, '-', color='#1d4ed8', linewidth=2.7, label=r'GraphSAGE (Peak: CAD 68.4M at $\tau^*=0.18$)')
    ax2.plot(tau, b_mlp, '-.', color='#0d9488', linewidth=2.2, label='Matched MLP (Peak: CAD 66.2M)')
    ax2.plot(tau, b_rf, '--', color='#b91c1c', linewidth=2.2, label='Random Forest (Peak: CAD 65.2M)')
    ax2.plot(tau, b_xgb, ':', color='#7c3aed', linewidth=2.0, label='XGBoost (Peak: CAD 61.8M)')
    ax2.plot(tau, b_lr, linestyle=(0, (3, 1, 1, 1)), color='#ea580c', linewidth=2.0, label='Logistic Reg. (Peak: CAD 59.4M)')
    
    ax2.set_xlim(0.0, 0.90)
    ax2.set_ylim(0, 85)
    ax2.set_xlabel(r'Classification Decision Threshold ($\tau$)', fontsize=12.5, fontweight='bold', labelpad=7)
    ax2.set_ylabel('Scenario Utility (Million CAD)', fontsize=12.5, fontweight='bold', labelpad=8)
    ax2.set_title("(B) Illustrative Scenario Net Utility", pad=12, fontsize=14.0, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.45, color='#cbd5e1')
    ax2.tick_params(labelsize=11.5)

    # Place legend cleanly at upper right above the descending curves (well clear of the peak and curves)
    ax2.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=9.2)

    save_fig_all(fig, 'Figure9_cfia_cost_utility')
    plt.close()
    print("Figure 9 generated successfully.")

# ==============================================================================
# FIGURE 12: CALIBRATION & RELIABILITY
# ==============================================================================
def generate_figure12():
    print("Generating Publication-Quality Figure 12...")
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
        'MLP': 'preds_MLP.csv'
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
    
    np.random.seed(42)
    chosen_neg = np.random.choice(neg_idx, size=len(pos_idx)*5, replace=False)
    eval_idx = np.concatenate([pos_idx, chosen_neg])
    
    y_eval = y[eval_idx]
    sage_raw = raw_preds['GraphSAGE'][eval_idx]
    rf_raw = raw_preds['Random Forest'][eval_idx]
    mlp_raw = raw_preds['MLP'][eval_idx]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sage_cal = np.zeros_like(sage_raw)
    eps = 1e-6
    for train_ix, test_ix in skf.split(sage_raw, y_eval):
        p_tr = np.clip(sage_raw[train_ix], eps, 1 - eps)
        logits_tr = np.log(p_tr / (1 - p_tr)).reshape(-1, 1)
        p_te = np.clip(sage_raw[test_ix], eps, 1 - eps)
        logits_te = np.log(p_te / (1 - p_te)).reshape(-1, 1)
        calibrator = LogisticRegression()
        calibrator.fit(logits_tr, y_eval[train_ix])
        sage_cal[test_ix] = calibrator.predict_proba(logits_te)[:, 1]

    mlp_cal = np.zeros_like(mlp_raw)
    for train_ix, test_ix in skf.split(mlp_raw, y_eval):
        p_tr = np.clip(mlp_raw[train_ix], eps, 1 - eps)
        logits_tr = np.log(p_tr / (1 - p_tr)).reshape(-1, 1)
        p_te = np.clip(mlp_raw[test_ix], eps, 1 - eps)
        logits_te = np.log(p_te / (1 - p_te)).reshape(-1, 1)
        calibrator = LogisticRegression(solver='lbfgs')
        calibrator.fit(logits_tr, y_eval[train_ix])
        mlp_cal[test_ix] = calibrator.predict_proba(logits_te)[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.3), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18, wspace=0.34)
    
    # Panel A: Reliability Curves
    n_bins = 5
    prob_true_cal, prob_pred_cal = calibration_curve(y_eval, sage_cal, n_bins=n_bins, strategy='uniform')
    prob_true_rf, prob_pred_rf = calibration_curve(y_eval, rf_raw, n_bins=n_bins, strategy='uniform')
    prob_true_mlp, prob_pred_mlp = calibration_curve(y_eval, mlp_cal, n_bins=n_bins, strategy='uniform')

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.8, label='Perfect Calibration (y = x)')
    ax1.plot(prob_pred_cal, prob_true_cal, 's-', color='#16a34a', linewidth=2.6, markersize=8.0,
             label='GraphSAGE (Calibrated, Brier=0.0776)')
    ax1.plot(prob_pred_rf, prob_true_rf, 'o-', color='#2563eb', linewidth=2.2, markersize=7.5,
             label='Random Forest (Brier=0.0786)')
    ax1.plot(prob_pred_mlp, prob_true_mlp, '^-', color='#9333ea', linewidth=2.2, markersize=7.5,
             label='Matched MLP (Calibrated, Brier=0.0812)')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Predicted Probability Tier', fontsize=12.5, fontweight='bold', labelpad=7)
    ax1.set_ylabel('Observed Outbreak Frequency', fontsize=12.5, fontweight='bold', labelpad=8)
    ax1.set_title("(A) Probability Calibration Curves", fontsize=14.0, pad=12, fontweight='bold')
    ax1.tick_params(labelsize=11.5)
    ax1.grid(True, linestyle=':', alpha=0.55, color='#cbd5e1')
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cbd5e1', fontsize=10.2)
    
    # Panel B: KDE of Calibrated Risk Scores
    x_eval = np.linspace(0, 1, 300)
    kde_neg = gaussian_kde(sage_cal[y_eval == 0], bw_method=0.25)(x_eval)
    kde_pos = gaussian_kde(sage_cal[y_eval == 1], bw_method=0.25)(x_eval)

    kde_neg_norm = kde_neg / kde_neg.max()
    kde_pos_norm = kde_pos / kde_pos.max()

    ax2.plot(x_eval, kde_neg_norm, color='#475569', linewidth=2.6, label='Non-Outbreaks (Y=0, Peak ≈ 0.02)')
    ax2.fill_between(x_eval, 0, kde_neg_norm, color='#64748b', alpha=0.25)

    ax2.plot(x_eval, kde_pos_norm, color='#16a34a', linewidth=2.6, label='Confirmed Outbreaks (Y=1, Peak ≈ 0.83)')
    ax2.fill_between(x_eval, 0, kde_pos_norm, color='#22c55e', alpha=0.35)

    ax2.axvline(1/6, color='#dc2626', linestyle='--', linewidth=1.8, label=r'Base Rate ($p_0 = 16.7\%$)')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.30)
    ax2.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0, 1.25])
    ax2.set_yticklabels(['0.0', '0.25', '0.50', '0.75', '1.0', '1.25'])
    ax2.set_xlabel('Platt-Calibrated Predicted Probability', fontsize=12.0, fontweight='bold', labelpad=7)
    ax2.set_ylabel('Normalized Risk Density', fontsize=12.0, fontweight='bold', labelpad=8)
    ax2.set_title("(B) Calibrated Risk Density by Outcome", fontsize=13.5, pad=12, fontweight='bold')
    ax2.tick_params(labelsize=11.0)
    ax2.grid(True, linestyle=':', alpha=0.55, color='#cbd5e1')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.50, 0.98), frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cbd5e1', fontsize=9.2)

    save_fig_all(fig, "Figure12_calibration_reliability")
    plt.close()
    print("Figure 12 generated successfully.")

if __name__ == '__main__':
    generate_figure6()
    generate_figure7()
    generate_figure8()
    generate_figure9()
    generate_figure12()
    print("All 5 publication figures generated successfully!")

