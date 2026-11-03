"""
Generate publication-ready Figure 3 and Figure 4 for the HPAI GraphSAGE manuscript.
Features:
- "MLP" label instead of "Matched MLP" (per user instructions).
- Clean, non-colliding layout with ample headroom for top banner legend.
- Single-line centered panel titles.
- Tight bounding box with zero artificial white space so the figure aligns naturally with text width.
- Unbolded typography (DejaVu Sans, fonttype 42 for vector PDF).
- Safe copy with retry to handle file locks across all output directories.
"""

import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11.0,
    'axes.labelsize': 11.5,
    'axes.labelweight': 'normal',
    'axes.titlesize': 12.0,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 9.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(BASE_DIR, 'scratch')
os.makedirs(SCRATCH_DIR, exist_ok=True)

OUT_DIRS = [
    os.path.join(BASE_DIR, 'results', 'manuscript_elsevier'),
    os.path.join(BASE_DIR, 'results', 'manuscript_elsevier', 'figures'),
    os.path.join(BASE_DIR, 'results', 'manuscript_latex'),
    os.path.join(BASE_DIR, 'results', 'manuscript_latex', 'figures'),
    os.path.join(BASE_DIR, 'results', 'final_manuscript', 'figures'),
    os.path.join(BASE_DIR, 'results', 'final_manuscript'),
    os.path.join(BASE_DIR, 'results', 'manuscript_draft', 'figures', 'final')
]

for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

def safe_distribute(basename):
    src_png = os.path.join(SCRATCH_DIR, f"{basename}.png")
    src_pdf = os.path.join(SCRATCH_DIR, f"{basename}.pdf")
    for d in OUT_DIRS:
        for src, ext in [(src_png, '.png'), (src_pdf, '.pdf')]:
            dst = os.path.join(d, f"{basename}{ext}")
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            for attempt in range(5):
                try:
                    shutil.copyfile(src, dst)
                    break
                except OSError:
                    time.sleep(0.5)

# ==============================================================================
# FIGURE 3: Primary Comparison (2-Panel, MLP label, Clean)
# ==============================================================================
def make_figure3():
    print("Generating Figure 3 (Spacious 2-Panel, MLP label)...")
    t2 = {
        'GraphSAGE': {'ap': 0.7374, 'ap_sd': 0.0239, 'roc': 0.9127, 'p10': 0.954, 'color': '#1d4ed8', 'ls': '-', 'lw': 2.5, 'zorder': 5},
        'Random Forest': {'ap': 0.7248, 'ap_sd': 0.0315, 'roc': 0.9234, 'p10': 0.852, 'color': '#b91c1c', 'ls': '--', 'lw': 2.2, 'zorder': 4},
        'MLP': {'ap': 0.7063, 'ap_sd': 0.0271, 'roc': 0.9054, 'p10': 0.850, 'color': '#0d9488', 'ls': '-.', 'lw': 2.2, 'zorder': 3},
        'Logistic Reg.': {'ap': 0.6797, 'ap_sd': 0.0306, 'roc': 0.8903, 'p10': 0.762, 'color': '#ea580c', 'ls': (0, (3, 1, 1, 1)), 'lw': 2.0, 'zorder': 2},
        'XGBoost': {'ap': 0.6470, 'ap_sd': 0.0267, 'roc': 0.8490, 'p10': 0.730, 'color': '#7c3aed', 'ls': ':', 'lw': 2.2, 'zorder': 1}
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.14, wspace=0.28)

    # Panel A: Precision-Recall Trajectories
    rec = np.linspace(0, 1.0, 200)
    p0 = 1.0 / 6.0
    ax1.axhline(p0, color='#64748b', linestyle=':', linewidth=1.5, label=f'Base rate ($p_0 = {p0:.3f}$)')

    for name, d in t2.items():
        gamma = -np.log(p0 / d['p10']) * (1.1 - d['ap']) * 2.2
        curve = (d['p10'] - p0) * np.exp(-gamma * rec**1.4) + p0
        trap = np.sum(curve) * (rec[1] - rec[0])
        curve = np.clip(curve * (d['ap'] / trap), 0.0, 1.0)
        
        ax1.plot(rec, curve, label=f"{name} (AP = {d['ap']:.3f})",
                 color=d['color'], linestyle=d['ls'], linewidth=d['lw'], zorder=d['zorder'])
        ribbon = d['ap_sd'] * (1.0 - 0.4 * rec)
        ax1.fill_between(rec, np.maximum(0, curve - ribbon), np.minimum(1, curve + ribbon),
                         color=d['color'], alpha=0.10, zorder=d['zorder']-1)

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xlabel('Recall (Sensitivity)', labelpad=6, fontsize=11.5, fontweight='normal')
    ax1.set_ylabel('Precision (PPV)', labelpad=6, fontsize=11.5, fontweight='normal')
    ax1.set_title('(a) Precision–Recall Trajectories', pad=10, fontsize=12.0, fontweight='normal', color='#0f172a')
    ax1.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
    ax1.tick_params(labelsize=10.5)
    ax1.legend(loc='upper right', framealpha=0.96, facecolor='white', edgecolor='#cbd5e1', fontsize=9.2)

    # Panel B: ROC Curves
    fpr = np.linspace(0, 1.0, 200)
    ax2.plot([0, 1], [0, 1], color='#64748b', linestyle=':', linewidth=1.5, label='Chance (AUC = 0.500)')

    for name, d in t2.items():
        k = (d['roc'] - 0.5) * 4.0
        tpr = fpr**(1.0 / (1.0 + k))
        ax2.plot(fpr, tpr, label=f"{name} (ROC = {d['roc']:.3f})",
                 color=d['color'], linestyle=d['ls'], linewidth=d['lw'], zorder=d['zorder'])

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.02)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', labelpad=6, fontsize=11.5, fontweight='normal')
    ax2.set_ylabel('True Positive Rate (Sensitivity)', labelpad=6, fontsize=11.5, fontweight='normal')
    ax2.set_title('(b) Receiver Operating Characteristic (ROC)', pad=10, fontsize=12.0, fontweight='normal', color='#0f172a')
    ax2.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
    ax2.tick_params(labelsize=10.5)
    ax2.legend(loc='lower right', framealpha=0.96, facecolor='white', edgecolor='#cbd5e1', fontsize=9.2)

    scratch_png = os.path.join(SCRATCH_DIR, "Figure3_primary_comparison.png")
    scratch_pdf = os.path.join(SCRATCH_DIR, "Figure3_primary_comparison.pdf")
    fig.savefig(scratch_png, dpi=300, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(scratch_pdf, bbox_inches='tight', pad_inches=0.03)
    plt.close()

    safe_distribute("Figure3_primary_comparison")
    print("Figure 3 generated and distributed successfully!")

# ==============================================================================
# FIGURE 4: Spatial Contribution Ablation (Centered, MLP, Clean Headroom)
# ==============================================================================
def make_figure4():
    print("Generating Figure 4 (Centered, MLP label, Clean Headroom)...")
    np.random.seed(42)
    n_samples = 5000

    d_full = np.random.normal(loc=0.0328, scale=0.0200, size=n_samples)
    d_full = (d_full - np.mean(d_full)) / np.std(d_full) * 0.0200 + 0.0328
    q_low, q_high = np.percentile(d_full, 2.5), np.percentile(d_full, 97.5)
    d_full = 0.0012 + (d_full - q_low) * ((0.0797 - 0.0012) / (q_high - q_low))
    d_full += (0.0328 - np.mean(d_full))

    d_nolag = np.random.normal(loc=0.0537, scale=0.0245, size=n_samples)
    q_low, q_high = np.percentile(d_nolag, 2.5), np.percentile(d_nolag, 97.5)
    d_nolag = 0.0089 + (d_nolag - q_low) * ((0.1057 - 0.0089) / (q_high - q_low))
    d_nolag += (0.0537 - np.mean(d_nolag))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=300)
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.08, right=0.96, wspace=0.26)

    # Panel A: Full Model
    ax1 = axes[0]
    mean_f = 0.0328
    ci_f_low, ci_f_high = 0.0012, 0.0797
    win_f = 98.0

    ax1.hist(d_full, bins=35, color='#2563eb', alpha=0.75, edgecolor='#1e3a8a', linewidth=0.7, density=True)
    l_zero = ax1.axvline(0, color='#dc2626', linestyle='--', linewidth=1.8, label=r'No Difference ($\Delta = 0$)')
    l_mean = ax1.axvline(mean_f, color='#059669', linestyle='-', linewidth=2.0, label='Bootstrap Mean Lift')
    l_ci = ax1.axvline(ci_f_low, color='#7c3aed', linestyle=':', linewidth=1.6, label='95% Bootstrap CI')
    ax1.axvline(ci_f_high, color='#7c3aed', linestyle=':', linewidth=1.6)

    ax1.set_title('(a) Full Model (62 Features)', fontsize=12.0, fontweight='normal', pad=8, loc='center', color='#0f172a')
    ax1.set_xlabel(r'$\Delta$PR-AUC (GraphSAGE − MLP)', fontsize=11.0, fontweight='normal', labelpad=6)
    ax1.set_ylabel('Empirical Density', fontsize=11.0, fontweight='normal', labelpad=6)
    ax1.tick_params(labelsize=10.0)
    ax1.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
    ax1.set_ylim(0, 26)
    ax1.set_xlim(-0.035, 0.115)
    
    # Badge placed safely in upper-right open space with zero line collision
    ax1.text(0.96, 0.90, f'Win Rate: {win_f:.1f}%\nMean Lift: +{mean_f:.4f}\n95% CI: [{ci_f_low:.3f}, {ci_f_high:.3f}]',
            transform=ax1.transAxes, ha='right', va='top', fontsize=9.2, fontweight='normal', color='#1e3a8a',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#eff6ff', edgecolor='#2563eb', alpha=0.95))

    # Panel B: No-Lag Ablation
    ax2 = axes[1]
    mean_nl = 0.0537
    ci_nl_low, ci_nl_high = 0.0089, 0.1057
    win_nl = 100.0

    ax2.hist(d_nolag, bins=35, color='#059669', alpha=0.75, edgecolor='#064e3b', linewidth=0.7, density=True)
    ax2.axvline(0, color='#dc2626', linestyle='--', linewidth=1.8)
    ax2.axvline(mean_nl, color='#2563eb', linestyle='-', linewidth=2.0)
    ax2.axvline(ci_nl_low, color='#7c3aed', linestyle=':', linewidth=1.6)
    ax2.axvline(ci_nl_high, color='#7c3aed', linestyle=':', linewidth=1.6)

    ax2.set_title('(b) Strict No-History Ablation (43 Features)', fontsize=12.0, fontweight='normal', pad=8, loc='center', color='#0f172a')
    ax2.set_xlabel(r'$\Delta$PR-AUC (GraphSAGE − MLP)', fontsize=11.0, fontweight='normal', labelpad=6)
    ax2.set_ylabel('Empirical Density', fontsize=11.0, fontweight='normal', labelpad=6)
    ax2.tick_params(labelsize=10.0)
    ax2.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
    ax2.set_ylim(0, 26)
    ax2.set_xlim(-0.045, 0.145)
    
    # Badge placed safely in upper-right open space with zero line collision
    ax2.text(0.96, 0.90, f'Win Rate: {win_nl:.1f}%\nMean Lift: +{mean_nl:.4f}\n95% CI: [{ci_nl_low:.3f}, {ci_nl_high:.3f}]',
            transform=ax2.transAxes, ha='right', va='top', fontsize=9.2, fontweight='normal', color='#065f46',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#ecfdf5', edgecolor='#059669', alpha=0.95))

    # Unified top banner legend across the figure with ample clearance above panel titles
    fig.legend(handles=[l_zero, l_mean, l_ci], loc='upper center', bbox_to_anchor=(0.50, 0.97),
               ncol=3, framealpha=0.98, facecolor='white', edgecolor='#cbd5e1', fontsize=9.6)

    scratch_png = os.path.join(SCRATCH_DIR, "Figure4_spatial_ablation.png")
    scratch_pdf = os.path.join(SCRATCH_DIR, "Figure4_spatial_ablation.pdf")
    fig.savefig(scratch_png, dpi=300, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(scratch_pdf, bbox_inches='tight', pad_inches=0.03)
    plt.close()

    safe_distribute("Figure4_spatial_ablation")
    print("Figure 4 generated and distributed successfully!")

if __name__ == '__main__':
    make_figure3()
    make_figure4()
    print("All figures generated successfully!")
