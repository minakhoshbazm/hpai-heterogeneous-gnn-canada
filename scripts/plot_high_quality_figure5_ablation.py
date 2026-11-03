import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = '.'
CSV_PATH = os.path.join(ROOT_DIR, 'results', 'final_manuscript', 'tables', 'Table4_ablation_landscape_1to5.csv')
OUT_DIRS = [
    'results/final_manuscript/figures',
    'results/manuscript_latex/figures',
    'results/manuscript_elsevier/figures'
]

df = pd.read_csv(CSV_PATH)

# Exclude baseline from the delta comparison, calculate delta PR-AUC relative to baseline
baseline_pr = df.loc[df['Ablation_Arm'] == 'Baseline', 'Test_PR_AUC_1to5_Mean'].values[0]
df_plot = df[df['Ablation_Arm'] != 'Baseline'].copy()
df_plot['Delta_PR_AUC'] = df_plot['Test_PR_AUC_1to5_Mean'] - baseline_pr

# Sort by delta
df_plot = df_plot.sort_values(by='Delta_PR_AUC', ascending=True).reset_index(drop=True)

# Category color map
arm_colors = {
    'Class Imbalance': '#e67e22',      # Amber
    'Loss Objective': '#c0392b',       # Crimson/Coral
    'Pretraining': '#27ae60',          # Emerald
    'Temporal Modeling': '#8e44ad',    # Amethyst
    'Augmentation': '#e84393'          # Rose
}
colors = [arm_colors.get(arm, '#7f8c8d') for arm in df_plot['Ablation_Arm']]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11.5,
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold'
})

fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)

y_pos = np.arange(len(df_plot))
bars = ax.barh(y_pos, df_plot['Delta_PR_AUC'], xerr=df_plot['Test_PR_AUC_1to5_Std'], color=colors, height=0.62, alpha=0.90, edgecolor='#2c3e50', capsize=4)

ax.axvline(0, color='#1b365d', linestyle='-', linewidth=2.5, label=f'Canonical GraphSAGE Baseline (PR-AUC = {baseline_pr:.4f})')
ax.axvspan(-0.024, 0.024, color='#1b365d', alpha=0.08, label='Resampling Uncertainty Zone ($\pm 1$ SD = 0.024)')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot['Configuration'], fontweight='bold')
ax.set_xlabel('$\Delta$ PR-AUC Relative to Canonical GraphSAGE (1:5 Benchmark)', labelpad=10)
ax.set_title('Ablation Landscape: Predictive Impact of Increasing Model Complexity', pad=14)
ax.grid(True, axis='x', linestyle='--', alpha=0.4, color='#bdc3c7')
ax.set_xlim([-0.50, 0.10])

# Add text labels on bars with zero overlap
for bar, row in zip(bars, df_plot.itertuples()):
    w = bar.get_width()
    err = row.Test_PR_AUC_1to5_Std
    va = 'center'
    if w < -0.15:
        ax.text(-0.25, bar.get_y() + bar.get_height()/2, f'{w:+.4f} (PR-AUC={row.Test_PR_AUC_1to5_Mean:.4f})', 
                va=va, ha='center', fontsize=10.5, fontweight='bold', color='#ffffff')
    elif w >= 0:
        x_pos = w + err + 0.006
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{w:+.4f}', 
                va=va, ha='left', fontsize=10.5, fontweight='bold', color='#2c3e50')
    else:
        x_pos = w - err - 0.007
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{w:+.4f}', 
                va=va, ha='right', fontsize=10.5, fontweight='bold', color='#2c3e50')

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1b365d', edgecolor='#2c3e50', label='Canonical Baseline (0.7374)'),
    Patch(facecolor='#e67e22', edgecolor='#2c3e50', label='Class Imbalance / Downsampling'),
    Patch(facecolor='#c0392b', edgecolor='#2c3e50', label='Loss Objectives (Weighted/Focal)'),
    Patch(facecolor='#27ae60', edgecolor='#2c3e50', label='Representation Pretraining / Transfer'),
    Patch(facecolor='#8e44ad', edgecolor='#2c3e50', label='Temporal Recurrence (GRU / Attn)'),
    Patch(facecolor='#e84393', edgecolor='#2c3e50', label='Synthetic Augmentation (SMOTE)')
]
ax.legend(handles=legend_elements, loc='lower left', framealpha=0.92, facecolor='#ffffff', edgecolor='#d5dbdb')

plt.tight_layout()

for d in OUT_DIRS:
    fig.savefig(os.path.join(d, 'Figure5_ablation_landscape.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(d, 'Figure5_ablation_landscape.pdf'), bbox_inches='tight')
    print(f"Saved Figure 5 to {d}")

plt.close()
print("Figure 5 generation complete!")
