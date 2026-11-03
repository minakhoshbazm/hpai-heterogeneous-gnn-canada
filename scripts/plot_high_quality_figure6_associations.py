import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIRS = [
    'results/final_manuscript/figures',
    'results/manuscript_latex/figures',
    'results/manuscript_elsevier/figures'
]

data = [
    {'var': 'Duck Density (Max)', 'domain': 'Host Density', 'or': 31.5, 'ci_low': 12.3, 'ci_high': 80.7, 'p': '<0.001', 'type': 'Risk Factor'},
    {'var': 'Chicken Density (p95)', 'domain': 'Host Density', 'or': 6.6, 'ci_low': 3.7, 'ci_high': 11.7, 'p': '<0.001', 'type': 'Risk Factor'},
    {'var': 'Major Road Length', 'domain': 'Infrastructure', 'or': 1.8, 'ci_low': 1.3, 'ci_high': 2.3, 'p': '<0.001', 'type': 'Risk Factor'},
    {'var': 'Provincial Outbreaks', 'domain': 'Surveillance History', 'or': 1.6, 'ci_low': 1.2, 'ci_high': 2.0, 'p': '<0.001', 'type': 'Risk Factor'},
    {'var': 'Same-CSD Outbreaks', 'domain': 'Surveillance History', 'or': 1.5, 'ci_low': 1.2, 'ci_high': 1.8, 'p': '<0.001', 'type': 'Risk Factor'},
    {'var': 'Distance to Highway', 'domain': 'Infrastructure', 'or': 0.21, 'ci_low': 0.13, 'ci_high': 0.35, 'p': '<0.001', 'type': 'Protective Factor'},
    {'var': 'CSD Geographic Area', 'domain': 'Geography', 'or': 0.03, 'ci_low': 0.01, 'ci_high': 0.08, 'p': '<0.001', 'type': 'Protective Factor'}
]

df = pd.DataFrame(data)

# Sort from lowest OR to highest OR
df = df.sort_values(by='or', ascending=True).reset_index(drop=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11.5,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold'
})

fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)

y_pos = np.arange(len(df))

# Plot reference line OR = 1.0 (No Effect)
ax.axvline(1.0, color='#7f8c8d', linestyle='--', linewidth=2.0, label='Baseline (OR = 1.0, No Effect)')

# Colors: Red for Risk factors, Blue for Protective factors
colors = ['#2980b9' if r['or'] < 1.0 else '#c0392b' for _, r in df.iterrows()]

# Plot error bars (confidence intervals on log scale)
for i, row in df.iterrows():
    c = '#2980b9' if row['or'] < 1.0 else '#c0392b'
    ax.plot([row['ci_low'], row['ci_high']], [i, i], color=c, linewidth=2.8, alpha=0.85)
    ax.scatter(row['or'], i, color=c, s=120, zorder=5, edgecolor='#2c3e50', linewidth=1.5)
    
    # Text label for OR and 95% CI
    label_text = f"OR = {row['or']:.2f}  [{row['ci_low']:.2f}, {row['ci_high']:.1f}]"
    if row['var'] == 'CSD Geographic Area':
        ax.text(row['ci_high'] * 1.35, i, label_text, va='center', ha='left', fontsize=11.0, fontweight='bold', color=c)
    elif row['or'] < 1.0:
        ax.text(row['ci_low'] * 0.82, i, label_text, va='center', ha='right', fontsize=11.0, fontweight='bold', color=c)
    else:
        ax.text(row['ci_high'] * 1.25, i, label_text, va='center', ha='left', fontsize=11.0, fontweight='bold', color=c)

ax.set_xscale('log')
ax.set_xlim([0.0015, 450.0])
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['var']} ({row['domain']})" for _, row in df.iterrows()], fontweight='bold')
ax.tick_params(axis='y', pad=8)
ax.set_xlabel('Standardized Odds Ratio (OR) per 1-SD Increase [Log Scale]', labelpad=10)
ax.set_title('Predictive Associations: Standardized Multivariable Logistic Regression', pad=14)
ax.grid(True, which='both', axis='x', linestyle=':', alpha=0.4, color='#bdc3c7')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#c0392b', marker='o', markersize=9, linewidth=2.5, label='Positive Risk Association (OR > 1.0)'),
    Line2D([0], [0], color='#2980b9', marker='o', markersize=9, linewidth=2.5, label='Protective Spatial Association (OR < 1.0)'),
    Line2D([0], [0], color='#7f8c8d', linestyle='--', linewidth=2.0, label='Baseline OR = 1.0')
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.92, facecolor='#ffffff', edgecolor='#d5dbdb')

plt.tight_layout()

for d in OUT_DIRS:
    fig.savefig(os.path.join(d, 'Figure6_predictive_associations.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(d, 'Figure6_predictive_associations.pdf'), bbox_inches='tight')
    print(f"Saved Figure 6 to {d}")

plt.close()
print("Figure 6 generation complete!")
