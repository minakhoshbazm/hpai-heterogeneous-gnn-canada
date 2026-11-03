import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
    'ps.fonttype': 42,
})

def save_fig_all(fig, basename):
    for d in OUT_DIRS:
        p_png = os.path.join(d, f"{basename}.png")
        p_pdf = os.path.join(d, f"{basename}.pdf")
        fig.savefig(p_png, dpi=300, bbox_inches='tight')
        fig.savefig(p_pdf, bbox_inches='tight')

def generate_figure6_flawless():
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

    fig = plt.figure(figsize=(9.5, 4.4), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 2.5], wspace=0.10)
    ax_plot = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    y_pos = np.arange(len(df))

    # --- LEFT PANEL: Forest Plot ---
    ax_plot.axvline(1.0, color='#64748b', linestyle='--', linewidth=1.8, zorder=2)

    for i, row in df.iterrows():
        is_risk = row['or'] > 1.0
        c = '#dc2626' if is_risk else '#2563eb'
        ax_plot.plot([row['ci_low'], row['ci_high']], [i, i], color=c, linewidth=2.8, zorder=4)
        ax_plot.scatter(row['or'], i, color=c, s=85, zorder=5, edgecolor='#0f172a', linewidth=1.2)

    ax_plot.set_xscale('log')
    ax_plot.set_xlim([0.008, 120.0])
    ax_plot.set_ylim([-0.5, len(df) - 0.5])
    ax_plot.set_yticks(y_pos)
    ax_plot.set_yticklabels(df['var'], fontsize=12.0, fontweight='bold', color='#0f172a')
    ax_plot.tick_params(axis='y', pad=6)
    ax_plot.tick_params(axis='x', labelsize=11.5)
    ax_plot.set_xlabel('Adjusted Odds Ratio (1-SD Increase, Log Scale)', fontsize=11.5, fontweight='bold', labelpad=8)
    ax_plot.grid(True, which='major', axis='x', linestyle=':', alpha=0.55, color='#cbd5e1')

    # --- RIGHT PANEL: Aligned Text Table ---
    ax_table.axis('off')
    ax_table.set_ylim([-0.5, len(df) - 0.5])
    ax_table.set_xlim([0, 1])

    # Table Headers
    ax_table.text(0.00, len(df) - 0.25, "Adjusted OR [95% CI]", fontsize=12.0, fontweight='bold', color='#0f172a', va='bottom', ha='left')
    ax_table.text(0.74, len(df) - 0.25, "p-value", fontsize=12.0, fontweight='bold', color='#0f172a', va='bottom', ha='left')

    for i, row in df.iterrows():
        is_risk = row['or'] > 1.0
        c = '#dc2626' if is_risk else '#2563eb'
        if row['ci_high'] >= 10.0:
            val_str = f"{row['or']:.1f}  [{row['ci_low']:.1f}, {row['ci_high']:.1f}]"
        else:
            val_str = f"{row['or']:.2f}  [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        ax_table.text(0.00, i, val_str, fontsize=12.0, fontweight='bold', color=c, va='center', ha='left')
        ax_table.text(0.74, i, row['p'], fontsize=11.5, fontweight='bold', color='#334155', va='center', ha='left')

    # Shared Legend at top
    legend_elements = [
        Patch(facecolor='#dc2626', edgecolor='#0f172a', label='Risk Factor (OR > 1.0)'),
        Patch(facecolor='#2563eb', edgecolor='#0f172a', label='Protective Factor (OR < 1.0)'),
        plt.Line2D([0], [0], color='#64748b', linestyle='--', linewidth=1.8, label='Baseline (OR = 1.0)')
    ]
    ax_plot.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 1.08),
                   ncol=3, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10.5)

    fig.subplots_adjust(left=0.25, right=0.98, top=0.84, bottom=0.14)
    save_fig_all(fig, 'Figure6_predictive_associations')
    plt.close()
    print("Flawless Figure 6 generated!")

if __name__ == '__main__':
    generate_figure6_flawless()
