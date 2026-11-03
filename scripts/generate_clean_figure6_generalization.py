"""
Generate clean, publication-ready Figure 6 (Figure7_generalization_shift.pdf & .png)
Addresses all reviewer/user requirements:
1. Unbolded panel titles: (a) Prospective Generalization Trajectory and (b) Epidemiological Shift (2024 vs. 2025)
2. Removed unnecessary floating text box inside the plot
3. Replaced vague dummy bars with exact, meaningful epidemiological metrics from Section 3.2.4:
   - Outbreak CSDs (Count: 28 in 2024 vs 68 in 2025; +143% expansion)
   - Affected Provinces (Count: 4 in 2024 vs 9 in 2025; +125% spread)
   - Prior Outbreak History (62.5% in 2024 vs 46.8% in 2025; >50% first-time naive CSDs in 2025)
   - Median Poultry Density in positive CSDs (574 birds/km² in 2024 vs 52 birds/km² in 2025; 91% drop, shifting to low-density areas)
4. Saves high-resolution vector PDF and 300 DPI PNG across all target output directories.
"""

import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

OUT_DIRS = [
    'results/manuscript_elsevier',
    'results/manuscript_elsevier/figures',
    'results/final_manuscript',
    'results/final_manuscript/figures',
    'results/manuscript_latex',
    'results/manuscript_latex/figures',
    'results/manuscript_draft/figures/final'
]

def generate_figure():
    print("Generating clean, unbolded Figure 6 (Figure7_generalization_shift)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=300)
    plt.subplots_adjust(top=0.90, wspace=0.28, bottom=0.15, left=0.08, right=0.97)

    # =========================================================================
    # Panel (a): Prospective Generalization Trajectory
    # =========================================================================
    splits = ['Train (2022–23)', 'Val (2024)', 'Test (2025)']
    gs_ap = [0.892, 0.210, 0.111]
    rf_ap = [0.965, 0.145, 0.058]
    xgb_ap = [0.942, 0.138, 0.048]

    x = np.arange(len(splits))
    l1, = ax1.plot(x, gs_ap, 'o-', color='#1d4ed8', linewidth=2.4, markersize=7.5, label='GraphSAGE (Spatial GNN)')
    l2, = ax1.plot(x, rf_ap, 's--', color='#b91c1c', linewidth=2.0, markersize=6.5, label='Random Forest')
    l3, = ax1.plot(x, xgb_ap, '^:', color='#7c3aed', linewidth=1.8, markersize=6.5, label='XGBoost')

    # Annotations for Model AP values
    ax1.annotate("0.892", (x[0], gs_ap[0]), textcoords="offset points", xytext=(-18, -14),
                 ha='center', fontsize=9.5, fontweight='normal', color='#1d4ed8')
    ax1.annotate("0.210", (x[1], gs_ap[1]), textcoords="offset points", xytext=(0, 9),
                 ha='center', fontsize=9.5, fontweight='normal', color='#1d4ed8')
    ax1.annotate("0.111", (x[2], gs_ap[2]), textcoords="offset points", xytext=(0, 9),
                 ha='center', fontsize=9.5, fontweight='normal', color='#1d4ed8')

    ax1.annotate("0.965", (x[0], rf_ap[0]), textcoords="offset points", xytext=(16, 7),
                 ha='left', fontsize=9.5, fontweight='normal', color='#b91c1c')
    ax1.annotate("0.145", (x[1], rf_ap[1]), textcoords="offset points", xytext=(-10, -17),
                 ha='right', fontsize=9.5, fontweight='normal', color='#b91c1c')
    ax1.annotate("0.058", (x[2], rf_ap[2]), textcoords="offset points", xytext=(0, -17),
                 ha='center', fontsize=9.5, fontweight='normal', color='#b91c1c')

    ax1.set_xlim([-0.28, 2.28])
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=10.0, fontweight='normal')
    ax1.set_ylabel('Average Precision (National Domain)', labelpad=6, fontsize=10.5, fontweight='normal')
    ax1.set_title('(a) Prospective Generalization Trajectory', pad=8, fontsize=11.5, fontweight='normal', loc='left', color='#0f172a')
    ax1.set_ylim([-0.05, 1.15])
    ax1.grid(True, linestyle='--', alpha=0.45, color='#cbd5e1')
    ax1.tick_params(labelsize=10.0)
    ax1.legend(handles=[l1, l2, l3], loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=9.0)

    # =========================================================================
    # Panel (b): Epidemiological Shift (2024 vs. 2025)
    # =========================================================================
    categories = ['Outbreak CSDs\n(Count)', 'Affected Prov.\n(Count)', 'Prior History\n(% of CSDs)', 'Median Density\n(Birds/km²)']
    val_norm = [100.0, 100.0, 100.0, 100.0]
    test_norm = [68/28*100, 9/4*100, 46.8/62.5*100, 52/574*100]

    x2 = np.arange(len(categories))
    w2 = 0.35
    b_val = ax2.bar(x2 - w2/2, val_norm, w2, label='2024 Validation Baseline', color='#64748b', alpha=0.85, edgecolor='#1e293b', linewidth=0.8)
    b_test = ax2.bar(x2 + w2/2, test_norm, w2, label='2025 Locked Test Shift', color='#ea580c', alpha=0.88, edgecolor='#7c2d12', linewidth=0.8)

    # Subtle horizontal reference line at 100% baseline
    ax2.axhline(100, color='#94a3b8', linestyle=':', linewidth=1.2, zorder=1)

    val_labels = ['28', '4', '62.5%', '574']
    test_labels = ['68', '9', '46.8%', '52']

    for i in range(len(categories)):
        ax2.annotate(val_labels[i], xy=(x2[i] - w2/2, val_norm[i]),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9.2, fontweight='normal', color='#1e293b')
        ax2.annotate(test_labels[i], xy=(x2[i] + w2/2, test_norm[i]),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9.2, fontweight='normal', color='#9a3412')

    ax2.set_ylabel('Relative Level (% of 2024 Baseline)', labelpad=6, fontsize=10.5, fontweight='normal')
    ax2.set_title('(b) Epidemiological Shift (2024 vs. 2025)', pad=8, fontsize=11.5, fontweight='normal', loc='left', color='#0f172a')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=9.5, fontweight='normal')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.45, color='#cbd5e1')
    ax2.set_ylim(0, 275)
    ax2.tick_params(labelsize=10.0)
    ax2.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=9.0)

    # Save initial copy in scratch
    os.makedirs('scratch', exist_ok=True)
    temp_pdf = 'scratch/Figure7_generalization_shift.pdf'
    temp_png = 'scratch/Figure7_generalization_shift.png'
    fig.savefig(temp_pdf)
    fig.savefig(temp_png, dpi=300)
    plt.close()
    print("Base figure generated in scratch.")

    # Distribute to all target directories with safe copy
    for d in OUT_DIRS:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        dst_pdf = os.path.join(d, 'Figure7_generalization_shift.pdf')
        dst_png = os.path.join(d, 'Figure7_generalization_shift.png')
        for src, dst in [(temp_pdf, dst_pdf), (temp_png, dst_png)]:
            copied = False
            for attempt in range(5):
                try:
                    shutil.copy2(src, dst)
                    copied = True
                    break
                except PermissionError:
                    time.sleep(0.5)
            if copied:
                print(f"Saved: {dst}")
            else:
                print(f"Warning: Could not overwrite {dst} (permission error)")

if __name__ == '__main__':
    generate_figure()
