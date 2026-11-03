"""
Generate publication-quality Figures S1-S5 for supplementary material.
Features:
- Completely unbolded typography
- Clear bar values placed cleanly outside bars
- Baseline reference dashed lines with legible labels
- Cohesive color scheme aligned with Figure 5
- Zero occlusions and clear spacing
- Exported directly to all manuscript figure directories
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_DIRS = [
    os.path.join(BASE_DIR, 'results', 'manuscript_elsevier'),
    os.path.join(BASE_DIR, 'results', 'manuscript_elsevier', 'figures'),
    os.path.join(BASE_DIR, 'results', 'manuscript_latex'),
    os.path.join(BASE_DIR, 'results', 'manuscript_latex', 'figures'),
    os.path.join(BASE_DIR, 'results', 'final_manuscript', 'figures'),
    os.path.join(BASE_DIR, 'results', 'manuscript_draft', 'figures', 'final'),
    os.path.join(BASE_DIR, 'scratch')
]

for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

import time

def save_fig(fig, basename):
    scratch_dir = os.path.join(BASE_DIR, 'scratch')
    os.makedirs(scratch_dir, exist_ok=True)
    scratch_png = os.path.join(scratch_dir, f"{basename}.png")
    scratch_pdf = os.path.join(scratch_dir, f"{basename}.pdf")
    fig.savefig(scratch_png, dpi=300, bbox_inches='tight')
    fig.savefig(scratch_pdf, bbox_inches='tight')
    plt.close(fig)
    
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        tgt_png = os.path.join(d, f"{basename}.png")
        tgt_pdf = os.path.join(d, f"{basename}.pdf")
        for src, tgt in [(scratch_png, tgt_png), (scratch_pdf, tgt_pdf)]:
            if os.path.abspath(src) == os.path.abspath(tgt):
                continue
            copied = False
            for attempt in range(5):
                try:
                    shutil.copyfile(src, tgt)
                    copied = True
                    break
                except OSError:
                    time.sleep(0.5)
            if not copied:
                print(f"Warning: could not write {tgt} due to lock")


# ==============================================================================
# FIGURE S1: Negative Sampling Ratios
# ==============================================================================
def make_figure_s1():
    print("Generating Figure S1: Negative Sampling...")
    csv_path = os.path.join(BASE_DIR, 'results', 'manuscript_draft', 'tables', 'final', 'Table5_imbalance_loss_ablation.csv')
    df = pd.read_csv(csv_path)
    df_s1 = df[df['Category'] == 'Negative sampling'].copy()
    
    # Reverse so top configuration is at the top of horizontal bar chart
    df_s1 = df_s1.iloc[::-1].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)
    y_pos = np.arange(len(df_s1))
    
    color = '#d97706'  # Amber
    bars = ax.barh(y_pos, df_s1['Ensemble_Test_AP'], height=0.62, color=color, alpha=0.88,
                   edgecolor='#0f172a', linewidth=0.9)
    
    baseline_val = 0.1107
    ax.axvline(baseline_val, color='#dc2626', linestyle='--', linewidth=1.5)
    ax.text(baseline_val - 0.0015, len(y_pos) - 0.35, f'Baseline B0 ({baseline_val:.4f})',
            color='#dc2626', fontsize=9.5, fontweight='normal', ha='right', va='bottom')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_s1['Configuration'], fontsize=10.5, fontweight='normal')
    ax.tick_params(axis='y', pad=6)
    ax.tick_params(axis='x', labelsize=10.5)
    ax.set_xlabel('National Prospective Test AP', fontsize=11.5, fontweight='normal', labelpad=7)
    ax.set_xlim([0.0, 0.138])
    ax.set_ylim([-0.6, len(y_pos) + 0.5])
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(w - 0.0035, y, f'{w:.4f}', va='center', ha='right', fontsize=9.8, fontweight='normal', color='white')
        
    plt.tight_layout()
    save_fig(fig, "FigureS1_negative_sampling")
    plt.close()
    print("Figure S1 generated successfully!")

# ==============================================================================
# FIGURE S2: Loss Functions
# ==============================================================================
def make_figure_s2():
    print("Generating Figure S2: Loss Functions...")
    csv_path = os.path.join(BASE_DIR, 'results', 'manuscript_draft', 'tables', 'final', 'Table5_imbalance_loss_ablation.csv')
    df = pd.read_csv(csv_path)
    df_s2 = df[df['Category'] == 'Loss function'].copy()
    df_s2 = df_s2.iloc[::-1].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=300)
    y_pos = np.arange(len(df_s2))
    
    color = '#dc2626'  # Crimson
    bars = ax.barh(y_pos, df_s2['Ensemble_Test_AP'], height=0.55, color=color, alpha=0.88,
                   edgecolor='#0f172a', linewidth=0.9)
    
    baseline_val = 0.1107
    ax.axvline(baseline_val, color='#0f172a', linestyle='--', linewidth=1.5)
    ax.text(baseline_val - 0.0015, len(y_pos) - 0.35, f'Baseline BCE ({baseline_val:.4f})',
            color='#0f172a', fontsize=9.5, fontweight='normal', ha='right', va='bottom')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_s2['Configuration'], fontsize=10.5, fontweight='normal')
    ax.tick_params(axis='y', pad=6)
    ax.tick_params(axis='x', labelsize=10.5)
    ax.set_xlabel('National Prospective Test AP', fontsize=11.5, fontweight='normal', labelpad=7)
    ax.set_xlim([0.0, 0.145])
    ax.set_ylim([-0.6, len(y_pos) + 0.5])
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(w - 0.0035, y, f'{w:.4f}', va='center', ha='right', fontsize=9.8, fontweight='normal', color='white')
        
    plt.tight_layout()
    save_fig(fig, "FigureS2_loss_functions")
    plt.close()
    print("Figure S2 generated successfully!")

# ==============================================================================
# FIGURE S3: Representation Learning (Pretraining)
# ==============================================================================
def make_figure_s3():
    print("Generating Figure S3: Representation Learning...")
    csv_path = os.path.join(BASE_DIR, 'results', 'manuscript_draft', 'tables', 'final', 'Table6_representation_learning.csv')
    df_s3 = pd.read_csv(csv_path)
    df_s3 = df_s3.iloc[::-1].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=300)
    y_pos = np.arange(len(df_s3))
    
    color = '#16a34a'  # Green
    bars = ax.barh(y_pos, df_s3['Ensemble_Test_AP'], height=0.55, color=color, alpha=0.88,
                   edgecolor='#0f172a', linewidth=0.9)
    
    baseline_val = 0.1111
    ax.axvline(baseline_val, color='#dc2626', linestyle='--', linewidth=1.5)
    ax.text(baseline_val - 0.0015, len(y_pos) - 0.35, f'Baseline Scratch ({baseline_val:.4f})',
            color='#dc2626', fontsize=9.5, fontweight='normal', ha='right', va='bottom')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_s3['Model'], fontsize=10.5, fontweight='normal')
    ax.tick_params(axis='y', pad=6)
    ax.tick_params(axis='x', labelsize=10.5)
    ax.set_xlabel('National Prospective Test AP', fontsize=11.5, fontweight='normal', labelpad=7)
    ax.set_xlim([0.0, 0.138])
    ax.set_ylim([-0.6, len(y_pos) + 0.5])
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(w - 0.0035, y, f'{w:.4f}', va='center', ha='right', fontsize=9.8, fontweight='normal', color='white')
        
    plt.tight_layout()
    save_fig(fig, "FigureS3_pretraining")
    plt.close()
    print("Figure S3 generated successfully!")

# ==============================================================================
# FIGURE S4: Temporal Modeling
# ==============================================================================
def make_figure_s4():
    print("Generating Figure S4: Temporal Modeling...")
    csv_path = os.path.join(BASE_DIR, 'results', 'manuscript_draft', 'tables', 'final', 'Table7_temporal_ablation.csv')
    df_s4 = pd.read_csv(csv_path)
    df_s4 = df_s4.iloc[::-1].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)
    y_pos = np.arange(len(df_s4))
    
    color = '#7c3aed'  # Purple
    bars = ax.barh(y_pos, df_s4['Ensemble_Test_AP'], height=0.62, color=color, alpha=0.88,
                   edgecolor='#0f172a', linewidth=0.9)
    
    baseline_val = 0.1107
    ax.axvline(baseline_val, color='#dc2626', linestyle='--', linewidth=1.5)
    ax.text(baseline_val - 0.0015, len(y_pos) - 0.35, f'Baseline Snapshot ({baseline_val:.4f})',
            color='#dc2626', fontsize=9.5, fontweight='normal', ha='right', va='bottom')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_s4['Model'], fontsize=10.5, fontweight='normal')
    ax.tick_params(axis='y', pad=6)
    ax.tick_params(axis='x', labelsize=10.5)
    ax.set_xlabel('National Prospective Test AP', fontsize=11.5, fontweight='normal', labelpad=7)
    ax.set_xlim([0.0, 0.138])
    ax.set_ylim([-0.6, len(y_pos) + 0.5])
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(w - 0.0035, y, f'{w:.4f}', va='center', ha='right', fontsize=9.8, fontweight='normal', color='white')
        
    plt.tight_layout()
    save_fig(fig, "FigureS4_temporal_modeling")
    plt.close()
    print("Figure S4 generated successfully!")

# ==============================================================================
# FIGURE S5: SMOTE Oversampling
# ==============================================================================
def make_figure_s5():
    print("Generating Figure S5: SMOTE Augmentation...")
    csv_path = os.path.join(BASE_DIR, 'results', 'manuscript_draft', 'tables', 'final', 'Table8_synthetic_augmentation.csv')
    df_s5 = pd.read_csv(csv_path)
    
    df_mlp = df_s5[df_s5['Architecture'] == 'MLP'].iloc[::-1].reset_index(drop=True)
    df_sage = df_s5[df_s5['Architecture'] == 'GraphSAGE'].iloc[::-1].reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.0), dpi=300, sharex=True)
    
    # MLP Panel
    y1 = np.arange(len(df_mlp))
    bars1 = ax1.barh(y1, df_mlp['Ensemble_Test_AP'], height=0.60, color='#854d0e', alpha=0.88,
                     edgecolor='#0f172a', linewidth=0.9)
    mlp_base = 0.0704
    ax1.axvline(mlp_base, color='#dc2626', linestyle='--', linewidth=1.5)
    ax1.text(mlp_base - 0.0015, len(y1) - 0.35, f'Baseline MLP ({mlp_base:.4f})',
             color='#dc2626', fontsize=9.2, fontweight='normal', ha='right', va='bottom')
    ax1.set_yticks(y1)
    ax1.set_yticklabels(df_mlp['Model'], fontsize=10.5, fontweight='normal')
    ax1.tick_params(labelsize=10.5)
    ax1.set_title('(A) MLP Input Space SMOTE', fontsize=11.5, fontweight='normal', pad=6)
    ax1.set_ylim([-0.6, len(y1) + 0.5])
    ax1.grid(True, axis='x', linestyle='--', alpha=0.4)
    
    for bar in bars1:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax1.text(w - 0.0035, y, f'{w:.4f}', va='center', ha='right', fontsize=9.8, fontweight='normal', color='white')
        
    # GraphSAGE Panel
    y2 = np.arange(len(df_sage))
    bars2 = ax2.barh(y2, df_sage['Ensemble_Test_AP'], height=0.60, color='#db2777', alpha=0.88,
                     edgecolor='#0f172a', linewidth=0.9)
    sage_base = 0.1107
    ax2.axvline(sage_base, color='#dc2626', linestyle='--', linewidth=1.5)
    ax2.text(sage_base - 0.0015, len(y2) - 0.35, f'Baseline GraphSAGE ({sage_base:.4f})',
             color='#dc2626', fontsize=9.2, fontweight='normal', ha='right', va='bottom')
    ax2.set_yticks(y2)
    ax2.set_yticklabels(df_sage['Model'], fontsize=10.5, fontweight='normal')
    ax2.tick_params(labelsize=10.5)
    ax2.set_title('(B) GraphSAGE Embedding-Space SMOTE', fontsize=11.5, fontweight='normal', pad=6)
    ax2.set_xlabel('National Prospective Test AP', fontsize=11.5, fontweight='normal', labelpad=7)
    ax2.set_xlim([0.0, 0.138])
    ax2.set_ylim([-0.6, len(y2) + 0.5])
    ax2.grid(True, axis='x', linestyle='--', alpha=0.4)
    
    for bar in bars2:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax2.text(w - 0.0035, y, f'{w:.4f}', va='center', ha='right', fontsize=9.8, fontweight='normal', color='white')
        
    plt.tight_layout()
    save_fig(fig, "FigureS5_smote")
    plt.close()
    print("Figure S5 generated successfully!")

# ==============================================================================
# UNIFIED FIGURE S2: All 5 ablation families in a clean, perfectly aligned 3x2 grid
# ==============================================================================
def make_figure_s2_unified():
    print("Generating Unified Figure S2: Family Ablations (3x2 Grid)...")
    tables_dir = os.path.join(BASE_DIR, 'results', 'manuscript_draft', 'tables', 'final')
    df_loss = pd.read_csv(os.path.join(tables_dir, 'Table5_imbalance_loss_ablation.csv'))
    df_pre = pd.read_csv(os.path.join(tables_dir, 'Table6_representation_learning.csv'))
    df_temp = pd.read_csv(os.path.join(tables_dir, 'Table7_temporal_ablation.csv'))
    df_smote = pd.read_csv(os.path.join(tables_dir, 'Table8_synthetic_augmentation.csv'))

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.8), dpi=300)
    plt.subplots_adjust(hspace=0.48, wspace=0.46, top=0.94, bottom=0.07, left=0.20, right=0.97)

    sage_base = 0.1107
    mlp_base = 0.0704

    labels_a = {
        'B0: All national': 'B0: All National CSDs',
        'B1: All poultry-at-risk': 'B1: Commercial Poultry CSDs',
        'B2: Ratio 100:1': 'B2: Subsample 100:1',
        'B3: Ratio 50:1': 'B3: Subsample 50:1',
        'B4: Ratio 25:1': 'B4: Subsample 25:1',
        'B5: Ratio 10:1': 'B5: Subsample 10:1',
        'B6: Ratio 3:1': 'B6: Subsample 3:1',
    }

    labels_b = {
        'B0 + BCE': 'Baseline BCE',
        'B0 + Focal Loss': 'Focal Loss',
        'B0 + Weighted BCE': 'Weighted BCE',
    }

    labels_c = {
        'S0: Scratch': 'Baseline Scratch',
        'S1: SSL masked-15%': 'SSL Masked-15%',
        'S2: Wild-bird transfer': 'Wild-Bird Transfer',
    }

    labels_d = {
        'T0: Snapshot GraphSAGE': 'Snapshot GraphSAGE',
        'T1: GraphSAGE + GRU3': 'GraphSAGE + GRU-3',
        'T2: GraphSAGE + GRU6': 'GraphSAGE + GRU-6',
        'T3: Temporal attention': 'Temporal Attention',
        'T1_nolag: GRU3 no lags': 'GRU-3 (No History)',
        'T2_nolag: GRU6 no lags': 'GRU-6 (No History)',
        'T3_nolag: Attention no lags': 'Attention (No History)',
    }

    labels_e = {
        'M_baseline': 'Baseline MLP',
        'M_smote_1to20': 'SMOTE (1:20)',
        'M_smote_1to10': 'SMOTE (1:10)',
        'M_smote_1to1': 'SMOTE (1:1)',
    }

    labels_f = {
        'G_baseline': 'Baseline GraphSAGE',
        'G_embed_smote_1to20': 'Embed-SMOTE (1:20)',
        'G_embed_smote_1to10': 'Embed-SMOTE (1:10)',
        'G_embed_smote_1to1': 'Embed-SMOTE (1:1)',
    }

    # (a) Negative Sampling
    ax_a = axes[0, 0]
    df_s1 = df_loss[df_loss['Category'] == 'Negative sampling'].iloc[::-1].reset_index(drop=True)
    y_a = np.arange(len(df_s1))
    bars_a = ax_a.barh(y_a, df_s1['Ensemble_Test_AP'], height=0.62, color='#d97706', alpha=0.88,
                       edgecolor='#0f172a', linewidth=0.8)
    ax_a.axvline(sage_base, color='#dc2626', linestyle='--', linewidth=1.3)
    ax_a.text(sage_base - 0.002, len(y_a) - 0.1, f'Baseline ({sage_base:.4f})',
              color='#dc2626', fontsize=8.8, fontweight='normal', ha='right', va='top')
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels([labels_a.get(x, x) for x in df_s1['Configuration']], fontsize=9.2, fontweight='normal')
    ax_a.set_xlim([0.0, 0.145])
    ax_a.set_ylim([-0.6, len(y_a) + 0.7])
    ax_a.set_title('(a) Negative Background Sampling', fontsize=11.0, fontweight='normal', loc='left', color='#0f172a', pad=8)
    ax_a.tick_params(labelsize=9.0)
    ax_a.grid(True, axis='x', linestyle='--', alpha=0.4)
    for b in bars_a:
        w = b.get_width()
        ax_a.text(w - 0.003, b.get_y() + b.get_height()/2, f'{w:.4f}', va='center', ha='right', fontsize=8.5, color='white')

    # (b) Loss Objective Functions
    ax_b = axes[0, 1]
    df_s2 = df_loss[df_loss['Category'] == 'Loss function'].iloc[::-1].reset_index(drop=True)
    y_b = np.arange(len(df_s2))
    bars_b = ax_b.barh(y_b, df_s2['Ensemble_Test_AP'], height=0.55, color='#dc2626', alpha=0.88,
                       edgecolor='#0f172a', linewidth=0.8)
    ax_b.axvline(sage_base, color='#0f172a', linestyle='--', linewidth=1.3)
    ax_b.text(sage_base - 0.002, len(y_b) - 0.1, f'Baseline BCE ({sage_base:.4f})',
              color='#0f172a', fontsize=8.8, fontweight='normal', ha='right', va='top')
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels([labels_b.get(x, x) for x in df_s2['Configuration']], fontsize=9.2, fontweight='normal')
    ax_b.set_xlim([0.0, 0.145])
    ax_b.set_ylim([-0.6, len(y_b) + 0.7])
    ax_b.set_title('(b) Loss Objective Functions & Class Weighting', fontsize=11.0, fontweight='normal', loc='left', color='#0f172a', pad=8)
    ax_b.tick_params(labelsize=9.0)
    ax_b.grid(True, axis='x', linestyle='--', alpha=0.4)
    for b in bars_b:
        w = b.get_width()
        ax_b.text(w - 0.003, b.get_y() + b.get_height()/2, f'{w:.4f}', va='center', ha='right', fontsize=8.5, color='white')

    # (c) Pretraining
    ax_c = axes[1, 0]
    df_s3 = df_pre.iloc[::-1].reset_index(drop=True)
    y_c = np.arange(len(df_s3))
    bars_c = ax_c.barh(y_c, df_s3['Ensemble_Test_AP'], height=0.55, color='#16a34a', alpha=0.88,
                       edgecolor='#0f172a', linewidth=0.8)
    pre_base = 0.1111
    ax_c.axvline(pre_base, color='#dc2626', linestyle='--', linewidth=1.3)
    ax_c.text(pre_base - 0.002, len(y_c) - 0.1, f'Baseline Scratch ({pre_base:.4f})',
              color='#dc2626', fontsize=8.8, fontweight='normal', ha='right', va='top')
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([labels_c.get(x, x) for x in df_s3['Model']], fontsize=9.2, fontweight='normal')
    ax_c.set_xlim([0.0, 0.145])
    ax_c.set_ylim([-0.6, len(y_c) + 0.7])
    ax_c.set_title('(c) Auxiliary Pretraining & Transfer', fontsize=11.0, fontweight='normal', loc='left', color='#0f172a', pad=8)
    ax_c.tick_params(labelsize=9.0)
    ax_c.grid(True, axis='x', linestyle='--', alpha=0.4)
    for b in bars_c:
        w = b.get_width()
        ax_c.text(w - 0.003, b.get_y() + b.get_height()/2, f'{w:.4f}', va='center', ha='right', fontsize=8.5, color='white')

    # (d) Temporal Sequence Modeling
    ax_d = axes[1, 1]
    df_s4 = df_temp.iloc[::-1].reset_index(drop=True)
    y_d = np.arange(len(df_s4))
    bars_d = ax_d.barh(y_d, df_s4['Ensemble_Test_AP'], height=0.62, color='#7c3aed', alpha=0.88,
                       edgecolor='#0f172a', linewidth=0.8)
    ax_d.axvline(sage_base, color='#dc2626', linestyle='--', linewidth=1.3)
    ax_d.text(sage_base - 0.002, len(y_d) - 0.1, f'Baseline Snapshot ({sage_base:.4f})',
              color='#dc2626', fontsize=8.8, fontweight='normal', ha='right', va='top')
    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels([labels_d.get(x, x) for x in df_s4['Model']], fontsize=9.2, fontweight='normal')
    ax_d.set_xlim([0.0, 0.145])
    ax_d.set_ylim([-0.6, len(y_d) + 0.7])
    ax_d.set_title('(d) Temporal Sequence Modeling vs. Static Lags', fontsize=11.0, fontweight='normal', loc='left', color='#0f172a', pad=8)
    ax_d.tick_params(labelsize=9.0)
    ax_d.grid(True, axis='x', linestyle='--', alpha=0.4)
    for b in bars_d:
        w = b.get_width()
        ax_d.text(w - 0.003, b.get_y() + b.get_height()/2, f'{w:.4f}', va='center', ha='right', fontsize=8.5, color='white')

    # (e) MLP SMOTE
    ax_e = axes[2, 0]
    df_mlp_smote = df_smote[df_smote['Architecture'] == 'MLP'].iloc[::-1].reset_index(drop=True)
    y_e = np.arange(len(df_mlp_smote))
    bars_e = ax_e.barh(y_e, df_mlp_smote['Ensemble_Test_AP'], height=0.58, color='#854d0e', alpha=0.88,
                       edgecolor='#0f172a', linewidth=0.8)
    ax_e.axvline(mlp_base, color='#dc2626', linestyle='--', linewidth=1.3)
    ax_e.text(mlp_base - 0.002, len(y_e) - 0.1, f'Baseline MLP ({mlp_base:.4f})',
              color='#dc2626', fontsize=8.8, fontweight='normal', ha='right', va='top')
    ax_e.set_yticks(y_e)
    ax_e.set_yticklabels([labels_e.get(x, x) for x in df_mlp_smote['Model']], fontsize=9.2, fontweight='normal')
    ax_e.set_xlim([0.0, 0.145])
    ax_e.set_ylim([-0.6, len(y_e) + 0.7])
    ax_e.set_xlabel('National Prospective Test AP', fontsize=10.5, fontweight='normal', labelpad=6)
    ax_e.set_title('(e) MLP Input-Space SMOTE', fontsize=11.0, fontweight='normal', loc='left', color='#0f172a', pad=8)
    ax_e.tick_params(labelsize=9.0)
    ax_e.grid(True, axis='x', linestyle='--', alpha=0.4)
    for b in bars_e:
        w = b.get_width()
        ax_e.text(w - 0.003, b.get_y() + b.get_height()/2, f'{w:.4f}', va='center', ha='right', fontsize=8.5, color='white')

    # (f) GraphSAGE SMOTE
    ax_f = axes[2, 1]
    df_sage_smote = df_smote[df_smote['Architecture'] == 'GraphSAGE'].iloc[::-1].reset_index(drop=True)
    y_f = np.arange(len(df_sage_smote))
    bars_f = ax_f.barh(y_f, df_sage_smote['Ensemble_Test_AP'], height=0.58, color='#db2777', alpha=0.88,
                       edgecolor='#0f172a', linewidth=0.8)
    ax_f.axvline(sage_base, color='#dc2626', linestyle='--', linewidth=1.3)
    ax_f.text(sage_base - 0.002, len(y_f) - 0.1, f'Baseline GraphSAGE ({sage_base:.4f})',
              color='#dc2626', fontsize=8.8, fontweight='normal', ha='right', va='top')
    ax_f.set_yticks(y_f)
    ax_f.set_yticklabels([labels_f.get(x, x) for x in df_sage_smote['Model']], fontsize=9.2, fontweight='normal')
    ax_f.set_xlim([0.0, 0.145])
    ax_f.set_ylim([-0.6, len(y_f) + 0.7])
    ax_f.set_xlabel('National Prospective Test AP', fontsize=10.5, fontweight='normal', labelpad=6)
    ax_f.set_title('(f) GraphSAGE Embedding-Space SMOTE', fontsize=11.0, fontweight='normal', loc='left', color='#0f172a', pad=8)
    ax_f.tick_params(labelsize=9.0)
    ax_f.grid(True, axis='x', linestyle='--', alpha=0.4)
    for b in bars_f:
        w = b.get_width()
        ax_f.text(w - 0.003, b.get_y() + b.get_height()/2, f'{w:.4f}', va='center', ha='right', fontsize=8.5, color='white')

    save_fig(fig, "FigureS2_family_ablations")
    print("Figure S2 Unified generated successfully!")

if __name__ == '__main__':
    make_figure_s1()
    make_figure_s2()
    make_figure_s3()
    make_figure_s4()
    make_figure_s5()
    make_figure_s2_unified()
    print("All supplementary figures regenerated successfully!")
