import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import NearestNeighbors

import time
import shutil

SCRATCH_DIR = 'scratch'
os.makedirs(SCRATCH_DIR, exist_ok=True)

OUT_DIRS = [
    'results/manuscript_elsevier',
    'results/manuscript_elsevier/figures',
    'results/manuscript_latex/figures',
    'results/final_manuscript/figures'
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

def safe_save(fig, basename, out_dirs):
    scratch_png = os.path.join(SCRATCH_DIR, f"{basename}.png")
    scratch_pdf = os.path.join(SCRATCH_DIR, f"{basename}.pdf")
    fig.savefig(scratch_pdf, bbox_inches='tight')
    try:
        fig.savefig(scratch_png, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: could not save {scratch_png}: {e}")
    plt.close(fig)
    
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        tgt_png = os.path.join(d, f"{basename}.png")
        tgt_pdf = os.path.join(d, f"{basename}.pdf")
        for src, tgt in [(scratch_png, tgt_png), (scratch_pdf, tgt_pdf)]:
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


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11.0,
    'axes.labelsize': 12.0,
    'axes.labelweight': 'normal',
    'axes.titlesize': 13.0,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# Load simplified CD boundaries
cd_path = "data/processed/boundaries/census_divisions_simplified.geojson"
gdf_cd = gpd.read_file(cd_path)

# Load precomputed centroids
cents_df = pd.read_csv("data/processed/boundaries/csd_centroids_epsg3347.csv", dtype={'csd_uid': str})
cents_df.rename(columns={'x': 'lon', 'y': 'lat'}, inplace=True)

# ==============================================================================
# FIGURE 2: Nationwide Outbreak Distribution & Proximity Network
# ==============================================================================
def make_figure2():
    print("Generating Figure 2 (Full-page unbolded layout)...")
    data_path = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
    df = pd.read_csv(data_path, usecols=['csd_uid', 'year', 'outbreak_next_month'], dtype={"csd_uid": str})
    
    df_pos = df[df['outbreak_next_month'] == 1].copy()
    df_pos['Period'] = 'Training (2022–2023)'
    df_pos.loc[df_pos['year'] == 2024, 'Period'] = 'Validation (2024)'
    df_pos.loc[df_pos['year'] == 2025, 'Period'] = 'Locked Test (2025)'
    
    pos_unique = df_pos.drop_duplicates(subset=['csd_uid', 'Period'])
    pos_merged = pos_unique.merge(cents_df, on='csd_uid', how='inner')
    
    # Large 2-row full page layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 10.5), dpi=300, gridspec_kw={'height_ratios': [1.18, 1.0]})
    
    # Panel A: Nationwide Outbreak Map
    gdf_cd.plot(ax=ax1, color='#f8fafc', edgecolor='#94a3b8', linewidth=0.6, zorder=1)
    
    colors = {
        'Training (2022–2023)': '#2563eb',
        'Validation (2024)': '#f59e0b',
        'Locked Test (2025)': '#dc2626'
    }
    markers = {
        'Training (2022–2023)': 'o',
        'Validation (2024)': 's',
        'Locked Test (2025)': '^'
    }
    sizes = {
        'Training (2022–2023)': 48,
        'Validation (2024)': 56,
        'Locked Test (2025)': 66
    }
    
    for period, c in colors.items():
        sub = pos_merged[pos_merged['Period'] == period]
        ax1.scatter(sub['lon'], sub['lat'], c=c, s=sizes[period], marker=markers[period],
                    label=f"{period} ({len(sub)} unique CSDs)", edgecolors='black', linewidth=0.6, alpha=0.9, zorder=3)
        
    ax1.set_xlim(-141, -52)
    ax1.set_ylim(41.0, 70.0)
    ax1.set_xticks([-130, -110, -90, -70])
    ax1.set_xticklabels(['130°W', '110°W', '90°W', '70°W'], fontsize=10.5)
    ax1.set_yticks([45, 55, 65])
    ax1.set_yticklabels(['45°N', '55°N', '65°N'], fontsize=10.5)
    ax1.set_title("(a) Outbreak Distribution Across Canadian CSDs (2022–2025)",
                  fontsize=13.0, fontweight='normal', pad=8, color='#0f172a')
    ax1.set_xlabel("Longitude", fontsize=11.5, fontweight='normal')
    ax1.set_ylabel("Latitude", fontsize=11.5, fontweight='normal')
    ax1.tick_params(labelsize=10.5)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cbd5e1', fontsize=10.0)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    
    rect_lon_min, rect_lon_max = -83.5, -73.0
    rect_lat_min, rect_lat_max = 42.0, 46.5
    rect = plt.Rectangle((rect_lon_min, rect_lat_min), rect_lon_max - rect_lon_min, rect_lat_max - rect_lat_min,
                         fill=False, edgecolor='#7c3aed', linewidth=1.5, linestyle='--', zorder=4)
    ax1.add_patch(rect)
    # Clean badge on the inset box (unbolded)
    ax1.text(-84.0, 47.0, "(b)", fontsize=10.5, fontweight='normal', color='#7c3aed', ha='right', va='bottom', zorder=5,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#7c3aed', alpha=0.92))

    # Panel B: Regional Spatial Proximity Network
    gdf_cd.plot(ax=ax2, color='#f1f5f9', edgecolor='#94a3b8', linewidth=0.7, zorder=1)
    
    region_csds = cents_df[(cents_df['lon'] >= rect_lon_min) & (cents_df['lon'] <= rect_lon_max) &
                           (cents_df['lat'] >= rect_lat_min) & (cents_df['lat'] <= rect_lat_max)].copy()
    
    coords = region_csds[['lon', 'lat']].values
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    lines = []
    for i, nbr_list in enumerate(indices):
        p1 = coords[i]
        for j in nbr_list[1:]:
            p2 = coords[j]
            lines.append([p1, p2])
            
    lc = LineCollection(lines, colors='#60a5fa', linewidths=0.75, alpha=0.65, zorder=2)
    ax2.add_collection(lc)
    
    ax2.scatter(region_csds['lon'], region_csds['lat'], s=20, c='#64748b', alpha=0.60, zorder=3)
    
    reg_pos = pos_merged[(pos_merged['lon'] >= rect_lon_min) & (pos_merged['lon'] <= rect_lon_max) &
                         (pos_merged['lat'] >= rect_lat_min) & (pos_merged['lat'] <= rect_lat_max)]
    
    for period, c in colors.items():
        sub = reg_pos[reg_pos['Period'] == period]
        if len(sub) > 0:
            ax2.scatter(sub['lon'], sub['lat'], c=c, s=sizes[period]*0.9, marker=markers[period],
                        edgecolors='black', linewidth=0.6, alpha=0.95, zorder=4)

    ax2.set_xlim(rect_lon_min, rect_lon_max)
    ax2.set_ylim(rect_lat_min, rect_lat_max)
    ax2.set_xticks([-82.0, -79.0, -76.0])
    ax2.set_xticklabels(['82°W', '79°W', '76°W'], fontsize=10.5, fontweight='normal')
    ax2.set_yticks([43.0, 44.5, 46.0])
    ax2.set_yticklabels(['43.0°N', '44.5°N', '46.0°N'], fontsize=10.5, fontweight='normal')
    ax2.set_title("(b) Regional Spatial k-NN (k = 3) Proximity Graph (ON/QC)",
                  fontsize=13.0, fontweight='normal', pad=8, color='#0f172a')
    ax2.set_xlabel("Longitude", fontsize=11.5, fontweight='normal')
    ax2.set_ylabel("Latitude", fontsize=11.5, fontweight='normal')
    ax2.tick_params(labelsize=10.5)
    ax2.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    
    # Overlapping label box removed completely per instructions

    plt.tight_layout()
    safe_save(fig, "Figure2_canada_graph", OUT_DIRS)
    print("Figure 2 generated successfully!")

# ==============================================================================
# FIGURE 11: Nationwide Prospective 2025 Risk Map with Fraser Valley & ON Insets
# ==============================================================================
def make_figure11():
    print("Generating Figure 11...")
    preds_path = "results/final_manuscript/canonical_predictions/preds_GraphSAGE.csv"
    df_preds = pd.read_csv(preds_path, dtype={'csd_uid': str})
    
    df_2025 = df_preds[df_preds['year'] == 2025].copy()
    risk_csd = df_2025.groupby('csd_uid')['pred_prob_ensemble'].max().reset_index()
    risk_csd.rename(columns={'pred_prob_ensemble': 'max_risk_2025'}, inplace=True)
    
    merged = cents_df.merge(risk_csd, on='csd_uid', how='left')
    merged['max_risk_2025'] = merged['max_risk_2025'].fillna(0.0)

    colors = ['#f8fafc', '#fef08a', '#f97316', '#dc2626', '#7f1d1d']
    cmap_risk = LinearSegmentedColormap.from_list('RiskCmap', colors, N=256)

    fig = plt.figure(figsize=(11.5, 6.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1, 1], wspace=0.22, hspace=0.48,
                           left=0.06, right=0.97, top=0.94, bottom=0.15)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_bc = fig.add_subplot(gs[0, 1])
    ax_on = fig.add_subplot(gs[1, 1])
    
    # Panel A: Main Map
    gdf_cd.plot(ax=ax_main, color='#f8fafc', edgecolor='#94a3b8', linewidth=0.6, zorder=1)
    
    low_risk = merged[merged['max_risk_2025'] < 0.05]
    ax_main.scatter(low_risk['lon'], low_risk['lat'], c='#cbd5e1', s=6, alpha=0.35, zorder=2)
    
    high_risk = merged[merged['max_risk_2025'] >= 0.05].sort_values('max_risk_2025')
    sc = ax_main.scatter(high_risk['lon'], high_risk['lat'], c=high_risk['max_risk_2025'],
                         cmap=cmap_risk, s=high_risk['max_risk_2025']*65 + 18,
                         edgecolors='black', linewidth=0.5, alpha=0.92, zorder=3, vmin=0, vmax=1.0)
    
    # Dedicated horizontal colorbar beneath Panel A with guaranteed vertical breathing room
    cbar_ax = fig.add_axes([0.08, 0.05, 0.48, 0.026])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Prospective 2025 Outbreak Risk Score (GraphSAGE)', fontsize=11.0, labelpad=5)
    cbar.ax.tick_params(labelsize=10.0)
    
    ax_main.set_xlim(-141, -52)
    ax_main.set_ylim(41.5, 70)
    ax_main.set_xticks([-130, -110, -90, -70])
    ax_main.set_xticklabels(['-130°', '-110°', '-90°', '-70°'])
    ax_main.set_yticks([45, 55, 65])
    ax_main.set_yticklabels(['45°', '55°', '65°'])
    ax_main.set_title("(a) Nationwide Prospective 2025 Risk Profile",
                      fontsize=12.5, pad=8, fontweight='normal', color='#0f172a')
    ax_main.set_xlabel("Longitude (°W)", fontsize=11.0, fontweight='normal')
    ax_main.set_ylabel("Latitude (°N)", fontsize=11.0, fontweight='normal')
    ax_main.tick_params(labelsize=10.0)
    ax_main.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    
    # Inset highlight boxes with oceanic and US whitespace placement
    bc_box = plt.Rectangle((-123.5, 48.9), 2.2, 1.0, fill=False, edgecolor='#2563eb', linewidth=1.6, linestyle='--', zorder=4)
    ax_main.add_patch(bc_box)
    ax_main.annotate("Fraser Valley (B)", xy=(-123.5, 49.4), xytext=(-137.5, 52.5),
                     arrowprops=dict(arrowstyle="->", color='#2563eb', lw=1.2),
                     color='#1d4ed8', fontsize=9.5, fontweight='normal', zorder=5,
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#2563eb', alpha=0.95))

    on_box = plt.Rectangle((-83.0, 42.3), 4.5, 2.7, fill=False, edgecolor='#dc2626', linewidth=1.6, linestyle='--', zorder=4)
    ax_main.add_patch(on_box)
    ax_main.annotate("SW Ontario (C)", xy=(-83.0, 43.5), xytext=(-91.5, 44.5),
                     arrowprops=dict(arrowstyle="->", color='#dc2626', lw=1.2),
                     color='#b91c1c', fontsize=9.5, fontweight='normal', zorder=5,
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#dc2626', alpha=0.95))

    # Panel B: Fraser Valley BC Zoom
    ax_bc.set_facecolor('#f8fafc')
    gdf_cd.plot(ax=ax_bc, color='#f1f5f9', edgecolor='#94a3b8', linewidth=0.7, zorder=1)
    pts_bc = merged[(merged['lon'] >= -123.5) & (merged['lon'] <= -121.3) &
                    (merged['lat'] >= 48.9) & (merged['lat'] <= 49.9)].sort_values('max_risk_2025')
    ax_bc.scatter(pts_bc['lon'], pts_bc['lat'], c=pts_bc['max_risk_2025'], cmap=cmap_risk,
                  s=pts_bc['max_risk_2025']*140 + 35, edgecolors='black', linewidth=0.5, alpha=0.95, zorder=3, vmin=0, vmax=1.0)
    ax_bc.set_xlim(-123.5, -121.3)
    ax_bc.set_ylim(48.9, 49.9)
    ax_bc.set_xticks([-123.0, -122.0])
    ax_bc.set_xticklabels(['-123°', '-122°'])
    ax_bc.set_yticks([49.0, 49.5])
    ax_bc.set_yticklabels(['49.0°', '49.5°'])
    ax_bc.set_title("(b) Fraser Valley Epicenter (BC)", fontsize=11.5, fontweight='normal', color='#1e3a8a', pad=8)
    ax_bc.set_xlabel("Longitude (°W)", fontsize=10.5, fontweight='normal')
    ax_bc.set_ylabel("Latitude (°N)", fontsize=10.5, fontweight='normal')
    ax_bc.tick_params(labelsize=10.0)
    ax_bc.grid(True, linestyle=':', alpha=0.5)

    # Panel C: Southwestern Ontario Zoom
    ax_on.set_facecolor('#f8fafc')
    gdf_cd.plot(ax=ax_on, color='#f1f5f9', edgecolor='#94a3b8', linewidth=0.7, zorder=1)
    pts_on = merged[(merged['lon'] >= -83.0) & (merged['lon'] <= -78.5) &
                    (merged['lat'] >= 42.3) & (merged['lat'] <= 45.0)].sort_values('max_risk_2025')
    ax_on.scatter(pts_on['lon'], pts_on['lat'], c=pts_on['max_risk_2025'], cmap=cmap_risk,
                  s=pts_on['max_risk_2025']*140 + 35, edgecolors='black', linewidth=0.5, alpha=0.95, zorder=3, vmin=0, vmax=1.0)
    ax_on.set_xlim(-83.0, -78.5)
    ax_on.set_ylim(42.3, 45.0)
    ax_on.set_xticks([-82.0, -80.0])
    ax_on.set_xticklabels(['-82°', '-80°'])
    ax_on.set_yticks([43.0, 44.0])
    ax_on.set_yticklabels(['43°', '44°'])
    ax_on.set_title("(c) Southwestern Ontario Belt", fontsize=11.5, fontweight='normal', color='#991b1b', pad=8)
    ax_on.set_xlabel("Longitude (°W)", fontsize=10.5, fontweight='normal')
    ax_on.set_ylabel("Latitude (°N)", fontsize=10.5, fontweight='normal')
    ax_on.tick_params(labelsize=10.0)
    ax_on.grid(True, linestyle=':', alpha=0.5)

    safe_save(fig, "Figure11_nationwide_risk_map", OUT_DIRS)
    print("Figure 11 generated successfully!")

if __name__ == '__main__':
    make_figure2()
    make_figure11()
    print("Figures 2 and 11 generated successfully!")
