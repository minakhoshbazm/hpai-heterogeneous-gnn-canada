import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from pathlib import Path

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

def safe_savefig(fig, path, dpi=300):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches='tight')

def generate_figure11_master():
    print("Generating Master Publication-Grade Figure 11...")
    
    # 1. Load data
    csd_shp = "data/shapefiles/clean_csd_boundaries.shp"
    cd_shp = "data/shapefiles/clean_cd_boundaries.shp"
    gdf_csd = gpd.read_file(csd_shp).to_crs(epsg=4326)
    gdf_cd = gpd.read_file(cd_shp).to_crs(epsg=4326)
    
    # Ground truth 2025 outbreaks
    gt_df = pd.read_csv("data/processed/csd_monthly_outbreaks_2022_2025.csv")
    gt_2025 = gt_df[(gt_df['year'] == 2025) & (gt_df['outbreak_occurred'] == 1)]
    outbreak_csds = gt_2025['csduid'].unique()
    
    csd_pos = gdf_csd[gdf_csd['CSDUID'].isin(outbreak_csds)].copy()
    csd_pos['lon'] = csd_pos.geometry.centroid.x
    csd_pos['lat'] = csd_pos.geometry.centroid.y

    # GraphSAGE predictions
    preds_file = "data/predictions/graphsage_canonical_test_2025.csv"
    if not os.path.exists(preds_file):
        preds_file = "data/predictions/test_predictions_2025.csv"
    preds_df = pd.read_csv(preds_file)
    if 'graphsage' in preds_df.columns:
        p_col = 'graphsage'
    elif 'y_pred_prob' in preds_df.columns:
        p_col = 'y_pred_prob'
    else:
        p_col = preds_df.columns[-1]

    # Max monthly risk per CSD
    max_risk = preds_df.groupby('csduid')[p_col].max().reset_index()
    max_risk['percentile_risk'] = max_risk[p_col].rank(pct=True) * 100.0

    gdf_csd['CSDUID_num'] = pd.to_numeric(gdf_csd['CSDUID'], errors='coerce')
    max_risk['csduid_num'] = pd.to_numeric(max_risk['csduid'], errors='coerce')
    data_gdf = gdf_csd.merge(max_risk, left_on='CSDUID_num', right_on='csduid_num', how='left')
    data_gdf['percentile_risk'] = data_gdf['percentile_risk'].fillna(0)

    # Color map
    colors_risk = ['#f8fafc', '#fed7aa', '#f97316', '#dc2626', '#7f1d1d']
    cmap_risk = mcolors.LinearSegmentedColormap.from_list('risk_map', colors_risk, N=256)
    norm = mcolors.Normalize(vmin=0, vmax=100)

    # FIGURE LAYOUT:
    # Top: Panel A (Full Canada Map, wide aspect)
    # Bottom: Panel B (Fraser Valley) and Panel C (SW Ontario) side-by-side!
    fig = plt.figure(figsize=(8.2, 7.6), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.0, 1.0], hspace=0.28, wspace=0.22)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_bc = fig.add_subplot(gs[1, 0])
    ax_on = fig.add_subplot(gs[1, 1])

    for ax in [ax_main, ax_bc, ax_on]:
        ax.set_facecolor('#f0f7ff')

    # --- PANEL A: Nationwide Whole-Cell Choropleth ---
    data_gdf.plot(ax=ax_main, column='percentile_risk', cmap=cmap_risk, norm=norm,
                  edgecolor='#94a3b8', linewidth=0.20, zorder=2)
    gdf_cd.plot(ax=ax_main, facecolor='none', edgecolor='#475569', linewidth=0.55, zorder=3)

    ax_main.scatter(csd_pos['lon'], csd_pos['lat'], marker='o',
                    s=40, facecolor='#06b6d4', edgecolor='#000000', linewidth=0.9,
                    alpha=0.95, zorder=5, label='Confirmed 2025 Outbreak (N=68 CSDs)')

    ax_main.set_xlim(-141, -52)
    ax_main.set_ylim(41.5, 70)
    ax_main.set_xticks([-130, -110, -90, -70])
    ax_main.set_xticklabels(['130°W', '110°W', '90°W', '70°W'], fontsize=11.0, fontweight='bold')
    ax_main.set_yticks([45, 55, 65])
    ax_main.set_yticklabels(['45°N', '55°N', '65°N'], fontsize=11.0, fontweight='bold')
    ax_main.set_title("(A) Nationwide Prospective 2025 HPAI Outbreak Risk: GraphSAGE",
                      fontsize=13.0, fontweight='bold', pad=8, color='#0f172a')
    ax_main.set_xlabel("Longitude", fontsize=11.5, fontweight='bold', labelpad=4)
    ax_main.set_ylabel("Latitude", fontsize=11.5, fontweight='bold', labelpad=4)
    ax_main.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    ax_main.legend(loc='upper left', fontsize=10.5, frameon=True, facecolor='#ffffff', edgecolor='#94a3b8')

    # Inset indicator boxes with clean arrows
    bc_box = patches.Rectangle((-123.5, 48.9), 2.2, 1.0, fill=False, edgecolor='#2563eb', linewidth=1.6, linestyle='--', zorder=6)
    ax_main.add_patch(bc_box)
    ax_main.annotate("Fraser Valley (B)", xy=(-123.5, 49.4), xytext=(-136.0, 52.0),
                     arrowprops=dict(arrowstyle="->", color='#2563eb', lw=1.2),
                     color='#1d4ed8', fontsize=10.5, fontweight='bold', zorder=7,
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#2563eb', linewidth=0.8, alpha=0.95))

    on_box = patches.Rectangle((-83.0, 42.3), 4.5, 2.7, fill=False, edgecolor='#dc2626', linewidth=1.6, linestyle='--', zorder=6)
    ax_main.add_patch(on_box)
    ax_main.annotate("SW Ontario (C)", xy=(-83.0, 43.5), xytext=(-88.0, 42.5),
                     arrowprops=dict(arrowstyle="->", color='#dc2626', lw=1.2),
                     color='#b91c1c', fontsize=10.5, fontweight='bold', zorder=7,
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#dc2626', linewidth=0.8, alpha=0.95))

    # Colorbar inside Panel A or below it
    cbar_ax = fig.add_axes([0.62, 0.84, 0.26, 0.016])
    sm_ind = plt.cm.ScalarMappable(norm=norm, cmap=cmap_risk)
    cbar = fig.colorbar(sm_ind, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Risk Percentile Tier (0–100%)', fontsize=9.5, fontweight='bold', labelpad=4)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.ax.tick_params(labelsize=9.0)

    # --- PANEL B: Fraser Valley BC Zoom ---
    bc_bounds = (-123.5, 48.9, -121.3, 49.9)
    gdf_bc = data_gdf.cx[bc_bounds[0]:bc_bounds[2], bc_bounds[1]:bc_bounds[3]]
    gdf_bc.plot(ax=ax_bc, column='percentile_risk', cmap=cmap_risk, norm=norm,
                edgecolor='#334155', linewidth=0.50, zorder=2)
    gdf_cd.plot(ax=ax_bc, facecolor='none', edgecolor='#64748b', linewidth=0.9, zorder=3)
    
    pos_bc = csd_pos[(csd_pos['lon'] >= -123.5) & (csd_pos['lon'] <= -121.3) &
                     (csd_pos['lat'] >= 48.9) & (csd_pos['lat'] <= 49.9)]
    ax_bc.scatter(pos_bc['lon'], pos_bc['lat'], marker='*', s=130, facecolor='#06b6d4',
                  edgecolor='#000000', linewidth=1.0, zorder=5, label='Actual Outbreak')

    ax_bc.set_xlim(-123.5, -121.3)
    ax_bc.set_ylim(48.9, 49.9)
    ax_bc.set_xticks([-123.0, -122.0])
    ax_bc.set_xticklabels(['123.0°W', '122.0°W'], fontsize=10.5, fontweight='bold')
    ax_bc.set_yticks([49.0, 49.5])
    ax_bc.set_yticklabels(['49.0°N', '49.5°N'], fontsize=10.5, fontweight='bold')
    ax_bc.set_title("(B) Fraser Valley Epicenter (BC)", fontsize=12.0, fontweight='bold', color='#1e3a8a', pad=8)
    ax_bc.set_xlabel("Longitude", fontsize=11.0, fontweight='bold')
    ax_bc.set_ylabel("Latitude", fontsize=11.0, fontweight='bold')
    ax_bc.grid(True, linestyle=':', alpha=0.5)
    ax_bc.tick_params(labelsize=10.5)
    ax_bc.legend(loc='upper right', fontsize=10.0, frameon=True, facecolor='#ffffff')

    # --- PANEL C: Southwestern Ontario Zoom ---
    on_bounds = (-83.0, 42.3, -78.5, 45.0)
    gdf_on = data_gdf.cx[on_bounds[0]:on_bounds[2], on_bounds[1]:on_bounds[3]]
    gdf_on.plot(ax=ax_on, column='percentile_risk', cmap=cmap_risk, norm=norm,
                edgecolor='#334155', linewidth=0.50, zorder=2)
    gdf_cd.plot(ax=ax_on, facecolor='none', edgecolor='#64748b', linewidth=0.9, zorder=3)

    pos_on = csd_pos[(csd_pos['lon'] >= -83.0) & (csd_pos['lon'] <= -78.5) &
                     (csd_pos['lat'] >= 42.3) & (csd_pos['lat'] <= 45.0)]
    ax_on.scatter(pos_on['lon'], pos_on['lat'], marker='*', s=130, facecolor='#06b6d4',
                  edgecolor='#000000', linewidth=1.0, zorder=5, label='Actual Outbreak')

    ax_on.set_xlim(-83.0, -78.5)
    ax_on.set_ylim(42.3, 45.0)
    ax_on.set_xticks([-82.0, -80.0])
    ax_on.set_xticklabels(['82.0°W', '80.0°W'], fontsize=10.5, fontweight='bold')
    ax_on.set_yticks([43.0, 44.0])
    ax_on.set_yticklabels(['43.0°N', '44.0°N'], fontsize=10.5, fontweight='bold')
    ax_on.set_title("(C) Southwestern Ontario Belt", fontsize=12.0, fontweight='bold', color='#991b1b', pad=8)
    ax_on.set_xlabel("Longitude", fontsize=11.0, fontweight='bold')
    ax_on.set_ylabel("Latitude", fontsize=11.0, fontweight='bold')
    ax_on.grid(True, linestyle=':', alpha=0.5)
    ax_on.tick_params(labelsize=10.5)
    ax_on.legend(loc='upper right', fontsize=10.0, frameon=True, facecolor='#ffffff')

    fig.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.08)

    for d in OUT_DIRS:
        safe_savefig(fig, os.path.join(d, "Figure11_nationwide_risk_map.png"), dpi=300)
        safe_savefig(fig, os.path.join(d, "Figure11_nationwide_risk_map.pdf"))
        safe_savefig(fig, os.path.join(d, "Risk_Map_GraphSAGE.png"), dpi=300)
        safe_savefig(fig, os.path.join(d, "Risk_Map_GraphSAGE.pdf"))

    plt.close()
    print("Master Figure 11 successfully generated!")

if __name__ == '__main__':
    generate_figure11_master()
