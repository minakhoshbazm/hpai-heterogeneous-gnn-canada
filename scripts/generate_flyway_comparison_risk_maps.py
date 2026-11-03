#!/usr/bin/env python3
"""
scripts/generate_flyway_comparison_risk_maps.py

Generates:
1. results/model_risk_maps/Risk_Map_GraphSAGE_Flyway.pdf (.png): Standalone 3-panel risk map for Flyway GraphSAGE.
2. results/model_risk_maps/Risk_Map_Static_vs_Flyway_Comparison.pdf (.png): Head-to-head comparison figure
   directly comparing Static GraphSAGE vs Flyway-Dynamic GraphSAGE nationwide and across key agricultural corridors
   (Fraser Valley, BC and Southwestern Ontario).
"""

import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'results', 'model_risk_maps')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_DIRS = [
    OUT_DIR,
    os.path.join(BASE_DIR, 'results', 'manuscript_elsevier'),
    os.path.join(BASE_DIR, 'results', 'manuscript_elsevier', 'figures'),
    os.path.join(BASE_DIR, 'results', 'final_manuscript', 'figures'),
    os.path.join(BASE_DIR, 'results', 'manuscript_latex'),
    os.path.join(BASE_DIR, 'results', 'manuscript_latex', 'figures'),
    os.path.join(BASE_DIR, 'scratch')
]

def save_to_all(fig, basename, dpi=300):
    for d in TARGET_DIRS:
        os.makedirs(d, exist_ok=True)
        safe_savefig(fig, os.path.join(d, f"{basename}.png"), dpi=dpi)
        safe_savefig(fig, os.path.join(d, f"{basename}.pdf"))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def safe_savefig(fig, path, **kwargs):
    for attempt in range(5):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            time.sleep(0.5)
    print(f"Warning: could not save {path}")

def main():
    print("Loading geographic boundaries and CSD polygons...")
    shp_path = os.path.join(BASE_DIR, "data", "processed", "gee_assets", "canada_csd_2021_boundaries_for_gee_simplified_100m_shp", "canada_csd_2021_boundaries_for_gee_simplified_100m.shp")
    gdf_csd = gpd.read_file(shp_path)
    gdf_csd['csd_uid'] = gdf_csd['csd_uid'].astype(str).str.strip()
    gdf_csd['geometry'] = gdf_csd['geometry'].simplify(0.008, preserve_topology=False)

    cd_path = os.path.join(BASE_DIR, "data", "processed", "boundaries", "census_divisions_wgs84.geojson")
    gdf_cd = gpd.read_file(cd_path)
    gdf_cd['geometry'] = gdf_cd['geometry'].simplify(0.03, preserve_topology=False)

    cents_path = os.path.join(BASE_DIR, "data", "processed", "boundaries", "csd_centroids_epsg3347.csv")
    cents_df = pd.read_csv(cents_path, dtype={'csd_uid': str})
    cents_df['csd_uid'] = cents_df['csd_uid'].astype(str).str.strip()
    cents_df.rename(columns={'x': 'lon', 'y': 'lat'}, inplace=True)

    # 1. Load Static and Flyway Predictions
    p_static_path = os.path.join(BASE_DIR, "results", "final_manuscript", "canonical_predictions", "preds_GraphSAGE.csv")
    p_flyway_path = os.path.join(BASE_DIR, "results", "final_manuscript", "canonical_predictions", "preds_GraphSAGE_flyway.csv")

    df_static = pd.read_csv(p_static_path, dtype={'csd_uid': str})
    df_flyway = pd.read_csv(p_flyway_path, dtype={'csd_uid': str})

    df_static['csd_uid'] = df_static['csd_uid'].astype(str).str.strip()
    df_flyway['csd_uid'] = df_flyway['csd_uid'].astype(str).str.strip()

    df_static_25 = df_static[df_static['year'] == 2025].copy()
    df_flyway_25 = df_flyway[df_flyway['year'] == 2025].copy()

    # Ground truth positives
    pos_df = df_static_25[df_static_25['outbreak_next_month'] == 1].drop_duplicates(subset=['csd_uid']).copy()
    ground_truth_pos = pos_df.merge(cents_df, on='csd_uid', how='inner')
    print(f"Loaded {len(ground_truth_pos)} unique confirmed 2025 outbreak CSDs.")

    # Calculate max risk per CSD across 2025
    r_static = df_static_25.groupby('csd_uid')['pred_prob_ensemble'].max().reset_index().rename(columns={'pred_prob_ensemble': 'raw_risk_static'})
    r_flyway = df_flyway_25.groupby('csd_uid')['pred_prob_ensemble'].max().reset_index().rename(columns={'pred_prob_ensemble': 'raw_risk_flyway'})

    r_static['pct_static'] = r_static['raw_risk_static'].rank(pct=True) * 100.0
    r_flyway['pct_flyway'] = r_flyway['raw_risk_flyway'].rank(pct=True) * 100.0

    merged = gdf_csd.merge(r_static, on='csd_uid', how='left').merge(r_flyway, on='csd_uid', how='left')
    merged['pct_static'] = merged['pct_static'].fillna(0.0)
    merged['pct_flyway'] = merged['pct_flyway'].fillna(0.0)
    merged['pct_diff'] = merged['pct_flyway'] - merged['pct_static']

    # High-contrast colormap for risk percentiles
    colors = ['#f8fafc', '#e2e8f0', '#fef08a', '#f97316', '#dc2626', '#7f1d1d']
    nodes = [0.0, 0.65, 0.80, 0.90, 0.96, 1.0]
    cmap_risk = LinearSegmentedColormap.from_list('RiskCellCmap', list(zip(nodes, colors)), N=256)
    norm = Normalize(vmin=0.0, vmax=100.0)

    # ==============================================================================
    # 1. STANDALONE FLYWAY-DYNAMIC RISK MAP (Matches Risk_Map_GraphSAGE.pdf format)
    # ==============================================================================
    print("Generating Standalone Flyway-Dynamic Risk Map...")
    fig = plt.figure(figsize=(19, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.0, 1.0])
    ax_main = fig.add_subplot(gs[:, 0])
    ax_bc = fig.add_subplot(gs[0, 1])
    ax_on = fig.add_subplot(gs[1, 1])

    ax_main.set_facecolor('#f0f7ff')
    ax_bc.set_facecolor('#f0f7ff')
    ax_on.set_facecolor('#f0f7ff')

    # Panel A: Nationwide
    merged.plot(ax=ax_main, column='pct_flyway', cmap=cmap_risk, norm=norm,
                edgecolor='#94a3b8', linewidth=0.20, zorder=2)
    gdf_cd.plot(ax=ax_main, facecolor='none', edgecolor='#475569', linewidth=0.6, zorder=3)
    ax_main.scatter(ground_truth_pos['lon'], ground_truth_pos['lat'], marker='*',
                    s=60, facecolor='#06b6d4', edgecolor='#000000', linewidth=0.9,
                    alpha=0.95, zorder=5, label='Confirmed 2025 Outbreak')

    ax_main.set_xlim(-141, -52)
    ax_main.set_ylim(41.5, 70)
    ax_main.set_title("(A) Nationwide 2025 HPAI Risk Profile: Flyway-Dynamic GraphSAGE\n[1:5 Benchmark PR-AUC: 0.7360 | Full Domain PR-AUC: 0.1170 | Precision@10: 0.0818]",
                      fontsize=17, fontweight='bold', pad=12, color='#0f172a')
    ax_main.set_xlabel("Longitude (°W)", fontsize=15, fontweight='bold')
    ax_main.set_ylabel("Latitude (°N)", fontsize=15, fontweight='bold')
    ax_main.tick_params(labelsize=13)
    ax_main.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    ax_main.legend(loc='lower left', fontsize=13, frameon=True, facecolor='#ffffff', edgecolor='#94a3b8')

    # Inset boxes
    bc_box = patches.Rectangle((-123.5, 48.9), 2.2, 1.0, fill=False, edgecolor='#2563eb', linewidth=2.2, linestyle='--', zorder=6)
    ax_main.add_patch(bc_box)
    ax_main.text(-128.5, 49.4, "Fraser Valley (B)", color='#1d4ed8', fontsize=13, fontweight='bold', zorder=7,
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#2563eb', linewidth=1.0, alpha=0.92))

    on_box = patches.Rectangle((-83.0, 42.3), 4.5, 2.7, fill=False, edgecolor='#dc2626', linewidth=2.2, linestyle='--', zorder=6)
    ax_main.add_patch(on_box)
    ax_main.text(-84.8, 41.1, "SW Ontario (C)", color='#dc2626', fontsize=13, fontweight='bold', zorder=7,
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#dc2626', linewidth=1.0, alpha=0.92))

    # Colorbar
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("bottom", size="3.2%", pad=0.52)
    sm = plt.cm.ScalarMappable(cmap=cmap_risk, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Relative Epidemiological Risk Percentile (0 to 100)', fontsize=16, fontweight='bold', labelpad=7)
    cbar.set_ticks([0, 25, 50, 75, 90, 95, 99, 100])
    cbar.ax.tick_params(labelsize=13.5)

    # Panel B: Fraser Valley
    bc_bounds = (-123.5, 48.9, -121.3, 49.9)
    gdf_bc = merged.cx[bc_bounds[0]:bc_bounds[2], bc_bounds[1]:bc_bounds[3]]
    gdf_bc.plot(ax=ax_bc, column='pct_flyway', cmap=cmap_risk, norm=norm,
                edgecolor='#334155', linewidth=0.50, zorder=2)
    gdf_cd.plot(ax=ax_bc, facecolor='none', edgecolor='#64748b', linewidth=0.9, zorder=3)
    pos_bc = ground_truth_pos[(ground_truth_pos['lon'] >= -123.5) & (ground_truth_pos['lon'] <= -121.3) &
                              (ground_truth_pos['lat'] >= 48.9) & (ground_truth_pos['lat'] <= 49.9)]
    ax_bc.scatter(pos_bc['lon'], pos_bc['lat'], marker='*', s=180, facecolor='#06b6d4',
                  edgecolor='#000000', linewidth=1.2, zorder=5, label='Actual Outbreak')
    ax_bc.set_xlim(-123.5, -121.3)
    ax_bc.set_ylim(48.9, 49.9)
    ax_bc.set_title("(B) Fraser Valley Agricultural Corridor (BC)", fontsize=17.5, fontweight='bold', color='#1e3a8a', pad=9)
    ax_bc.set_xlabel("Longitude (°W)", fontsize=15.5, fontweight='bold')
    ax_bc.set_ylabel("Latitude (°N)", fontsize=15.5, fontweight='bold')
    ax_bc.grid(True, linestyle=':', alpha=0.5)
    ax_bc.tick_params(labelsize=13.5)
    ax_bc.legend(loc='lower left', fontsize=13, frameon=True)

    # Panel C: SW Ontario
    on_bounds = (-83.0, 42.3, -78.5, 45.0)
    gdf_on = merged.cx[on_bounds[0]:on_bounds[2], on_bounds[1]:on_bounds[3]]
    gdf_on.plot(ax=ax_on, column='pct_flyway', cmap=cmap_risk, norm=norm,
                edgecolor='#334155', linewidth=0.50, zorder=2)
    gdf_cd.plot(ax=ax_on, facecolor='none', edgecolor='#64748b', linewidth=0.9, zorder=3)
    pos_on = ground_truth_pos[(ground_truth_pos['lon'] >= -83.0) & (ground_truth_pos['lon'] <= -78.5) &
                              (ground_truth_pos['lat'] >= 42.3) & (ground_truth_pos['lat'] <= 45.0)]
    ax_on.scatter(pos_on['lon'], pos_on['lat'], marker='*', s=180, facecolor='#06b6d4',
                  edgecolor='#000000', linewidth=1.2, zorder=5, label='Actual Outbreak')
    ax_on.set_xlim(-83.0, -78.5)
    ax_on.set_ylim(42.3, 45.0)
    ax_on.set_title("(C) Southwestern Ontario Poultry Belt", fontsize=17.5, fontweight='bold', color='#991b1b', pad=9)
    ax_on.set_xlabel("Longitude (°W)", fontsize=15.5, fontweight='bold')
    ax_on.set_ylabel("Latitude (°N)", fontsize=15.5, fontweight='bold')
    ax_on.grid(True, linestyle=':', alpha=0.5)
    ax_on.tick_params(labelsize=13.5)
    ax_on.legend(loc='lower left', fontsize=13, frameon=True)

    plt.subplots_adjust(left=0.045, right=0.98, top=0.93, bottom=0.07, wspace=0.14, hspace=0.24)
    save_to_all(fig, "Risk_Map_GraphSAGE_Flyway")
    plt.close()
    print("Standalone Flyway-Dynamic Risk Map generated and saved to all locations!")

    # ==============================================================================
    # 2. DIRECT SIDE-BY-SIDE COMPARISON (Static vs Flyway-Dynamic)
    # ==============================================================================
    print("Generating Direct Side-by-Side Static vs Flyway Comparison Figure...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11.5), dpi=300)

    # Row 1: Static GraphSAGE (A1: Nationwide, A2: Fraser Valley, A3: SW Ontario)
    # Row 2: Flyway GraphSAGE (B1: Nationwide, B2: Fraser Valley, B3: SW Ontario)

    # --- A1: Static Nationwide ---
    ax = axes[0, 0]
    ax.set_facecolor('#f0f7ff')
    merged.plot(ax=ax, column='pct_static', cmap=cmap_risk, norm=norm,
                edgecolor='#94a3b8', linewidth=0.15, zorder=2)
    gdf_cd.plot(ax=ax, facecolor='none', edgecolor='#475569', linewidth=0.5, zorder=3)
    ax.scatter(ground_truth_pos['lon'], ground_truth_pos['lat'], marker='*',
               s=75, facecolor='#06b6d4', edgecolor='#000000', linewidth=1.0, zorder=5)
    ax.set_xlim(-141, -52); ax.set_ylim(41.5, 70)
    ax.set_title("(A1) Static GraphSAGE ($k$NN-3) — Nationwide\nPR-AUC: 0.7347  |  Full Domain: 0.1107",
                 fontsize=14.5, fontweight='normal', color='#1e293b', pad=8)
    ax.set_ylabel("Latitude (°N)", fontsize=13.5, fontweight='normal')
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)

    # --- B1: Flyway Nationwide ---
    ax = axes[1, 0]
    ax.set_facecolor('#f0f7ff')
    merged.plot(ax=ax, column='pct_flyway', cmap=cmap_risk, norm=norm,
                edgecolor='#94a3b8', linewidth=0.15, zorder=2)
    gdf_cd.plot(ax=ax, facecolor='none', edgecolor='#475569', linewidth=0.5, zorder=3)
    ax.scatter(ground_truth_pos['lon'], ground_truth_pos['lat'], marker='*',
               s=75, facecolor='#06b6d4', edgecolor='#000000', linewidth=1.0, zorder=5)
    ax.set_xlim(-141, -52); ax.set_ylim(41.5, 70)
    ax.set_title("(B1) Flyway-Dynamic GraphSAGE — Nationwide\nPR-AUC: 0.7360  |  Full Domain: 0.1170",
                 fontsize=14.5, fontweight='normal', color='#1e293b', pad=8)
    ax.set_xlabel("Longitude (°W)", fontsize=13.5, fontweight='normal')
    ax.set_ylabel("Latitude (°N)", fontsize=13.5, fontweight='normal')
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)

    # --- A2: Static Fraser Valley ---
    ax = axes[0, 1]
    ax.set_facecolor('#f0f7ff')
    gdf_bc.plot(ax=ax, column='pct_static', cmap=cmap_risk, norm=norm,
                edgecolor='#334155', linewidth=0.45, zorder=2)
    gdf_cd.plot(ax=ax, facecolor='none', edgecolor='#64748b', linewidth=0.8, zorder=3)
    ax.scatter(pos_bc['lon'], pos_bc['lat'], marker='*', s=180, facecolor='#06b6d4',
               edgecolor='#000000', linewidth=1.2, zorder=5)
    ax.set_xlim(-123.5, -121.3); ax.set_ylim(48.9, 49.9)
    ax.set_title("(A2) Static GraphSAGE\nFraser Valley (BC)", fontsize=14.5, fontweight='normal', color='#1e3a8a', pad=8)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)

    # --- B2: Flyway Fraser Valley ---
    ax = axes[1, 1]
    ax.set_facecolor('#f0f7ff')
    gdf_bc.plot(ax=ax, column='pct_flyway', cmap=cmap_risk, norm=norm,
                edgecolor='#334155', linewidth=0.45, zorder=2)
    gdf_cd.plot(ax=ax, facecolor='none', edgecolor='#64748b', linewidth=0.8, zorder=3)
    ax.scatter(pos_bc['lon'], pos_bc['lat'], marker='*', s=180, facecolor='#06b6d4',
               edgecolor='#000000', linewidth=1.2, zorder=5)
    ax.set_xlim(-123.5, -121.3); ax.set_ylim(48.9, 49.9)
    ax.set_title("(B2) Flyway GraphSAGE\nFraser Valley (BC)", fontsize=14.5, fontweight='normal', color='#1e3a8a', pad=8)
    ax.set_xlabel("Longitude (°W)", fontsize=13.5, fontweight='normal')
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)

    # --- A3: Static SW Ontario ---
    ax = axes[0, 2]
    ax.set_facecolor('#f0f7ff')
    gdf_on.plot(ax=ax, column='pct_static', cmap=cmap_risk, norm=norm,
                edgecolor='#334155', linewidth=0.45, zorder=2)
    gdf_cd.plot(ax=ax, facecolor='none', edgecolor='#64748b', linewidth=0.8, zorder=3)
    ax.scatter(pos_on['lon'], pos_on['lat'], marker='*', s=180, facecolor='#06b6d4',
               edgecolor='#000000', linewidth=1.2, zorder=5)
    ax.set_xlim(-83.0, -78.5); ax.set_ylim(42.3, 45.0)
    ax.set_title("(A3) Static GraphSAGE\nSW Ontario Belt", fontsize=14.5, fontweight='normal', color='#991b1b', pad=8)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)

    # --- B3: Flyway SW Ontario ---
    ax = axes[1, 2]
    ax.set_facecolor('#f0f7ff')
    gdf_on.plot(ax=ax, column='pct_flyway', cmap=cmap_risk, norm=norm,
                edgecolor='#334155', linewidth=0.45, zorder=2)
    gdf_cd.plot(ax=ax, facecolor='none', edgecolor='#64748b', linewidth=0.8, zorder=3)
    ax.scatter(pos_on['lon'], pos_on['lat'], marker='*', s=180, facecolor='#06b6d4',
               edgecolor='#000000', linewidth=1.2, zorder=5)
    ax.set_xlim(-83.0, -78.5); ax.set_ylim(42.3, 45.0)
    ax.set_title("(B3) Flyway GraphSAGE\nSW Ontario Belt", fontsize=14.5, fontweight='normal', color='#991b1b', pad=8)
    ax.set_xlabel("Longitude (°W)", fontsize=13.5, fontweight='normal')
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)

    # Global Colorbar at bottom
    cbar_ax = fig.add_axes([0.18, 0.052, 0.64, 0.018])
    sm = plt.cm.ScalarMappable(cmap=cmap_risk, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Relative Epidemiological Risk Percentile (0 to 100)', fontsize=14.0, fontweight='normal', labelpad=6)
    cbar.set_ticks([0, 25, 50, 75, 90, 100])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '90%', '100%'])
    cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(left=0.045, right=0.98, top=0.93, bottom=0.12, wspace=0.20, hspace=0.30)
    save_to_all(fig, "Risk_Map_Static_vs_Flyway_Comparison")
    save_to_all(fig, "Figure_flyway_risk_map_comparison")
    plt.close()
    print("Comparative Risk Map saved successfully to all locations!")

if __name__ == '__main__':
    main()
