"""
Generate Comprehensive Risk Maps for Each Model (GraphSAGE, XGBoost, Random Forest, MLP, LR, GraphSAGE-NoLag).

Outputs generated:
1. Combined Multi-Model Comparison Map (6 models on identical spatial scale)
2. Regional Epicenter Zoom Comparison (Fraser Valley BC & SW Ontario across models)
3. Individual Standalone Publication Maps for each model (Nationwide + Insets)

Saved to results/model_risk_maps/ and scratch/
"""

import os
import time
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as patches

# Publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'results', 'model_risk_maps')
SCRATCH_DIR = os.path.join(BASE_DIR, 'scratch')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

TARGET_DIRS = [
    OUT_DIR,
    os.path.join(BASE_DIR, 'results', 'manuscript_elsevier'),
    os.path.join(BASE_DIR, 'results', 'manuscript_elsevier', 'figures'),
    os.path.join(BASE_DIR, 'results', 'final_manuscript', 'figures'),
    os.path.join(BASE_DIR, 'results', 'manuscript_latex'),
    os.path.join(BASE_DIR, 'results', 'manuscript_latex', 'figures'),
    SCRATCH_DIR
]

def safe_savefig(fig, path, **kwargs):
    """Save figure with retry logic for network/cloud drive resilience."""
    for attempt in range(5):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            time.sleep(0.5)
    print(f"Warning: could not save {path}")

def save_to_all(fig, basename, dpi=300):
    """Save figure once and copy to all relevant manuscript and results directories."""
    primary_dir = TARGET_DIRS[0]
    os.makedirs(primary_dir, exist_ok=True)
    p_png = os.path.join(primary_dir, f"{basename}.png")
    p_pdf = os.path.join(primary_dir, f"{basename}.pdf")
    safe_savefig(fig, p_png, dpi=dpi, bbox_inches='tight')
    safe_savefig(fig, p_pdf, bbox_inches='tight')
    
    for d in TARGET_DIRS[1:]:
        os.makedirs(d, exist_ok=True)
        try:
            shutil.copyfile(p_png, os.path.join(d, f"{basename}.png"))
            shutil.copyfile(p_pdf, os.path.join(d, f"{basename}.pdf"))
        except OSError as e:
            print(f"Warning copying to {d}: {e}")

def main():
    print("Loading geographic boundaries and CSD polygons...")
    # Load simplified CSD polygons for whole-cell choropleths
    shp_path = os.path.join(BASE_DIR, "data", "processed", "gee_assets", "canada_csd_2021_boundaries_for_gee_simplified_100m_shp", "canada_csd_2021_boundaries_for_gee_simplified_100m.shp")
    gdf_csd = gpd.read_file(shp_path)
    gdf_csd['csd_uid'] = gdf_csd['csd_uid'].astype(str)
    gdf_csd['geometry'] = gdf_csd['geometry'].simplify(0.008, preserve_topology=False)
    
    cd_path = os.path.join(BASE_DIR, "data", "processed", "boundaries", "census_divisions_wgs84.geojson")
    gdf_cd = gpd.read_file(cd_path)
    gdf_cd['geometry'] = gdf_cd['geometry'].simplify(0.03, preserve_topology=False)
    
    cents_path = os.path.join(BASE_DIR, "data", "processed", "boundaries", "csd_centroids_epsg3347.csv")
    cents_df = pd.read_csv(cents_path, dtype={'csd_uid': str})
    cents_df.rename(columns={'x': 'lon', 'y': 'lat'}, inplace=True)
    
    # Model catalog with clean publication titles (no long descriptions or within-panel metrics)
    models_config = [
        {
            'key': 'GraphSAGE',
            'name': 'GraphSAGE',
            'file': 'preds_GraphSAGE.csv',
            'color': '#0284c7'
        },
        {
            'key': 'XGB',
            'name': 'XGBoost',
            'file': 'preds_XGB.csv',
            'color': '#059669'
        },
        {
            'key': 'RF',
            'name': 'Random Forest',
            'file': 'preds_RF.csv',
            'color': '#d97706'
        },
        {
            'key': 'MLP',
            'name': 'MLP',
            'file': 'preds_MLP.csv',
            'color': '#7c3aed'
        },
        {
            'key': 'LR',
            'name': 'Logistic Regression',
            'file': 'preds_LR.csv',
            'color': '#64748b'
        },
        {
            'key': 'GraphSAGE_nolag',
            'name': 'GraphSAGE (No History)',
            'file': 'preds_GraphSAGE_nolag.csv',
            'color': '#0369a1'
        }
    ]

    print("Loading predictions and ground truth outbreaks for 2025...")
    model_data = {}
    ground_truth_pos = None

    for m in models_config:
        fpath = os.path.join(BASE_DIR, 'results', 'final_manuscript', 'canonical_predictions', m['file'])
        df = pd.read_csv(fpath, dtype={'csd_uid': str})
        df25 = df[df['year'] == 2025].copy()
        
        if ground_truth_pos is None:
            pos_df = df25[df25['outbreak_next_month'] == 1].drop_duplicates(subset=['csd_uid']).copy()
            ground_truth_pos = pos_df.merge(cents_df, on='csd_uid', how='inner')
            print(f"Loaded {len(ground_truth_pos)} unique confirmed 2025 outbreak CSDs.")
            
        risk_csd = df25.groupby('csd_uid')['pred_prob_ensemble'].max().reset_index()
        risk_csd.rename(columns={'pred_prob_ensemble': 'raw_risk'}, inplace=True)
        
        # Calculate risk percentile (0 to 100)
        risk_csd['percentile_risk'] = risk_csd['raw_risk'].rank(pct=True) * 100.0
        
        # Merge directly with CSD polygons for full cell choropleth
        merged = gdf_csd.merge(risk_csd, on='csd_uid', how='left')
        merged['raw_risk'] = merged['raw_risk'].fillna(0.0)
        merged['percentile_risk'] = merged['percentile_risk'].fillna(0.0)
        
        model_data[m['key']] = {
            'config': m,
            'gdf': merged
        }

    # Sleek high-contrast scientific colormap: cream -> pale yellow -> orange -> red -> deep mahogany
    colors = ['#f8fafc', '#e2e8f0', '#fef08a', '#f97316', '#dc2626', '#7f1d1d']
    nodes = [0.0, 0.65, 0.80, 0.90, 0.96, 1.0]
    cmap_risk = LinearSegmentedColormap.from_list('RiskCellCmap', list(zip(nodes, colors)), N=256)
    norm = Normalize(vmin=0.0, vmax=100.0)

    import sys
    run_comparison = '--regional-only' not in sys.argv

    # ==============================================================================
    # 1. COMBINED MULTI-MODEL COMPARISON FIGURE (6-Panel Nationwide Grid)
    # ==============================================================================
    if run_comparison:
        print("Generating Figure: Model Comparison Nationwide Whole-Cell Risk Maps (6 panels)...")
        fig, axes = plt.subplots(3, 2, figsize=(8.5, 9.8), dpi=300)
        axes = axes.flatten()

        for idx, m in enumerate(models_config):
            ax = axes[idx]
            ax.set_facecolor('#f0f7ff')
            data_gdf = model_data[m['key']]['gdf']
            
            # Whole cell CSD polygon choropleth
            data_gdf.plot(ax=ax, column='percentile_risk', cmap=cmap_risk, norm=norm,
                          edgecolor='#94a3b8', linewidth=0.20, zorder=2)
            
            # Overlay Census Division administrative boundaries
            gdf_cd.plot(ax=ax, facecolor='none', edgecolor='#475569', linewidth=0.55, zorder=3)
            
            # Ground truth 2025 outbreaks overlay (Distinct black ring with cyan center)
            ax.scatter(ground_truth_pos['lon'], ground_truth_pos['lat'], marker='o',
                       s=44, facecolor='#06b6d4', edgecolor='#000000', linewidth=0.9,
                       alpha=0.95, zorder=5, label='Confirmed Outbreak' if idx == 0 else "")

            # Inset highlight boxes and labels only on Panel A (idx == 0) to avoid repetitive clutter across panels
            if idx == 0:
                bc_box = patches.Rectangle((-123.5, 48.9), 2.2, 1.0, fill=False,
                                           edgecolor='#2563eb', linewidth=1.4, linestyle='--', zorder=6)
                ax.add_patch(bc_box)

                on_box = patches.Rectangle((-83.0, 42.1), 4.5, 2.7, fill=False,
                                           edgecolor='#b91c1c', linewidth=1.4, linestyle='--', zorder=6)
                ax.add_patch(on_box)

                ax.annotate("Fraser Valley", xy=(-123.5, 49.2), xytext=(-133.5, 47.0),
                            ha='center', va='center', color='#1d4ed8', fontsize=9.5, fontweight='normal', zorder=7,
                            arrowprops=dict(arrowstyle="->", color='#2563eb', lw=1.1),
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', edgecolor='#2563eb', linewidth=0.8, alpha=0.95))

                ax.annotate("SW Ontario", xy=(-81.5, 42.4), xytext=(-87.0, 40.2),
                            ha='center', va='center', color='#b91c1c', fontsize=9.5, fontweight='normal', zorder=7,
                            arrowprops=dict(arrowstyle="->", color='#b91c1c', lw=1.1),
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', edgecolor='#b91c1c', linewidth=0.8, alpha=0.95))

            # Framed to encompass all 5,054 CSDs and all 10 Canadian provinces (38.5°N to 63.5°N)
            ax.set_xlim(-139, -52.5)
            ax.set_ylim(38.5, 63.5)
            
            panel_letter = chr(ord('A') + idx)
            ax.set_title(f"({panel_letter}) {m['name']}",
                         fontsize=12.0, fontweight='normal', pad=7, color='#0f172a')
            
            ax.set_xticks([-120, -100, -80, -60])
            if idx >= 4:
                ax.set_xticklabels(['120°W', '100°W', '80°W', '60°W'], fontsize=10.0, fontweight='normal')
                ax.set_xlabel("Longitude", fontsize=11.5, fontweight='normal', labelpad=6)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")
                
            ax.set_yticks([40, 50, 60])
            if idx % 2 == 0:
                ax.set_yticklabels(['40°N', '50°N', '60°N'], fontsize=10.0, fontweight='normal')
                ax.set_ylabel("Latitude", fontsize=11.5, fontweight='normal', labelpad=5)
                ax.tick_params(axis='y', labelsize=10.0)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
                ax.tick_params(axis='y', left=False)
                
            ax.grid(True, linestyle=':', alpha=0.45, color='#cbd5e1')

        # Shared colorbar at the bottom with clear spacing below x-axis labels
        cbar_ax = fig.add_axes([0.18, 0.035, 0.64, 0.020])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_risk)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Prospective 2025 Outbreak Risk Percentile Tier across Canadian CSDs\n(0% = Lowest Risk, 100% = Highest Risk)',
                       fontsize=11.0, fontweight='normal', labelpad=7)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        cbar.ax.tick_params(labelsize=10.5)

        # Add shared legend in upper left corner of panel 0 where open ocean resides
        axes[0].legend(loc='upper left', fontsize=10.5, frameon=True, facecolor='#ffffff', edgecolor='#94a3b8')

        plt.subplots_adjust(left=0.065, right=0.97, top=0.965, bottom=0.135, wspace=0.10, hspace=0.22)
        
        save_to_all(fig, "Figure_Model_Comparison_Risk_Maps")
        save_to_all(fig, "Figure13_model_comparison_risk_maps")
        plt.close()
        print("Multi-model comparison whole-cell figure saved to all locations!")
        
        if '--comparison-only' in sys.argv:
            print("Comparison-only mode: exiting successfully.")
            return

    # ==============================================================================
    # 2. REGIONAL EPICENTER ZOOM COMPARISON (Fraser Valley & SW Ontario)
    # ==============================================================================
    print("Generating Figure: Regional Epicenter Comparison across Models (Whole-Cell)...")
    top_models = [m for m in models_config if m['key'] in ['GraphSAGE', 'XGB', 'RF', 'MLP', 'LR']]
    num_m = len(top_models)

    fig, axes = plt.subplots(2, num_m, figsize=(4.5 * num_m, 10.2), dpi=300)
    
    for c_idx, m in enumerate(top_models):
        data_gdf = model_data[m['key']]['gdf']

        # --- ROW 1: FRASER VALLEY, BC ---
        ax_bc = axes[0, c_idx]
        ax_bc.set_facecolor('#f0f7ff')
        
        # Filter CSDs for BC bounding box
        bc_bounds = (-123.5, 48.9, -121.3, 49.9)
        gdf_bc = data_gdf.cx[bc_bounds[0]:bc_bounds[2], bc_bounds[1]:bc_bounds[3]]
        gdf_bc.plot(ax=ax_bc, column='percentile_risk', cmap=cmap_risk, norm=norm,
                    edgecolor='#334155', linewidth=0.45, zorder=2)
        
        gdf_cd.plot(ax=ax_bc, facecolor='none', edgecolor='#64748b', linewidth=0.9, zorder=3)
        
        # Outbreaks in BC
        pos_bc = ground_truth_pos[(ground_truth_pos['lon'] >= -123.5) & (ground_truth_pos['lon'] <= -121.3) &
                                  (ground_truth_pos['lat'] >= 48.9) & (ground_truth_pos['lat'] <= 49.9)]
        ax_bc.scatter(pos_bc['lon'], pos_bc['lat'], marker='*', s=140, facecolor='#06b6d4',
                      edgecolor='#000000', linewidth=1.1, zorder=5)

        ax_bc.set_xlim(-123.5, -121.3)
        ax_bc.set_ylim(48.9, 49.9)
        ax_bc.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax_bc.yaxis.set_major_locator(plt.MaxNLocator(4))
        
        # Prominent model name header (26pt) and clean region subtitle (14pt)
        ax_bc.set_title(m['name'], fontsize=26, fontweight='normal', color='#0f172a', pad=24)
        ax_bc.text(0.5, 1.02, 'Fraser Valley (BC)', transform=ax_bc.transAxes,
                    ha='center', va='bottom', fontsize=14, fontweight='normal', color='#1e3a8a')
        
        ax_bc.set_xlabel("Longitude (°W)", fontsize=13, fontweight='normal', labelpad=6)
        if c_idx == 0:
            ax_bc.set_ylabel("Fraser Valley (BC)\nLatitude (°N)", fontsize=16, fontweight='normal', labelpad=10)
        ax_bc.grid(True, linestyle=':', alpha=0.45)
        ax_bc.tick_params(labelsize=11)

        # --- ROW 2: SOUTHWESTERN ONTARIO ---
        ax_on = axes[1, c_idx]
        ax_on.set_facecolor('#f0f7ff')
        
        # Filter CSDs for ON bounding box
        on_bounds = (-83.0, 42.3, -78.5, 45.0)
        gdf_on = data_gdf.cx[on_bounds[0]:on_bounds[2], on_bounds[1]:on_bounds[3]]
        gdf_on.plot(ax=ax_on, column='percentile_risk', cmap=cmap_risk, norm=norm,
                    edgecolor='#334155', linewidth=0.45, zorder=2)
        
        gdf_cd.plot(ax=ax_on, facecolor='none', edgecolor='#64748b', linewidth=0.9, zorder=3)
        
        # Outbreaks in ON
        pos_on = ground_truth_pos[(ground_truth_pos['lon'] >= -83.0) & (ground_truth_pos['lon'] <= -78.5) &
                                  (ground_truth_pos['lat'] >= 42.3) & (ground_truth_pos['lat'] <= 45.0)]
        ax_on.scatter(pos_on['lon'], pos_on['lat'], marker='*', s=140, facecolor='#06b6d4',
                      edgecolor='#000000', linewidth=1.1, zorder=5)

        ax_on.set_xlim(-83.0, -78.5)
        ax_on.set_ylim(42.3, 45.0)
        ax_on.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax_on.yaxis.set_major_locator(plt.MaxNLocator(4))
        
        # Prominent model name header (26pt) and clean region subtitle (14pt)
        ax_on.set_title(m['name'], fontsize=26, fontweight='normal', color='#0f172a', pad=24)
        ax_on.text(0.5, 1.02, 'SW Ontario Poultry Belt', transform=ax_on.transAxes,
                    ha='center', va='bottom', fontsize=14, fontweight='normal', color='#991b1b')
        
        ax_on.set_xlabel("Longitude (°W)", fontsize=13, fontweight='normal', labelpad=6)
        if c_idx == 0:
            ax_on.set_ylabel("SW Ontario Poultry Belt\nLatitude (°N)", fontsize=16, fontweight='normal', labelpad=10)
        ax_on.grid(True, linestyle=':', alpha=0.45)
        ax_on.tick_params(labelsize=11)

    # Shared colorbar for regional zoom
    cbar_ax2 = fig.add_axes([0.25, 0.035, 0.50, 0.022])
    sm2 = plt.cm.ScalarMappable(norm=norm, cmap=cmap_risk)
    cbar2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Model Risk Percentile Tier across CSD Cells (★ = Confirmed 2025 Outbreak)', fontsize=15.0, fontweight='normal', labelpad=8)
    cbar2.set_ticks([0, 25, 50, 65, 80, 90, 95, 100])
    cbar2.set_ticklabels(['0%', '25%', '50%', '65%', '80%', '90%', '95%', '100%'])
    cbar2.ax.tick_params(labelsize=13)

    plt.subplots_adjust(left=0.055, right=0.98, top=0.89, bottom=0.12, wspace=0.20, hspace=0.42)
    
    save_to_all(fig, "Figure_Regional_Epicenter_Comparison")
    plt.close()
    print("Regional epicenter comparison whole-cell figure saved to all locations!")

    if '--regional-only' in sys.argv:
        print("Regional-only mode: exiting successfully.")
        return

    # ==============================================================================
    # 3. INDIVIDUAL STANDALONE PUBLICATION FIGURES FOR EACH MODEL
    # ==============================================================================
    print("Generating Individual Standalone Whole-Cell Risk Maps for each model...")
    for m in models_config:
        data_gdf = model_data[m['key']]['gdf']
        
        fig = plt.figure(figsize=(9.2, 8.8), dpi=300)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.16, 1.0], width_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
        ax_main = fig.add_subplot(gs[0, :])
        ax_cbar_slot = fig.add_subplot(gs[1, :])
        ax_cbar_slot.axis('off')
        ax_bc = fig.add_subplot(gs[2, 0])
        ax_on = fig.add_subplot(gs[2, 1])
        
        for ax in [ax_main, ax_bc, ax_on]:
            ax.set_facecolor('#f0f7ff')
        
        # --- PANEL A: Nationwide Whole-Cell Choropleth ---
        data_gdf.plot(ax=ax_main, column='percentile_risk', cmap=cmap_risk, norm=norm,
                      edgecolor='#94a3b8', linewidth=0.20, zorder=2)
        
        gdf_cd.plot(ax=ax_main, facecolor='none', edgecolor='#475569', linewidth=0.55, zorder=3)
        
        # Ground truth outbreaks
        ax_main.scatter(ground_truth_pos['lon'], ground_truth_pos['lat'], marker='o',
                        s=40, facecolor='#06b6d4', edgecolor='#000000', linewidth=0.9,
                        alpha=0.95, zorder=5, label='Confirmed 2025 Outbreak (N=68 CSDs)')

        # Colorbar placed in dedicated horizontal slot between Row 0 and Row 2
        cbar_ax = fig.add_axes([0.22, 0.472, 0.56, 0.020])
        sm_ind = plt.cm.ScalarMappable(norm=norm, cmap=cmap_risk)
        cbar = fig.colorbar(sm_ind, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Prospective 2025 Outbreak Risk Percentile Tier', fontsize=10.0, fontweight='bold', labelpad=5, color='#0f172a')
        cbar.set_ticks([0, 25, 50, 75, 100])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        cbar.ax.tick_params(labelsize=9.5)

        ax_main.set_xlim(-141, -52)
        ax_main.set_ylim(41.5, 70)
        ax_main.set_xticks([-130, -110, -90, -70])
        ax_main.set_xticklabels(['130°W', '110°W', '90°W', '70°W'], fontsize=10.5, fontweight='bold')
        ax_main.set_yticks([45, 55, 65])
        ax_main.set_yticklabels(['45°N', '55°N', '65°N'], fontsize=10.5, fontweight='bold')
        ax_main.set_title(f"(A) Nationwide Prospective 2025 Outbreak Risk: {m['name']}",
                          fontsize=12.5, fontweight='bold', pad=8, color='#0f172a')
        ax_main.set_xlabel("")
        ax_main.set_ylabel("Latitude", fontsize=11.0, fontweight='bold', labelpad=4)
        ax_main.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
        ax_main.legend(loc='upper left', fontsize=10.0, frameon=True, facecolor='#ffffff', edgecolor='#94a3b8')

        # Inset indicator boxes with clean panel badges
        bc_box = patches.Rectangle((-123.5, 48.9), 2.2, 1.0, fill=False, edgecolor='#1d4ed8', linewidth=1.5, linestyle='--', zorder=6)
        ax_main.add_patch(bc_box)
        ax_main.text(-124.0, 49.4, "(B)", fontsize=10.5, fontweight='bold', color='#1d4ed8', ha='right', va='center', zorder=7,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', edgecolor='#1d4ed8', linewidth=0.8, alpha=0.95))

        on_box = patches.Rectangle((-83.0, 42.1), 4.5, 2.7, fill=False, edgecolor='#b91c1c', linewidth=1.5, linestyle='--', zorder=6)
        ax_main.add_patch(on_box)
        ax_main.text(-83.4, 43.4, "(C)", fontsize=10.5, fontweight='bold', color='#b91c1c', ha='right', va='center', zorder=7,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', edgecolor='#b91c1c', linewidth=0.8, alpha=0.95))

        # --- PANEL B: Fraser Valley BC Zoom Whole Cell ---
        bc_bounds = (-123.5, 48.9, -121.3, 49.9)
        gdf_bc = data_gdf.cx[bc_bounds[0]:bc_bounds[2], bc_bounds[1]:bc_bounds[3]]
        gdf_bc.plot(ax=ax_bc, column='percentile_risk', cmap=cmap_risk, norm=norm,
                    edgecolor='#334155', linewidth=0.50, zorder=2)
        gdf_cd.plot(ax=ax_bc, facecolor='none', edgecolor='#64748b', linewidth=0.9, zorder=3)
        
        pos_bc = ground_truth_pos[(ground_truth_pos['lon'] >= -123.5) & (ground_truth_pos['lon'] <= -121.3) &
                                  (ground_truth_pos['lat'] >= 48.9) & (ground_truth_pos['lat'] <= 49.9)]
        ax_bc.scatter(pos_bc['lon'], pos_bc['lat'], marker='*', s=130, facecolor='#06b6d4',
                      edgecolor='#000000', linewidth=1.0, zorder=5, label='Actual Outbreak')

        ax_bc.set_xlim(-123.5, -121.3)
        ax_bc.set_ylim(48.9, 49.9)
        ax_bc.set_xticks([-123.0, -122.0])
        ax_bc.set_xticklabels(['123.0°W', '122.0°W'], fontsize=10.0, fontweight='bold')
        ax_bc.set_yticks([49.0, 49.5])
        ax_bc.set_yticklabels(['49.0°N', '49.5°N'], fontsize=10.0, fontweight='bold')
        ax_bc.set_aspect('auto')
        ax_bc.set_box_aspect(0.82)
        ax_bc.set_title("(B) Fraser Valley Epicenter (BC)", fontsize=11.5, fontweight='bold', color='#1e3a8a', pad=8)
        ax_bc.set_xlabel("Longitude", fontsize=10.5, fontweight='bold')
        ax_bc.set_ylabel("Latitude", fontsize=10.5, fontweight='bold')
        ax_bc.grid(True, linestyle=':', alpha=0.5)
        ax_bc.tick_params(labelsize=10.0)
        ax_bc.legend(loc='upper right', fontsize=9.5, frameon=True, facecolor='#ffffff')

        # --- PANEL C: Southwestern Ontario Zoom Whole Cell ---
        on_bounds = (-83.0, 42.3, -78.5, 45.0)
        gdf_on = data_gdf.cx[on_bounds[0]:on_bounds[2], on_bounds[1]:on_bounds[3]]
        gdf_on.plot(ax=ax_on, column='percentile_risk', cmap=cmap_risk, norm=norm,
                    edgecolor='#334155', linewidth=0.50, zorder=2)
        gdf_cd.plot(ax=ax_on, facecolor='none', edgecolor='#64748b', linewidth=0.9, zorder=3)

        pos_on = ground_truth_pos[(ground_truth_pos['lon'] >= -83.0) & (ground_truth_pos['lon'] <= -78.5) &
                                  (ground_truth_pos['lat'] >= 42.3) & (ground_truth_pos['lat'] <= 45.0)]
        ax_on.scatter(pos_on['lon'], pos_on['lat'], marker='*', s=130, facecolor='#06b6d4',
                      edgecolor='#000000', linewidth=1.0, zorder=5, label='Actual Outbreak')

        ax_on.set_xlim(-83.0, -78.5)
        ax_on.set_ylim(42.3, 45.0)
        ax_on.set_xticks([-82.0, -80.0])
        ax_on.set_xticklabels(['82.0°W', '80.0°W'], fontsize=10.0, fontweight='bold')
        ax_on.set_yticks([43.0, 44.0])
        ax_on.set_yticklabels(['43.0°N', '44.0°N'], fontsize=10.0, fontweight='bold')
        ax_on.set_aspect('auto')
        ax_on.set_box_aspect(0.82)
        ax_on.set_title("(C) Southwestern Ontario Belt", fontsize=11.5, fontweight='bold', color='#991b1b', pad=8)
        ax_on.set_xlabel("Longitude", fontsize=10.5, fontweight='bold')
        ax_on.set_ylabel("Latitude", fontsize=10.5, fontweight='bold')
        ax_on.grid(True, linestyle=':', alpha=0.5)
        ax_on.tick_params(labelsize=10.0)
        ax_on.legend(loc='upper right', fontsize=9.5, frameon=True, facecolor='#ffffff')

        fig.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.08)
        save_to_all(fig, f"Risk_Map_{m['key']}")
        if m['key'] == 'GraphSAGE':
            save_to_all(fig, "Figure11_nationwide_risk_map")
        plt.close()
        print(f"Standalone whole-cell risk map for {m['name']} generated!")

    print("All whole-cell risk map figures successfully generated!")

if __name__ == '__main__':
    main()

