import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize

def safe_save(fig, path, **kwargs):
    for attempt in range(5):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            time.sleep(0.5)
    print("Warning: could not save", path)

def generate_figure11():
    t0 = time.time()
    print("Generating Figure 11 (Figure 4 in manuscript) nationwide choropleth risk map...")

    # 1. Load CSD polygon boundaries
    shp_path = "data/processed/gee_assets/canada_csd_2021_boundaries_for_gee_simplified_100m_shp/canada_csd_2021_boundaries_for_gee_simplified_100m.shp"
    gdf = gpd.read_file(shp_path)
    gdf['csd_uid'] = gdf['csd_uid'].astype(str)

    # 2. Load GraphSAGE predictions
    preds_path = "results/final_manuscript/canonical_predictions/preds_GraphSAGE.csv"
    df_preds = pd.read_csv(preds_path, dtype={'csd_uid': str})
    df_2025 = df_preds[df_preds['year'] == 2025].copy()

    # Ground truth outbreaks for 2025
    outbreaks_2025 = df_2025[df_2025['outbreak_next_month'] == 1].drop_duplicates(subset=['csd_uid']).copy()
    cents_path = "data/processed/boundaries/csd_centroids_epsg3347.csv"
    df_cents = pd.read_csv(cents_path, dtype={'csd_uid': str})
    df_cents.rename(columns={'x': 'lon', 'y': 'lat'}, inplace=True)
    outbreaks_geo = outbreaks_2025.merge(df_cents, on='csd_uid', how='inner')

    # Max risk per CSD across 2025
    risk_csd = df_2025.groupby('csd_uid')['pred_prob_ensemble'].max().reset_index()
    risk_csd.rename(columns={'pred_prob_ensemble': 'raw_risk'}, inplace=True)
    risk_csd['percentile_risk'] = risk_csd['raw_risk'].rank(pct=True) * 100.0

    merged = gdf.merge(risk_csd, on='csd_uid', how='left')
    merged['percentile_risk'] = merged['percentile_risk'].fillna(0.0)
    merged['raw_risk'] = merged['raw_risk'].fillna(0.0)

    # Census divisions for province/regional context boundaries
    cd_path = "data/processed/boundaries/census_divisions_wgs84.geojson"
    gdf_cd = gpd.read_file(cd_path)
    gdf_cd['geometry'] = gdf_cd['geometry'].simplify(0.03, preserve_topology=True)

    # Colormap matching Figure 5: cream -> yellow -> orange -> red -> deep mahogany
    colors = ['#fef9c3', '#fde047', '#f97316', '#dc2626', '#7f1d1d']
    cmap_risk = LinearSegmentedColormap.from_list('RiskCmap', colors, N=256)
    norm = Normalize(vmin=85.0, vmax=100.0)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

    fig = plt.figure(figsize=(19, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.38, 1.0], height_ratios=[1, 1], wspace=0.14, hspace=0.24)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_bc = fig.add_subplot(gs[0, 1])
    ax_on = fig.add_subplot(gs[1, 1])

    # Water background for oceanic and lacustrine realism
    for ax in [ax_main, ax_bc, ax_on]:
        ax.set_facecolor('#f0f7ff')

    # --- PANEL A: Nationwide Choropleth ---
    low_risk = merged[merged['percentile_risk'] < 85.0]
    low_risk.plot(ax=ax_main, color='#f8fafc', edgecolor='#cbd5e1', linewidth=0.25, zorder=1)

    hi_risk = merged[merged['percentile_risk'] >= 85.0].sort_values('percentile_risk')
    hi_risk.plot(ax=ax_main, column='percentile_risk', cmap=cmap_risk, norm=norm,
                 edgecolor='#475569', linewidth=0.35, zorder=2)

    gdf_cd.plot(ax=ax_main, facecolor='none', edgecolor='#94a3b8', linewidth=0.55, zorder=3)

    # Confirmed 2025 outbreaks overlay
    ax_main.scatter(outbreaks_geo['lon'], outbreaks_geo['lat'], marker='o',
                    s=28, facecolor='#06b6d4', edgecolor='#000000', linewidth=0.8,
                    alpha=0.95, zorder=5, label='Confirmed 2025 Outbreak (N=79)')

    ax_main.set_xlim(-141, -52)
    ax_main.set_ylim(41.5, 70)
    ax_main.set_title('(A) Nationwide Prospective 2025 HPAI Outbreak Risk Profile Across Canadian CSDs',
                      fontsize=13, fontweight='bold', pad=10, color='#0f172a')
    ax_main.set_xlabel('Longitude (°W)', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold')
    ax_main.grid(True, linestyle=':', alpha=0.45, color='#cbd5e1')
    ax_main.legend(loc='lower left', fontsize=10, frameon=True, facecolor='#ffffff', edgecolor='#94a3b8')

    # Inset highlight boxes with clean non-overlapping label badges
    bc_box = patches.Rectangle((-123.5, 48.9), 2.2, 1.0, fill=False, edgecolor='#2563eb', linewidth=2.0, linestyle='--', zorder=6)
    ax_main.add_patch(bc_box)
    ax_main.text(-128.5, 49.4, 'Fraser Valley (B)', color='#1d4ed8', fontsize=10.5, fontweight='bold', zorder=7,
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#2563eb', linewidth=1.0, alpha=0.92))

    on_box = patches.Rectangle((-83.0, 42.3), 4.5, 2.7, fill=False, edgecolor='#b91c1c', linewidth=2.0, linestyle='--', zorder=6)
    ax_main.add_patch(on_box)
    ax_main.text(-86.8, 42.2, 'SW Ontario (C)', color='#b91c1c', fontsize=10.5, fontweight='bold', zorder=7,
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#b91c1c', linewidth=1.0, alpha=0.92))

    # --- PANEL B: Fraser Valley BC Choropleth Zoom ---
    merged_bc = merged.cx[-123.3:-121.6, 48.98:49.45].copy()
    merged_bc_low = merged_bc[merged_bc['percentile_risk'] < 85.0]
    merged_bc_hi = merged_bc[merged_bc['percentile_risk'] >= 85.0].sort_values('percentile_risk')

    merged_bc_low.plot(ax=ax_bc, color='#f1f5f9', edgecolor='#94a3b8', linewidth=0.6, zorder=1)
    if len(merged_bc_hi) > 0:
        merged_bc_hi.plot(ax=ax_bc, column='percentile_risk', cmap=cmap_risk, norm=norm,
                          edgecolor='#1e293b', linewidth=0.7, zorder=2)

    outbreaks_bc = outbreaks_geo[(outbreaks_geo['lon'] >= -123.3) & (outbreaks_geo['lon'] <= -121.6) &
                                 (outbreaks_geo['lat'] >= 48.98) & (outbreaks_geo['lat'] <= 49.45)]
    ax_bc.scatter(outbreaks_bc['lon'], outbreaks_bc['lat'], marker='o',
                  s=60, facecolor='#06b6d4', edgecolor='#000000', linewidth=1.1,
                  alpha=0.95, zorder=5)

    ax_bc.text(-122.30, 49.05, 'Abbotsford', fontsize=9.5, fontweight='bold', color='#ffffff',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='#0f172a', alpha=0.7, edgecolor='none'), zorder=6)
    ax_bc.text(-121.95, 49.16, 'Chilliwack', fontsize=9.5, fontweight='bold', color='#ffffff',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='#0f172a', alpha=0.7, edgecolor='none'), zorder=6)
    ax_bc.text(-122.65, 49.10, 'Langley', fontsize=9.5, fontweight='bold', color='#0f172a',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffff', alpha=0.8, edgecolor='#64748b'), zorder=6)

    ax_bc.set_xlim(-123.3, -121.6)
    ax_bc.set_ylim(48.98, 49.45)
    ax_bc.set_title('(B) High-Risk Epicenter: Fraser Valley Agricultural Corridor (BC)',
                    fontsize=11.5, fontweight='bold', color='#1e3a8a', pad=8)
    ax_bc.set_xlabel('Longitude (°W)', fontsize=9.5, fontweight='bold')
    ax_bc.set_ylabel('Latitude (°N)', fontsize=9.5, fontweight='bold')
    ax_bc.grid(True, linestyle=':', alpha=0.5)

    # --- PANEL C: Southwestern Ontario Choropleth Zoom ---
    merged_on = merged.cx[-83.0:-79.2, 42.3:44.7].copy()
    merged_on_low = merged_on[merged_on['percentile_risk'] < 85.0]
    merged_on_hi = merged_on[merged_on['percentile_risk'] >= 85.0].sort_values('percentile_risk')

    merged_on_low.plot(ax=ax_on, color='#f1f5f9', edgecolor='#94a3b8', linewidth=0.6, zorder=1)
    if len(merged_on_hi) > 0:
        merged_on_hi.plot(ax=ax_on, column='percentile_risk', cmap=cmap_risk, norm=norm,
                          edgecolor='#1e293b', linewidth=0.7, zorder=2)

    outbreaks_on = outbreaks_geo[(outbreaks_geo['lon'] >= -83.0) & (outbreaks_geo['lon'] <= -79.2) &
                                 (outbreaks_geo['lat'] >= 42.3) & (outbreaks_geo['lat'] <= 44.7)]
    ax_on.scatter(outbreaks_on['lon'], outbreaks_on['lat'], marker='o',
                  s=55, facecolor='#06b6d4', edgecolor='#000000', linewidth=1.1,
                  alpha=0.95, zorder=5)

    ax_on.text(-81.0, 43.45, 'Perth / Waterloo', fontsize=9.5, fontweight='bold', color='#ffffff',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='#0f172a', alpha=0.7, edgecolor='none'), zorder=6)
    ax_on.text(-80.8, 43.15, 'Oxford', fontsize=9.5, fontweight='bold', color='#ffffff',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='#0f172a', alpha=0.7, edgecolor='none'), zorder=6)
    ax_on.text(-82.2, 44.0, 'Huron', fontsize=9.5, fontweight='bold', color='#0f172a',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffff', alpha=0.8, edgecolor='#64748b'), zorder=6)
    ax_on.text(-82.0, 42.6, 'Chatham-Kent', fontsize=9.0, fontweight='bold', color='#0f172a',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffff', alpha=0.8, edgecolor='#64748b'), zorder=6)

    ax_on.set_xlim(-83.0, -79.2)
    ax_on.set_ylim(42.3, 44.7)
    ax_on.set_title('(C) High-Risk Epicenter: Southwestern Ontario Poultry Belt',
                    fontsize=11.5, fontweight='bold', color='#991b1b', pad=8)
    ax_on.set_xlabel('Longitude (°W)', fontsize=9.5, fontweight='bold')
    ax_on.set_ylabel('Latitude (°N)', fontsize=9.5, fontweight='bold')
    ax_on.grid(True, linestyle=':', alpha=0.5)

    # Shared Colorbar
    cbar_ax = fig.add_axes([0.15, 0.04, 0.40, 0.022])
    sm = plt.cm.ScalarMappable(cmap=cmap_risk, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Prospective 2025 Outbreak Risk Percentile Tier (Top Decile Focus: 85th – 100th Percentile)',
                   fontsize=11, fontweight='bold', labelpad=6)
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.05, right=0.96, top=0.95, bottom=0.12, wspace=0.12, hspace=0.25)

    out_dirs = [
        "results/manuscript_elsevier",
        "results/manuscript_elsevier/figures",
        "results/manuscript_latex/figures",
        "results/final_manuscript/figures"
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        safe_save(fig, os.path.join(d, "Figure11_nationwide_risk_map.png"), dpi=300, bbox_inches='tight')

    out_pdf = "results/manuscript_elsevier/Figure11_nationwide_risk_map.pdf"
    safe_save(fig, out_pdf, bbox_inches='tight')

    plt.close()
    print(f"Figure 11 / Figure 4 choropleth successfully generated in {time.time()-t0:.2f}s!")

if __name__ == '__main__':
    generate_figure11()
