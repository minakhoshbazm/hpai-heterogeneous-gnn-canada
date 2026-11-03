import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors

def generate_figure2():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    print("Generating Figure 2 with simplified geometries...")
    # Load and simplify boundaries
    cd_path = "data/processed/boundaries/census_divisions_wgs84.geojson"
    gdf_cd = gpd.read_file(cd_path)
    gdf_cd['geometry'] = gdf_cd['geometry'].simplify(0.03, preserve_topology=True)
    
    # Load model table for coordinates & positives
    data_path = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
    df = pd.read_csv(data_path, dtype={"csd_uid": str})
    
    # Use crosswalk or GEE shapefile centroids
    # To be fast and exact, we read centroids
    shp_path = "data/processed/gee_assets/canada_csd_2021_boundaries_for_gee_simplified_100m_shp/canada_csd_2021_boundaries_for_gee_simplified_100m.shp"
    gdf_csd = gpd.read_file(shp_path)
    # Project to 3347 first then get centroid lon/lat
    cents = gdf_csd.to_crs("EPSG:4326").geometry.centroid
    gdf_csd_pts = pd.DataFrame({
        'csd_uid': gdf_csd['csd_uid'].astype(str),
        'lon': cents.x,
        'lat': cents.y
    })

    # Outbreak positives
    df_pos = df[df['outbreak_next_month'] == 1].copy()
    df_pos['Period'] = 'Training (2022–2023)'
    df_pos.loc[df_pos['year'] == 2024, 'Period'] = 'Validation (2024)'
    df_pos.loc[df_pos['year'] == 2025, 'Period'] = 'Locked Test (2025)'
    
    pos_unique = df_pos.drop_duplicates(subset=['csd_uid', 'Period'])
    pos_merged = pos_unique.merge(gdf_csd_pts, on='csd_uid', how='inner')
    
    # Figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.8), dpi=300, gridspec_kw={'width_ratios': [1.3, 1.0]})
    
    # --- PANEL A: Canada Map ---
    gdf_cd.plot(ax=ax1, color='#f8fafc', edgecolor='#94a3b8', linewidth=0.7, zorder=1)
    
    colors = {
        'Training (2022–2023)': '#2563eb',   # Royal Blue
        'Validation (2024)': '#f59e0b',      # Amber
        'Locked Test (2025)': '#dc2626'      # Crimson
    }
    markers = {
        'Training (2022–2023)': 'o',
        'Validation (2024)': 's',
        'Locked Test (2025)': '^'
    }
    sizes = {
        'Training (2022–2023)': 45,
        'Validation (2024)': 55,
        'Locked Test (2025)': 70
    }
    
    for period, c in colors.items():
        sub = pos_merged[pos_merged['Period'] == period]
        ax1.scatter(sub['lon'], sub['lat'], c=c, s=sizes[period], marker=markers[period],
                    label=f"{period} ({len(sub)} unique CSDs)", edgecolors='black', linewidth=0.6, alpha=0.9, zorder=3)
        
    ax1.set_xlim(-141, -52)
    ax1.set_ylim(41.5, 70)
    ax1.set_title("(A) Geographic Distribution of Confirmed Poultry Outbreaks Across Canada (2022–2025)",
                  fontsize=13, fontweight='bold', pad=12, color='#0f172a')
    ax1.set_xlabel("Longitude (°W)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Latitude (°N)", fontsize=11, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cbd5e1', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    
    # Inset rectangle for Southern Ontario / St. Lawrence
    rect_lon_min, rect_lon_max = -83.5, -73.0
    rect_lat_min, rect_lat_max = 42.0, 46.5
    rect = plt.Rectangle((rect_lon_min, rect_lat_min), rect_lon_max - rect_lon_min, rect_lat_max - rect_lat_min,
                         fill=False, edgecolor='#7c3aed', linewidth=2.0, linestyle='--', zorder=4)
    ax1.add_patch(rect)
    ax1.text(rect_lon_min - 2.0, rect_lat_max + 0.6, "Network Inset (Panel B)", color='#7c3aed',
             fontsize=11, fontweight='bold', zorder=5)

    # --- PANEL B: Regional Spatial Proximity Network ---
    gdf_cd.plot(ax=ax2, color='#f1f5f9', edgecolor='#94a3b8', linewidth=0.9, zorder=1)
    
    region_csds = gdf_csd_pts[(gdf_csd_pts['lon'] >= rect_lon_min) & (gdf_csd_pts['lon'] <= rect_lon_max) &
                              (gdf_csd_pts['lat'] >= rect_lat_min) & (gdf_csd_pts['lat'] <= rect_lat_max)].copy()
    
    coords = region_csds[['lon', 'lat']].values
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Collect line segments for fast LineCollection
    lines = []
    for i, nbr_list in enumerate(indices):
        p1 = coords[i]
        for j in nbr_list[1:]:
            p2 = coords[j]
            lines.append([p1, p2])
            
    lc = LineCollection(lines, colors='#60a5fa', linewidths=0.9, alpha=0.7, zorder=2)
    ax2.add_collection(lc)
    
    # Plot CSD nodes
    ax2.scatter(region_csds['lon'], region_csds['lat'], s=20, c='#64748b', alpha=0.6,
                label=f'CSD Centroid Node (N={len(region_csds)})', zorder=3)
    
    # Plot regional outbreaks
    reg_pos = pos_merged[(pos_merged['lon'] >= rect_lon_min) & (pos_merged['lon'] <= rect_lon_max) &
                         (pos_merged['lat'] >= rect_lat_min) & (pos_merged['lat'] <= rect_lat_max)]
    
    for period, c in colors.items():
        sub = reg_pos[reg_pos['Period'] == period]
        if len(sub) > 0:
            ax2.scatter(sub['lon'], sub['lat'], c=c, s=sizes[period]*1.3, marker=markers[period],
                        edgecolors='black', linewidth=0.7, alpha=0.95, zorder=4)

    ax2.set_xlim(rect_lon_min, rect_lon_max)
    ax2.set_ylim(rect_lat_min, rect_lat_max)
    ax2.set_title("(B) Spatial kNN-3 Message Passing Network (Ontario/Quebec Agricultural Belt)",
                  fontsize=13, fontweight='bold', pad=12, color='#0f172a')
    ax2.set_xlabel("Longitude (°W)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Latitude (°N)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    
    ax2.text(0.03, 0.95, "kNN-3 Spatial Graph Architecture:\n• Blue edges: Message passing channels (k=3)\n• Gray dots: CSD Centroid nodes\n• Colored symbols: Outbreak locations",
             transform=ax2.transAxes, fontsize=10.5, fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#60a5fa', alpha=0.95), zorder=5)

    plt.tight_layout()
    
    out_dirs = [
        "results/manuscript_elsevier/figures",
        "results/manuscript_latex/figures",
        "results/final_manuscript/figures"
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "Figure2_canada_graph.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(d, "Figure2_canada_graph.pdf"), bbox_inches='tight')
    plt.close()
    print("Figure 2 generated successfully in seconds!")

if __name__ == '__main__':
    generate_figure2()
