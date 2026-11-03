import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
SHAPEFILE_PATH = ROOT / "data/processed/gee_assets/canada_csd_2021_boundaries_for_gee_simplified_100m_shp/canada_csd_2021_boundaries_for_gee_simplified_100m.shp"
FIG_DIR = ROOT / "results/manuscript_draft/figures/final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_figure_2():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, dtype={"csd_uid": str})
    
    # Get all positive CSDs
    df_pos = df[df['outbreak_next_month'] == 1].copy()
    
    df_pos['Period'] = 'Train (2022-2023)'
    df_pos.loc[df_pos['year'] == 2024, 'Period'] = 'Validation (2024)'
    df_pos.loc[df_pos['year'] == 2025, 'Period'] = 'Test (2025)'
    
    pos_csds = df_pos.drop_duplicates(subset=['csd_uid', 'Period'])
    
    print("Loading shapefile...")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    if gdf.crs != "EPSG:3347":
        gdf = gdf.to_crs("EPSG:3347")
        
    gdf['centroid'] = gdf.geometry.centroid
    
    # Merge
    pos_gdf = gdf.merge(pos_csds[['csd_uid', 'Period']], on='csd_uid', how='inner')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot all CSDs lightly
    gdf.plot(ax=ax, color='whitesmoke', edgecolor='lightgray', linewidth=0.2)
    
    # Plot outbreaks
    colors = {'Train (2022-2023)': '#1f77b4', 'Validation (2024)': '#ff7f0e', 'Test (2025)': '#2ca02c'}
    
    for period, color in colors.items():
        subset = pos_gdf[pos_gdf['Period'] == period]
        subset.centroid.plot(ax=ax, color=color, markersize=15, label=period, alpha=0.8)
        
    ax.legend(title="Outbreak Period", loc='lower left', fontsize=10)
    ax.set_title("Figure 2. Distribution of HPAI Poultry Outbreaks across Canadian CSDs", fontsize=14)
    ax.axis('off')
    
    # Inset for kNN-3 Graph Topology
    # Let's pick Southern Ontario/Quebec region using bounding box
    minx, miny, maxx, maxy = 6500000, 800000, 8000000, 1500000  # approx coordinates for EPSG:3347
    
    axins = ax.inset_axes([0.65, 0.55, 0.33, 0.33])
    
    inset_gdf = gdf.cx[minx:maxx, miny:maxy]
    inset_gdf.plot(ax=axins, color='white', edgecolor='gray', linewidth=0.5)
    
    # Build kNN-3 for the inset
    if len(inset_gdf) > 3:
        coords = np.array([(pt.x, pt.y) for pt in inset_gdf.centroid])
        nn = NearestNeighbors(n_neighbors=4) # 4 because 1st is self
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        
        # Plot edges
        for i in range(len(coords)):
            for j in indices[i][1:]:
                axins.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], color='blue', linewidth=0.5, alpha=0.5)
                
        # Highlight some nodes
        axins.scatter(coords[:, 0], coords[:, 1], color='red', s=5, zorder=3)
        
    axins.set_xlim(minx, maxx)
    axins.set_ylim(miny, maxy)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Inset: kNN-3 Geographic Proximity Graph", fontsize=10)
    
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(FIG_DIR / f"Figure2_canada_graph.{ext}", bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 2 generated.")

if __name__ == "__main__":
    plot_figure_2()
