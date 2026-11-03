import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
SHAPEFILE_PATH = ROOT / "data/processed/gee_assets/canada_csd_2021_boundaries_for_gee_simplified_100m_shp/canada_csd_2021_boundaries_for_gee_simplified_100m.shp"
OUT_FILE = ROOT / "results/manuscript_draft/distribution_shift_quant.txt"

def verify_shift():
    print("Loading modeling table...")
    df = pd.read_csv(DATA_PATH, dtype={"csd_uid": str})
    
    # 2024 positives
    val_pos = df[(df["year"] == 2024) & (df["outbreak_next_month"] == 1)]
    # 2025 positives
    test_pos = df[(df["year"] == 2025) & (df["outbreak_next_month"] == 1)]
    
    val_csds = val_pos["csd_uid"].unique()
    test_csds = test_pos["csd_uid"].unique()
    
    print(f"2024 unique positive CSDs: {len(val_csds)}")
    print(f"2025 unique positive CSDs: {len(test_csds)}")
    
    print("Loading shapefile...")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    # Ensure CRS is projected for accurate distance/area (e.g., EPSG:3347 Statistics Canada Lambert)
    if gdf.crs != "EPSG:3347":
        gdf = gdf.to_crs("EPSG:3347")
    
    # Calculate centroids
    gdf['centroid'] = gdf.geometry.centroid
    
    # Get centroids for 2024 and 2025
    val_gdf = gdf[gdf['csd_uid'].isin(val_csds)]
    test_gdf = gdf[gdf['csd_uid'].isin(test_csds)]
    
    def calc_standard_distance(centroids):
        if len(centroids) == 0:
            return 0
        mean_x = centroids.x.mean()
        mean_y = centroids.y.mean()
        # standard distance formula: sqrt( sum((x-mean_x)^2 + (y-mean_y)^2) / n )
        var_x = ((centroids.x - mean_x)**2).sum() / len(centroids)
        var_y = ((centroids.y - mean_y)**2).sum() / len(centroids)
        return np.sqrt(var_x + var_y) / 1000  # in km
        
    sd_2024 = calc_standard_distance(val_gdf['centroid'])
    sd_2025 = calc_standard_distance(test_gdf['centroid'])
    
    print(f"2024 Standard Distance: {sd_2024:.1f} km")
    print(f"2025 Standard Distance: {sd_2025:.1f} km")
    
    with open(OUT_FILE, 'w') as f:
        f.write(f"2024 Unique Positive CSDs: {len(val_csds)}\n")
        f.write(f"2025 Unique Positive CSDs: {len(test_csds)}\n")
        f.write(f"2024 Standard Distance: {sd_2024:.1f} km\n")
        f.write(f"2025 Standard Distance: {sd_2025:.1f} km\n")
        f.write(f"Is 2025 more dispersed? {'Yes' if sd_2025 > sd_2024 else 'No'}\n")

if __name__ == "__main__":
    verify_shift()
