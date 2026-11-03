import geopandas as gpd
import pandas as pd
import time

t0 = time.time()
gdf_cd = gpd.read_file("data/processed/boundaries/census_divisions_epsg3347.gpkg")
print(f"GPKG loaded in {time.time()-t0:.2f}s, rows={len(gdf_cd)}, CRS={gdf_cd.crs}")

t1 = time.time()
shp_path = "data/processed/gee_assets/canada_csd_2021_boundaries_for_gee_simplified_100m_shp/canada_csd_2021_boundaries_for_gee_simplified_100m.shp"
gdf_csd = gpd.read_file(shp_path)
print(f"CSD SHP loaded in {time.time()-t1:.2f}s, rows={len(gdf_csd)}, CRS={gdf_csd.crs}")

cents = gdf_csd.geometry.centroid
df_cents = pd.DataFrame({
    'csd_uid': gdf_csd['csd_uid'].astype(str),
    'x': cents.x,
    'y': cents.y
})
df_cents.to_csv("data/processed/boundaries/csd_centroids_epsg3347.csv", index=False)
print("Saved precomputed centroids to data/processed/boundaries/csd_centroids_epsg3347.csv!")
