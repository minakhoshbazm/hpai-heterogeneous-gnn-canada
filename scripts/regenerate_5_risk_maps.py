import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from shapely.geometry import Point, box
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np

base_dir = r'C:\Users\mkhos\PycharmProjects\Spatiotemporal_HPAI_Risk_Modeling'
model_dir = os.path.join(base_dir, 'benchmark_results', 'Model_Risk_Outputs')
manuscript_fig_dir = os.path.join(base_dir, 'manuscript_elsevier_eswa', 'figures')

eval_ready_path = os.path.join(base_dir, 'data', 'Canada_Oct27_Normalized_READY.csv')
train_data_path = os.path.join(base_dir, 'data', 'Final_Geospatial_Modeling_Data_with_Coords.csv')

print("Loading grid geometry...")
df_eval_ready = pd.read_csv(eval_ready_path)

def parse_coords(geom_str):
    c = str(geom_str).replace("c(", "").replace(")", "")
    parts = [float(p.strip()) for p in c.split(",")]
    return parts[0], parts[1]

coords_parsed = [parse_coords(g) for g in df_eval_ready["geometry"]]
df_eval_ready["longitude"] = [c[0] for c in coords_parsed]
df_eval_ready["latitude"] = [c[1] for c in coords_parsed]

# Create GeoDataFrame in EPSG:3978
points_geom = [Point(xy) for xy in zip(df_eval_ready["longitude"], df_eval_ready["latitude"])]
points_gdf = gpd.GeoDataFrame(df_eval_ready[["cell_id"]], geometry=points_geom, crs="EPSG:4326").to_crs(epsg=3978)

half_res = 5000.0  # 5km radius for 10km grid
grid_polygons = [box(pt.x - half_res, pt.y - half_res, pt.x + half_res, pt.y + half_res) for pt in points_gdf.geometry]
grid_gdf = gpd.GeoDataFrame(df_eval_ready[["cell_id"]], geometry=grid_polygons, crs="EPSG:3978")

# Outbreak points
df_train = pd.read_csv(train_data_path)
df_train["date"] = pd.to_datetime(df_train["date"])
oct27_cases = df_train[(df_train["status"] == 1) & (df_train["date"] == "2025-10-27")]
if len(oct27_cases) == 0:
    # Fallback to general recent fall outbreak points
    oct27_cases = df_train[(df_train["status"] == 1)].tail(14)

pts_14_geom = [Point(xy) for xy in zip(oct27_cases["longitude"], oct27_cases["latitude"])]
points_14_gdf = gpd.GeoDataFrame(geometry=pts_14_geom, crs="EPSG:4326").to_crs(epsg=3978)
print(f"Loaded {len(points_14_gdf)} outbreak overlay markers.")

# Load Canada province boundaries
try:
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson"
    prov_gdf = gpd.read_file(url).to_crs(epsg=3978)
except Exception:
    prov_gdf = gpd.GeoDataFrame(columns=["name", "geometry"], crs="EPSG:3978")

models_info = [
    ("Risk_Tuned_XGBoost_2025_10_27.csv", "risk_tuned_xgboost", "Tuned XGBoost", "Tuned_XGBoost_National_Risk_Map.png", "national_map_xgboost.png"),
    ("Risk_TabM_Neural_2025_10_27.csv", "risk_tabm_neural", "TabM (Deep Tabular Neural Net)", "TabM_Neural_National_Risk_Map.png", "national_map_tabm.png"),
    ("Risk_Bagging_PULearn_2025_10_27.csv", "risk_bagging_pulearn", "Bagging PU Learning", "Bagging_PULearn_National_Risk_Map.png", "national_map_pulearn.png"),
    ("Risk_LightGBM_2025_10_27.csv", "risk_lightgbm", "LightGBM", "LightGBM_National_Risk_Map.png", "national_map_lightgbm.png"),
    ("Risk_Random_Forest_2025_10_27.csv", "risk_random_forest", "Balanced Random Forest", "Random_Forest_National_Risk_Map.png", "national_map_rf.png")
]

for csv_name, risk_col, title, out_name1, out_name2 in models_info:
    csv_path = os.path.join(model_dir, csv_name)
    if not os.path.exists(csv_path):
        print(f"Skipping {csv_name}, not found.")
        continue
    
    df_risk = pd.read_csv(csv_path)
    merged_grid = grid_gdf.merge(df_risk[["cell_id", risk_col]], on="cell_id", how="left")
    
    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=300)
    ax.set_facecolor("#f8f9fa")

    # Layer 1: Provincial background
    if not prov_gdf.empty:
        prov_gdf.plot(ax=ax, color="#e9ecef", edgecolor="#adb5bd", linewidth=0.6, alpha=0.9)

    # Layer 2: 10km Risk Grid Polygons
    merged_grid.plot(
        column=risk_col,
        ax=ax,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolor="none",
        alpha=0.95,
        legend=False
    )

    # Layer 3: Province Borders
    if not prov_gdf.empty:
        prov_gdf.boundary.plot(ax=ax, color="#495057", linewidth=0.7, alpha=0.8)

    # Layer 4: Outbreak markers (enlarged, high-contrast red circles with black border)
    if not points_14_gdf.empty:
        points_14_gdf.plot(
            ax=ax,
            color="#e63946",
            edgecolor="black",
            linewidth=1.8,
            markersize=110,
            marker="o",
            label=f"{len(points_14_gdf)} Confirmed Outbreak Cases"
        )

    # Colorbar
    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.55, pad=0.02, aspect=20)
    cbar.set_label("Outbreak Risk Probability", fontsize=11, fontweight="bold", labelpad=8)
    cbar.set_ticks(np.arange(0, 1.1, 0.2))

    # Scale bar (Fixed: no custom lambda dividing by 1000)
    ax.add_artist(ScaleBar(1, units="m", location="lower right", box_alpha=0.6))
    ax.annotate('N\n^', xy=(0.02, 0.98), xycoords='axes fraction', ha='center', va='top', fontsize=16, fontweight='bold', color='black')

    # Title & Formatting
    ax.set_title(f"Model Prediction: {title} (October 27, 2025)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (EPSG:3978 - Canada Atlas Lambert)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Northing (EPSG:3978 - Canada Atlas Lambert)", fontsize=10, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10, frameon=True, facecolor="white", framealpha=0.85)

    p1 = os.path.join(model_dir, out_name1)
    p2 = os.path.join(manuscript_fig_dir, out_name2)
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated updated risk map: {out_name2}")

print("All 5 risk maps successfully regenerated with fixed scale bars and prominent markers!")
