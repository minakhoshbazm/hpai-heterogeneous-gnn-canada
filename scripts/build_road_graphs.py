#!/usr/bin/env python3
"""
scripts/build_road_graphs.py

Constructs road-informed spatial edge relations:
1. Road-Connectivity Graph (Gravity Model): Top 3 road connections per CSD.
2. Hybrid Graph: Union of Geographic kNN-3 and Road-Connectivity Graph.

Outputs:
- data/processed/graph/csd_graph_road_corridor_v1.csv
- data/processed/graph/csd_graph_hybrid_knn3_road_v1.csv
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree

MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
BOUNDARY_SHP_PATH = 'data/raw/boundaries/lcsd000b21a_e.shp'
ROAD_FEATURES_PATH = 'data/processed/roads/csd_road_features.csv'
KNN3_GRAPH_PATH = 'data/processed/graph/csd_graph_knn_3_v1.csv'
GRAPH_OUT_DIR = 'data/processed/graph'

def main():
    print("=== Building Road-Informed Spatial Graphs ===")

    # 1. Load CSDs from model table
    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str})
    model_csds = sorted(df_model['csd_uid'].unique())
    num_model_csds = len(model_csds)
    csd_set = set(model_csds)
    
    print(f"Loaded {num_model_csds} model CSDs.")

    # 2. Load Road Features
    df_roads = pd.read_csv(ROAD_FEATURES_PATH, dtype={'csd_uid': str})
    df_roads = df_roads[df_roads['csd_uid'].isin(csd_set)].set_index('csd_uid')
    
    # Compute Road Mass (M) for Gravity Model
    # Emphasize highways, but also include major roads
    M = (df_roads['highway_length_km'].fillna(0) + 
         0.5 * df_roads['major_road_length_km'].fillna(0)).reindex(model_csds).fillna(0).values

    # 3. Load centroids
    print("Loading boundaries...")
    gdf = gpd.read_file(BOUNDARY_SHP_PATH)
    if "CSDUID" in gdf.columns:
        gdf["csd_uid"] = gdf["CSDUID"].astype(str)
    else:
        gdf["csd_uid"] = gdf["csd_uid"].astype(str)

    gdf_model = gdf[gdf["csd_uid"].isin(csd_set)].copy()
    if gdf_model.crs is None or gdf_model.crs.to_epsg() != 3347:
        gdf_model = gdf_model.to_crs(epsg=3347)

    gdf_model["centroid"] = gdf_model.geometry.centroid
    centroids_df = gdf_model.set_index("csd_uid").reindex(model_csds)
    coords = np.column_stack([centroids_df["centroid"].x.values, centroids_df["centroid"].y.values])

    kdtree = KDTree(coords)

    # 4. Find all pairs within 100km and calculate Gravity weights
    print("Querying pairs within 100km...")
    pairs_100km = kdtree.query_pairs(100000.0) # 100 km radius
    
    # We want directed edges to easily pick top-K per CSD, then symmetrize
    # pairs_100km returns (i, j) where i < j. Let's make it symmetric.
    directed_pairs = []
    for u, v in pairs_100km:
        directed_pairs.append((u, v))
        directed_pairs.append((v, u))
        
    print(f"Total directed candidate pairs within 100km: {len(directed_pairs)}")
    
    # Calculate weights
    weights = []
    for u, v in directed_pairs:
        if M[u] > 0 and M[v] > 0:
            dx = coords[u, 0] - coords[v, 0]
            dy = coords[u, 1] - coords[v, 1]
            dist_km = math.sqrt(dx*dx + dy*dy) / 1000.0
            # Add 1km to avoid div by zero for extremely small/overlapping CSDs
            weight = (M[u] * M[v]) / ((dist_km + 1.0) ** 2)
            weights.append((u, v, weight))
            
    # 5. Select Top 3 Road Neighbors per CSD
    # Group by source CSD
    road_edges_dict = {}
    for u, v, w in weights:
        if u not in road_edges_dict:
            road_edges_dict[u] = []
        road_edges_dict[u].append((v, w))
        
    selected_road_edges = set()
    for u, neighbors in road_edges_dict.items():
        # Sort by weight descending
        neighbors.sort(key=lambda x: x[1], reverse=True)
        # Take top 3
        top_k = neighbors[:3]
        for v, w in top_k:
            selected_road_edges.add((u, v))
            selected_road_edges.add((v, u)) # symmetrize
            
    print(f"Generated {len(selected_road_edges)} symmetric road-connectivity edges.")
    
    # 6. Save Road-Connectivity Graph
    os.makedirs(GRAPH_OUT_DIR, exist_ok=True)
    road_csv_path = os.path.join(GRAPH_OUT_DIR, 'csd_graph_road_corridor_v1.csv')
    df_road_edges = pd.DataFrame([
        {'source_csd_uid': model_csds[u], 'target_csd_uid': model_csds[v]}
        for u, v in selected_road_edges
    ])
    df_road_edges.to_csv(road_csv_path, index=False)
    print(f"Saved to {road_csv_path}")
    
    # 7. Create and Save Hybrid Graph
    df_knn3 = pd.read_csv(KNN3_GRAPH_PATH, dtype=str)
    knn3_edges = set(zip(df_knn3['source_csd_uid'], df_knn3['target_csd_uid']))
    
    hybrid_edges = knn3_edges.union(set((model_csds[u], model_csds[v]) for u, v in selected_road_edges))
    
    hybrid_csv_path = os.path.join(GRAPH_OUT_DIR, 'csd_graph_hybrid_knn3_road_v1.csv')
    df_hybrid_edges = pd.DataFrame([
        {'source_csd_uid': u, 'target_csd_uid': v}
        for u, v in hybrid_edges
    ])
    df_hybrid_edges.to_csv(hybrid_csv_path, index=False)
    print(f"Generated {len(hybrid_edges)} hybrid (kNN-3 + road) edges.")
    print(f"Saved to {hybrid_csv_path}")

if __name__ == '__main__':
    main()
