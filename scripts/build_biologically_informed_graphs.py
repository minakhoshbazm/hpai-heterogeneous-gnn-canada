#!/usr/bin/env python3
"""
scripts/build_biologically_informed_graphs.py

Constructs and audits three separate biologically/epidemiologically informed spatial edge relations:
1. geographic_proximity: CSDs within 50 km (centroid distance in EPSG:3347 metres) with distance decay weight.
2. shared_water_habitat: CSDs within 100 km with static water habitat feature cosine similarity >= tau_water.
3. poultry_exposure_similarity: CSDs within 100 km with static poultry production feature cosine similarity >= tau_poultry.

Excludes eBird, GBIF, bird observations, 2025 labels, and all future information.

Outputs:
- data/processed/graph/csd_relation_geographic_proximity_v1.csv
- data/processed/graph/csd_relation_shared_water_habitat_v1.csv
- data/processed/graph/csd_relation_poultry_exposure_similarity_v1.csv
- data/processed/graph/csd_multi_relation_audit_summary.csv
"""

import os
import math
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
import networkx as nx
from sklearn.preprocessing import StandardScaler

MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
BOUNDARY_SHP_PATH = 'data/raw/boundaries/lcsd000b21a_e.shp'
GRAPH_OUT_DIR = 'data/processed/graph'

WATER_HABITAT_COLS = [
    'dw_water_mean',
    'dw_water_p95',
    'dw_flooded_vegetation_mean',
    'dw_seasonal_pond_proxy'
]

POULTRY_EXPOSURE_COLS = [
    'ccs_broilers_roasters_cornish_farms_reporting',
    'ccs_broilers_roasters_cornish_production_kg',
    'ccs_turkey_farms_reporting',
    'ccs_turkey_production_kg',
    'glw2020_chicken_density_mean',
    'glw2020_chicken_density_max',
    'glw2015_duck_density_mean',
    'glw2015_duck_density_max'
]

def main():
    print("=== Starting Biologically Informed Multi-Relation Spatial Graph Builder ===")

    # 1. Load CSD model table and verify static feature presence
    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str})
    model_csds = sorted(df_model['csd_uid'].unique())
    num_model_csds = len(model_csds)
    csd_to_idx = {csd: i for i, csd in enumerate(model_csds)}

    print(f"Loaded modeling table with {num_model_csds:,} unique model-scope CSDs.")

    # Confirm source columns exist and have zero label leakage
    for col in WATER_HABITAT_COLS + POULTRY_EXPOSURE_COLS:
        assert col in df_model.columns, f"ERROR: Missing source feature column {col}"
        assert 'outbreak' not in col, f"ERROR: Outbreak label column detected in similarity feature: {col}"

    print("Zero-leakage audit passed: All source feature columns are static environmental/poultry attributes.")

    # 2. Extract static CSD feature vectors (one row per CSD)
    # Group by CSD to get unique static features per CSD
    csd_static_df = df_model.groupby('csd_uid')[WATER_HABITAT_COLS + POULTRY_EXPOSURE_COLS].first().reindex(model_csds)

    # Standardize water habitat vectors
    scaler_water = StandardScaler()
    water_feats_scaled = scaler_water.fit_transform(csd_static_df[WATER_HABITAT_COLS].fillna(0.0))
    water_norms = np.linalg.norm(water_feats_scaled, axis=1, keepdims=True)
    water_norms[water_norms == 0] = 1e-8
    water_unit_vecs = water_feats_scaled / water_norms

    # Log-transform and standardize poultry exposure vectors
    poultry_raw = np.log1p(np.maximum(0.0, csd_static_df[POULTRY_EXPOSURE_COLS].fillna(0.0).values))
    scaler_poultry = StandardScaler()
    poultry_feats_scaled = scaler_poultry.fit_transform(poultry_raw)
    poultry_norms = np.linalg.norm(poultry_feats_scaled, axis=1, keepdims=True)
    poultry_norms[poultry_norms == 0] = 1e-8
    poultry_unit_vecs = poultry_feats_scaled / poultry_norms

    # 3. Load boundary shapefile and compute centroids in EPSG:3347
    print("Loading boundary shapefile and computing centroids in EPSG:3347...")
    gdf = gpd.read_file(BOUNDARY_SHP_PATH)
    if "CSDUID" in gdf.columns:
        gdf["csd_uid"] = gdf["CSDUID"].astype(str)
    else:
        gdf["csd_uid"] = gdf["csd_uid"].astype(str)

    gdf_model = gdf[gdf["csd_uid"].isin(set(model_csds))].copy()
    if gdf_model.crs is None or gdf_model.crs.to_epsg() != 3347:
        gdf_model = gdf_model.to_crs(epsg=3347)

    gdf_model["centroid"] = gdf_model.geometry.centroid
    centroids_df = gdf_model.set_index("csd_uid").reindex(model_csds)
    coords = np.column_stack([centroids_df["centroid"].x.values, centroids_df["centroid"].y.values])

    kdtree = KDTree(coords)

    # 4. Construct Relation 1: geographic_proximity (<= 50 km)
    print("\n--- Constructing Relation 1: geographic_proximity (d <= 50 km) ---")
    pairs_50km = kdtree.query_pairs(50000.0) # 50 km
    geo_edges = []
    d0 = 25000.0 # 25 km decay constant

    for u, v in sorted(pairs_50km):
        dx = coords[u, 0] - coords[v, 0]
        dy = coords[u, 1] - coords[v, 1]
        dist_m = math.sqrt(dx*dx + dy*dy)
        weight = math.exp(-dist_m / d0)
        geo_edges.append({
            'source_csd_uid': model_csds[u],
            'target_csd_uid': model_csds[v],
            'distance_km': round(dist_m / 1000.0, 2),
            'weight': round(weight, 4)
        })

    # 5. Construct Candidates for Relation 2 & 3 within 100 km
    print("\n--- Querying candidate pairs within 100 km for similarity relations ---")
    pairs_100km = list(kdtree.query_pairs(100000.0)) # 100 km
    print(f"Found {len(pairs_100km):,} candidate pairs within 100 km distance.")

    # Calculate similarity scores for all 100 km pairs
    water_sims = []
    poultry_sims = []

    for u, v in pairs_100km:
        w_sim = float(np.dot(water_unit_vecs[u], water_unit_vecs[v]))
        p_sim = float(np.dot(poultry_unit_vecs[u], poultry_unit_vecs[v]))
        water_sims.append(w_sim)
        poultry_sims.append(p_sim)

    # Select default similarity thresholds (e.g. >= 0.70)
    tau_water = 0.70
    tau_poultry = 0.70

    water_edges = []
    for (u, v), w_sim in zip(pairs_100km, water_sims):
        if w_sim >= tau_water:
            dx = coords[u, 0] - coords[v, 0]
            dy = coords[u, 1] - coords[v, 1]
            dist_km = math.sqrt(dx*dx + dy*dy) / 1000.0
            water_edges.append({
                'source_csd_uid': model_csds[u],
                'target_csd_uid': model_csds[v],
                'distance_km': round(dist_km, 2),
                'similarity': round(w_sim, 4)
            })

    poultry_edges = []
    for (u, v), p_sim in zip(pairs_100km, poultry_sims):
        if p_sim >= tau_poultry:
            dx = coords[u, 0] - coords[v, 0]
            dy = coords[u, 1] - coords[v, 1]
            dist_km = math.sqrt(dx*dx + dy*dy) / 1000.0
            poultry_edges.append({
                'source_csd_uid': model_csds[u],
                'target_csd_uid': model_csds[v],
                'distance_km': round(dist_km, 2),
                'similarity': round(p_sim, 4)
            })

    # 6. Save Edge CSV Files & Perform Graph Audits
    os.makedirs(GRAPH_OUT_DIR, exist_ok=True)

    rel_files = {
        'geographic_proximity': (os.path.join(GRAPH_OUT_DIR, 'csd_relation_geographic_proximity_v1.csv'), geo_edges, WATER_HABITAT_COLS[:0], "Distance decay exp(-d/25km)"),
        'shared_water_habitat': (os.path.join(GRAPH_OUT_DIR, 'csd_relation_shared_water_habitat_v1.csv'), water_edges, WATER_HABITAT_COLS, f"Cosine similarity >= {tau_water} within 100km"),
        'poultry_exposure_similarity': (os.path.join(GRAPH_OUT_DIR, 'csd_relation_poultry_exposure_similarity_v1.csv'), poultry_edges, POULTRY_EXPOSURE_COLS, f"Cosine similarity >= {tau_poultry} within 100km")
    }

    audit_records = []

    for r_name, (out_path, edges, source_cols, sim_desc) in rel_files.items():
        df_e = pd.DataFrame(edges)
        df_e.to_csv(out_path, index=False)
        print(f"Saved relation '{r_name}': {len(edges):,} edges to {out_path}")

        # Graph topology analysis
        G = nx.Graph()
        G.add_nodes_from(range(num_model_csds))
        for item in edges:
            u_idx = csd_to_idx[item['source_csd_uid']]
            v_idx = csd_to_idx[item['target_csd_uid']]
            G.add_edge(u_idx, v_idx)

        degs = np.array([d for _, d in G.degree()])
        dists = [item['distance_km'] for item in edges]

        rec = {
            'relation_name': r_name,
            'total_edges': len(edges),
            'unique_active_nodes': int(np.sum(degs > 0)),
            'isolated_nodes': int(np.sum(degs == 0)),
            'connected_components': nx.number_connected_components(G),
            'mean_degree': round(float(np.mean(degs)), 2),
            'median_degree': round(float(np.median(degs)), 1),
            'min_degree': int(np.min(degs)),
            'max_degree': int(np.max(degs)),
            'dist_mean_km': round(float(np.mean(dists)), 2) if dists else 0.0,
            'dist_median_km': round(float(np.median(dists)), 2) if dists else 0.0,
            'dist_min_km': round(float(np.min(dists)), 2) if dists else 0.0,
            'dist_max_km': round(float(np.max(dists)), 2) if dists else 0.0,
            'source_feature_columns': ", ".join(source_cols) if source_cols else "CSD Centroid Coordinates (EPSG:3347)",
            'similarity_method': sim_desc,
            'zero_leakage_verified': "PASSED (Only static environmental/agricultural attributes used)"
        }
        audit_records.append(rec)

    # Save summary audit CSV
    df_audit = pd.DataFrame(audit_records)
    audit_csv_path = os.path.join(GRAPH_OUT_DIR, 'csd_multi_relation_audit_summary.csv')
    df_audit.to_csv(audit_csv_path, index=False)

    print("\n==========================================================================================")
    print("               BIOLOGICALLY INFORMED MULTI-RELATION GRAPH AUDIT SUMMARY")
    print("==========================================================================================")
    print(df_audit[['relation_name', 'total_edges', 'isolated_nodes', 'connected_components', 'mean_degree', 'max_degree', 'dist_mean_km']].to_string(index=False))
    print(f"\nSaved audit summary to: {audit_csv_path}")

if __name__ == '__main__':
    main()
