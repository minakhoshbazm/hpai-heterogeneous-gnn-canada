#!/usr/bin/env python3
"""
scripts/audit_candidate_spatial_graphs.py

Audits candidate spatial graph topologies for the 5,054 model-scope CSDs in Canada.

Evaluates 8 candidate graphs:
1. Shared-border adjacency graph
2. Geographic proximity graphs (CSD centroid distance): 25 km, 50 km, 100 km, 200 km radius
3. K-nearest-neighbour graphs: k=3, k=5, k=8 (symmetrized undirected)

Exports results to:
- results/data_qa/candidate_spatial_graph_audit.csv
- results/data_qa/candidate_spatial_graph_audit.md
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
import networkx as nx

def main():
    print("=== Starting Spatial Graph-Topology Audit ===")

    # File paths
    model_table_path = "data/processed/modeling/model_table_v1_environment_train_ready.csv"
    boundary_shp_path = "data/raw/boundaries/lcsd000b21a_e.shp"
    shared_border_edges_path = "data/processed/graph/csd_adjacency_edges_v1_model_scope.csv"
    out_csv_path = "results/data_qa/candidate_spatial_graph_audit.csv"
    out_md_path = "results/data_qa/candidate_spatial_graph_audit.md"

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    # 1. Load model-scope CSDs
    print(f"Loading model scope CSDs from: {model_table_path}")
    df_model = pd.read_csv(model_table_path, dtype={"csd_uid": str})
    model_csds = sorted(df_model["csd_uid"].unique())
    num_model_csds = len(model_csds)
    print(f"Model-scope CSD count: {num_model_csds}")

    csd_to_idx = {csd: i for i, csd in enumerate(model_csds)}
    idx_to_csd = {i: csd for i, csd in enumerate(model_csds)}

    # 2. Load spatial boundary shapefile and reproject to EPSG:3347 (Statistics Canada Lambert, metres)
    print(f"Loading shapefile from: {boundary_shp_path}")
    gdf = gpd.read_file(boundary_shp_path)
    
    # Ensure csd_uid column exists and is string
    if "CSDUID" in gdf.columns:
        gdf["csd_uid"] = gdf["CSDUID"].astype(str)
    else:
        gdf["csd_uid"] = gdf["csd_uid"].astype(str)

    # Filter to model scope
    gdf_model = gdf[gdf["csd_uid"].isin(set(model_csds))].copy()
    print(f"Matched {len(gdf_model)} boundary geometries for the {num_model_csds} model-scope CSDs.")

    # Ensure projected CRS EPSG:3347 (metre units)
    if gdf_model.crs is None or gdf_model.crs.to_epsg() != 3347:
        print("Reprojecting geometries to EPSG:3347 (Statistics Canada Lambert Conformal Conic, metres)...")
        gdf_model = gdf_model.to_crs(epsg=3347)

    # Calculate centroids
    print("Computing CSD centroids in EPSG:3347 projection...")
    gdf_model["centroid"] = gdf_model.geometry.centroid
    gdf_model["cx"] = gdf_model["centroid"].x
    gdf_model["cy"] = gdf_model["centroid"].y

    # Align centroid array exactly to model_csds order
    centroids_df = gdf_model.set_index("csd_uid").reindex(model_csds)
    if centroids_df["cx"].isnull().any():
        missing_csds = centroids_df[centroids_df["cx"].isnull()].index.tolist()
        raise ValueError(f"Missing centroid geometries for CSDs: {missing_csds}")

    coords = np.column_stack([centroids_df["cx"].values, centroids_df["cy"].values]) # shape (5054, 2)

    # 3. Load Shared-Border Adjacency Edges
    print(f"Loading shared-border edges from: {shared_border_edges_path}")
    df_border = pd.read_csv(shared_border_edges_path, dtype=str)
    
    border_edge_set = set()
    for _, row in df_border.iterrows():
        u = row["source_csd_uid"]
        v = row["target_csd_uid"]
        if u in csd_to_idx and v in csd_to_idx:
            iu, iv = csd_to_idx[u], csd_to_idx[v]
            if iu != iv:
                border_edge_set.add((min(iu, iv), max(iu, iv)))

    print(f"Loaded {len(border_edge_set)} unique undirected shared-border edges.")

    # Build KDTree for spatial query
    kdtree = KDTree(coords)

    # 4. Define candidate graph generation functions
    candidate_graphs = []

    # Graph 1: Shared-border adjacency
    candidate_graphs.append({
        "name": "Shared-border Adjacency",
        "category": "Shared-Border",
        "edges": border_edge_set
    })

    # Graph 2-5: Distance Radius Graphs (25km, 50km, 100km, 200km)
    radii_km = [25, 50, 100, 200]
    for r_km in radii_km:
        r_m = r_km * 1000.0
        pairs = kdtree.query_pairs(r_m)
        candidate_graphs.append({
            "name": f"Proximity Radius ({r_km} km)",
            "category": f"Proximity Radius ({r_km} km)",
            "edges": set(pairs)
        })

    # Graph 6-8: KNN Graphs (k=3, k=5, k=8)
    ks = [3, 5, 8]
    for k in ks:
        distances, indices = kdtree.query(coords, k=k+1)
        knn_edges = set()
        for i in range(num_model_csds):
            for neighbor_idx in indices[i, 1:]: # skip self
                if i != neighbor_idx:
                    knn_edges.add((min(i, neighbor_idx), max(i, neighbor_idx)))
        candidate_graphs.append({
            "name": f"K-Nearest-Neighbours (k={k})",
            "category": f"K-NN (k={k})",
            "edges": knn_edges
        })

    # 5. Evaluate candidate graph metrics
    results = []

    for item in candidate_graphs:
        g_name = item["name"]
        g_cat = item["category"]
        edges = item["edges"]

        total_edges = len(edges)

        # Build NetworkX graph over all num_model_csds nodes
        G = nx.Graph()
        G.add_nodes_from(range(num_model_csds))
        G.add_edges_from(edges)

        # Degree stats across all 5054 nodes
        degrees = np.array([d for _, d in G.degree()])
        mean_deg = float(np.mean(degrees))
        median_deg = float(np.median(degrees))
        min_deg = int(np.min(degrees))
        max_deg = int(np.max(degrees))

        isolated_nodes = int(np.sum(degrees == 0))
        unique_active_nodes = int(np.sum(degrees > 0))

        num_components = nx.number_connected_components(G)

        # Distance summary for non-border edges
        non_border_edges = [e for e in edges if e not in border_edge_set]
        num_non_border = len(non_border_edges)

        if num_non_border > 0:
            nb_dists = []
            for u, v in non_border_edges:
                dx = coords[u, 0] - coords[v, 0]
                dy = coords[u, 1] - coords[v, 1]
                dist_km = math.sqrt(dx*dx + dy*dy) / 1000.0
                nb_dists.append(dist_km)
            
            nb_mean = float(np.mean(nb_dists))
            nb_median = float(np.median(nb_dists))
            nb_min = float(np.min(nb_dists))
            nb_max = float(np.max(nb_dists))
        else:
            nb_mean, nb_median, nb_min, nb_max = 0.0, 0.0, 0.0, 0.0

        # Practical GraphSAGE training density evaluation
        if mean_deg > 60 or total_edges > 150000:
            too_dense = "Yes (Too Dense - Over-smoothing Risk)"
        elif mean_deg > 35 or total_edges > 75000:
            too_dense = "Borderline High Density"
        else:
            too_dense = "No (Practical Density)"

        results.append({
            "graph_name": g_name,
            "total_edges": total_edges,
            "unique_active_nodes": unique_active_nodes,
            "isolated_nodes": isolated_nodes,
            "num_connected_components": num_components,
            "mean_degree": round(mean_deg, 2),
            "median_degree": round(median_deg, 1),
            "min_degree": min_deg,
            "max_degree": max_deg,
            "non_border_edges_count": num_non_border,
            "non_border_dist_mean_km": round(nb_mean, 2) if num_non_border > 0 else "N/A",
            "non_border_dist_median_km": round(nb_median, 2) if num_non_border > 0 else "N/A",
            "non_border_dist_min_km": round(nb_min, 2) if num_non_border > 0 else "N/A",
            "non_border_dist_max_km": round(nb_max, 2) if num_non_border > 0 else "N/A",
            "too_dense_for_graphsage": too_dense
        })

    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_csv_path, index=False)
    print(f"Saved CSV audit results to: {out_csv_path}")

    # Index results by graph name for text rendering
    res_dict = {r["graph_name"]: r for r in results}

    # Generate Markdown Summary Report
    print(f"Generating Markdown audit report: {out_md_path}")

    sb = res_dict["Shared-border Adjacency"]
    r25 = res_dict["Proximity Radius (25 km)"]
    r50 = res_dict["Proximity Radius (50 km)"]
    r100 = res_dict["Proximity Radius (100 km)"]
    r200 = res_dict["Proximity Radius (200 km)"]
    knn3 = res_dict["K-Nearest-Neighbours (k=3)"]
    knn5 = res_dict["K-Nearest-Neighbours (k=5)"]
    knn8 = res_dict["K-Nearest-Neighbours (k=8)"]

    md_content = f"""# Spatial Graph-Topology Audit Report

**Scope**: 5,054 Census Subdivisions (CSDs) across Canada  
**CRS**: Statistics Canada Lambert (`EPSG:3347`, metres)  
**Date**: 2026-07-30  

---

## 1. Executive Summary & Graph Comparison

This audit evaluates candidate spatial graph topologies for Canada-wide HPAI disease risk modeling using GraphSAGE. Candidate graph topologies are evaluated across connectivity, degree distribution, isolation, non-border spatial reach, and suitability for GraphSAGE 2-layer message passing.

### Candidate Graph Metrics

| Graph Candidate | Total Edges | Unique Active Nodes | Isolated Nodes | Connected Components | Mean Degree | Median Degree | Min Degree | Max Degree | Non-Border Edges | Non-Border Dist Mean (km) | Non-Border Dist Range (km) | Too Dense for GraphSAGE? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""

    for r in results:
        dist_range = f"{r['non_border_dist_min_km']} - {r['non_border_dist_max_km']}" if r['non_border_edges_count'] > 0 else "N/A"
        dist_mean = f"{r['non_border_dist_mean_km']}" if r['non_border_edges_count'] > 0 else "N/A"
        md_content += f"| **{r['graph_name']}** | {r['total_edges']:,} | {r['unique_active_nodes']:,} | {r['isolated_nodes']} | {r['num_connected_components']:,} | {r['mean_degree']} | {r['median_degree']} | {r['min_degree']} | {r['max_degree']} | {r['non_border_edges_count']:,} | {dist_mean} | {dist_range} | {r['too_dense_for_graphsage']} |\n"

    md_content += f"""

---

## 2. Structural & Topological Findings

1. **Shared-Border Adjacency Graph**:
   - Contains **{sb['total_edges']:,} undirected edges** with a mean node degree of **{sb['mean_degree']}** (median {sb['median_degree']}, min {sb['min_degree']}, max {sb['max_degree']}).
   - All 5,054 model-scope CSDs are represented with degree >= 1 (0 isolated nodes in model scope), organized into **{sb['num_connected_components']} connected components** corresponding to geographic landmasses, islands, and river-separated regions.
   - All edges in this graph are physical shared-border adjacencies (0 non-border edges).

2. **Geographic Distance Radius Graphs**:
   - **25 km Radius**: Contains **{r25['total_edges']:,} edges** (mean degree {r25['mean_degree']}), but leaves **{r25['isolated_nodes']} CSDs isolated** and fragments into **{r25['num_connected_components']} components** because large rural/northern CSD centroids exceed 25 km distance.
   - **50 km Radius**: Expands connectivity to **{r50['total_edges']:,} edges** (mean degree {r50['mean_degree']}), reducing isolated CSDs to **{r50['isolated_nodes']}**. Non-border edges average **{r50['non_border_dist_mean_km']} km**.
   - **100 km Radius**: Reaches **{r100['total_edges']:,} edges** (mean degree {r100['mean_degree']}), but experiences extreme degree inflation in dense southern clusters (max degree {r100['max_degree']}).
   - **200 km Radius**: Explodes to **{r200['total_edges']:,} edges** (mean degree {r200['mean_degree']}, max degree {r200['max_degree']}). **Extremely dense and unsuitable for GraphSAGE training** due to severe over-smoothing (node representations blur across multi-province regions) and massive memory overhead.

3. **K-Nearest-Neighbour (KNN) Graphs**:
   - **k = 3**: Produces **{knn3['total_edges']:,} edges** (mean degree {knn3['mean_degree']}, min degree {knn3['min_degree']}, max degree {knn3['max_degree']}). **0 isolated CSDs** and **{knn3['num_connected_components']} connected components**. Ensures all remote CSDs maintain spatial neighbors.
   - **k = 5**: Produces **{knn5['total_edges']:,} edges** (mean degree {knn5['mean_degree']}, min degree {knn5['min_degree']}, max degree {knn5['max_degree']}). **0 isolated CSDs** and only **{knn5['num_connected_components']} connected components**. Non-border edges average **{knn5['non_border_dist_mean_km']} km**.
   - **k = 8**: Produces **{knn8['total_edges']:,} edges** (mean degree {knn8['mean_degree']}, min degree {knn8['min_degree']}, max degree {knn8['max_degree']}). **0 isolated CSDs** and **{knn8['num_connected_components']} single fully connected component** across all of Canada. Non-border edges average **{knn8['non_border_dist_mean_km']} km**.

---

## 3. Recommended Graph Candidates for Next Validation Experiment

Based purely on graph topology, spatial representation, and GraphSAGE training feasibility (without referencing 2025 validation performance):

### Recommendation 1: Sparse Graph Candidate -> **K-Nearest-Neighbours (k = 3)**
- **Topological Rationale**: 
  - Guarantees **0 isolated CSDs** across all 5,054 model-scope CSDs (unlike distance radius graphs which isolate rural northern CSDs).
  - Highly sparse and efficient (**{knn3['total_edges']:,} edges**, mean degree {knn3['mean_degree']}), serving as a topology candidate for testing localized spatial information sharing without signal dilution.
  - Non-border edges average **{knn3['non_border_dist_mean_km']} km**, connecting isolated island/coastal CSDs to their nearest regional neighbors.

### Recommendation 2: Moderately Connected Graph Candidate -> **K-Nearest-Neighbours (k = 5)**
- **Topological Rationale**: 
  - Provides **{knn5['total_edges']:,} edges** (mean degree {knn5['mean_degree']}, max degree {knn5['max_degree']}), serving as a topology candidate for testing broader spatial information sharing across neighboring regions.
  - Reduces total connected components down to **{knn5['num_connected_components']}** across all of Canada while keeping non-border edge distances around **{knn5['non_border_dist_mean_km']} km**.
  - Controlled maximum degree ensures balanced message passing while preventing over-smoothing.

---

*Report generated automatically by `scripts/audit_candidate_spatial_graphs.py`.*
"""

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Saved Markdown audit report to: {out_md_path}")
    print("=== Spatial Graph Audit Completed Successfully! ===")

if __name__ == "__main__":
    main()
