import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import shapely
    from shapely.geometry import Point
except ImportError:
    print("Error: 'geopandas' and 'shapely' are required to run this script.")
    print("Please install them before running.")
    sys.exit(1)

# Suppress geometry warnings during intersection checks
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    start_time = time.time()
    
    # Project paths setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    raw_shp_path = os.path.join(base_dir, "data", "raw", "boundaries", "lcsd000b21a_e.shp")
    output_dir = os.path.join(base_dir, "data", "processed", "graph")
    
    output_csv_path = os.path.join(output_dir, "csd_adjacency_edges.csv")
    output_summary_path = os.path.join(output_dir, "csd_adjacency_summary.json")
    
    print(f"Base Directory: {base_dir}")
    print(f"Loading raw CSD boundaries from: {raw_shp_path}")
    if not os.path.exists(raw_shp_path):
        print(f"Error: Boundary shapefile not found at {raw_shp_path}")
        sys.exit(1)
        
    csd_gdf = gpd.read_file(raw_shp_path)
    total_csds = len(csd_gdf)
    print(f"Loaded {total_csds} CSD boundaries.")
    
    # Ensure CSD boundaries are projected to EPSG:3347 and cleaned
    print("Ensuring CSD boundaries are in EPSG:3347...")
    csd_gdf = csd_gdf.to_crs("EPSG:3347")
    csd_gdf['CSDUID'] = csd_gdf['CSDUID'].str.strip()
    
    # Compute centroids for distance metrics
    csd_gdf['centroid'] = csd_gdf.geometry.centroid
    
    # Map CSDs to their indexes for faster retrieval
    csduid_list = csd_gdf['CSDUID'].tolist()
    csd_centroids = csd_gdf['centroid'].tolist()
    csd_geoms = csd_gdf.geometry.tolist()
    
    # Build edges using spatial index
    print("Building adjacency list using spatial index...")
    sindex = csd_gdf.sindex
    edges = []
    
    for idx, row in csd_gdf.iterrows():
        csd_uid = row['CSDUID']
        geom = row['geometry']
        boundary = geom.boundary
        centroid = row['centroid']
        
        # Query spatial index for candidate intersecting polygons
        candidates_idx = sindex.query(geom)
        for candidate_idx in candidates_idx:
            candidate_uid = csduid_list[candidate_idx]
            
            # Undirected constraint: source_uid < target_uid to prevent duplicates/self-loops
            if csd_uid >= candidate_uid:
                continue
                
            candidate_geom = csd_geoms[candidate_idx]
            
            # Check if geometries intersect/touch
            if geom.intersects(candidate_geom):
                candidate_centroid = csd_centroids[candidate_idx]
                centroid_dist = float(centroid.distance(candidate_centroid) / 1000.0)
                
                # Check shared boundary length
                try:
                    intersection_geom = boundary.intersection(candidate_geom.boundary)
                    shared_len = float(intersection_geom.length)
                except Exception:
                    try:
                        intersection_geom = boundary.make_valid().intersection(candidate_geom.boundary.make_valid())
                        shared_len = float(intersection_geom.length)
                    except Exception:
                        shared_len = 0.0  # Default fallback
                
                # If shared length > 0, it's Rook. Else it's Queen (single point touching).
                edge_type = 'rook' if shared_len > 0.0 else 'queen'
                
                edges.append({
                    'source_csd_uid': csd_uid,
                    'target_csd_uid': candidate_uid,
                    'edge_type': edge_type,
                    'shared_boundary_length_m': shared_len,
                    'centroid_distance_km': centroid_dist,
                    'source_province': csd_uid[:2],
                    'target_province': candidate_uid[:2],
                    'cross_province_edge': csd_uid[:2] != candidate_uid[:2],
                    'adjacency_method': 'spatial_contiguity'
                })
                
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_csds:
            print(f"  Processed contiguity for {idx + 1}/{total_csds} CSDs...")
            
    print(f"Extracted {len(edges)} spatial contiguity edges.")
    
    # Identify isolated CSDs (islands or remote locations with 0 contiguity edges)
    connected_uids = set()
    for e in edges:
        connected_uids.add(e['source_csd_uid'])
        connected_uids.add(e['target_csd_uid'])
        
    all_uids = set(csduid_list)
    isolated_uids = sorted(list(all_uids - connected_uids))
    print(f"Found {len(isolated_uids)} isolated CSDs (islands/remote nodes).")
    
    # Compute nearest-neighbor centroid fallbacks for isolated CSDs
    fallback_edges_count = 0
    if isolated_uids:
        print("Computing nearest centroid fallbacks for isolated CSDs...")
        # Create an STRtree of all centroids for fast nearest neighbor search
        centroid_tree = shapely.STRtree(csd_centroids)
        
        for iso_uid in isolated_uids:
            iso_idx = csduid_list.index(iso_uid)
            iso_centroid = csd_centroids[iso_idx]
            
            # Query STRtree. Nearest will be itself, so we get the top 2 closest
            nearest_indices = centroid_tree.nearest(iso_centroid)
            # Find closest centroid that is not itself
            best_idx = None
            min_dist = float('inf')
            
            for other_idx, other_centroid in enumerate(csd_centroids):
                if other_idx == iso_idx:
                    continue
                dist = float(iso_centroid.distance(other_centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = other_idx
                    
            if best_idx is not None:
                target_uid = csduid_list[best_idx]
                
                # Undirected normalization
                u, v = sorted([iso_uid, target_uid])
                
                # Check if this edge already exists in edges to avoid duplication
                edge_exists = any(e['source_csd_uid'] == u and e['target_csd_uid'] == v for e in edges)
                if not edge_exists:
                    centroid_dist = min_dist / 1000.0
                    edges.append({
                        'source_csd_uid': u,
                        'target_csd_uid': v,
                        'edge_type': 'nearest_neighbor_fallback',
                        'shared_boundary_length_m': 0.0,
                        'centroid_distance_km': centroid_dist,
                        'source_province': u[:2],
                        'target_province': v[:2],
                        'cross_province_edge': u[:2] != v[:2],
                        'adjacency_method': 'nearest_centroid'
                    })
                    fallback_edges_count += 1
                    
        print(f"Added {fallback_edges_count} nearest-neighbor fallback edges.")
        
    # Re-verify connectivity
    final_connected = set()
    for e in edges:
        final_connected.add(e['source_csd_uid'])
        final_connected.add(e['target_csd_uid'])
    final_isolated_count = len(all_uids - final_connected)
    
    # Build DataFrame
    edges_df = pd.DataFrame(edges)
    
    # QA Checks
    print("\n--- Adjacency Graph QA Checks ---")
    print(f"Total Edges extracted: {len(edges_df)}")
    
    # Self-loops check
    self_loops = (edges_df['source_csd_uid'] == edges_df['target_csd_uid']).sum()
    print(f"Self-loops found: {self_loops} (Expected: 0)")
    
    # Duplicate edge check
    duplicates = edges_df.duplicated(subset=['source_csd_uid', 'target_csd_uid']).sum()
    print(f"Duplicate edges found: {duplicates} (Expected: 0)")
    
    # Unidirectional check (source < target)
    unordered = (edges_df['source_csd_uid'] >= edges_df['target_csd_uid']).sum()
    print(f"Edges violation source >= target: {unordered} (Expected: 0)")
    
    # Isolated nodes check
    print(f"Isolated CSDs after fallback: {final_isolated_count} (Expected: 0)")
    
    # All nodes present check
    missing_nodes_count = len(all_uids - final_connected)
    print(f"CSDs missing from graph: {missing_nodes_count} (Expected: 0)")
    
    # Cross-province edges
    cross_prov_count = int(edges_df['cross_province_edge'].sum())
    print(f"Cross-province edges: {cross_prov_count}")
    
    # Degree distribution
    # Reconstruct degrees for undirected graph
    all_incident_nodes = list(edges_df['source_csd_uid']) + list(edges_df['target_csd_uid'])
    degrees = pd.Series(all_incident_nodes).value_counts()
    
    # Fill in degree 0 if any (should be 0 missing)
    for u in csduid_list:
        if u not in degrees:
            degrees[u] = 0
            
    print("\nDegree Distribution Summary:")
    print(f"  Min degree: {degrees.min()}")
    print(f"  Max degree: {degrees.max()}")
    print(f"  Mean degree: {degrees.mean():.2f}")
    print(f"  Median degree: {degrees.median()}")
    
    # Save edges to CSV
    os.makedirs(output_dir, exist_ok=True)
    cols_order = [
        'source_csd_uid', 'target_csd_uid', 'edge_type',
        'shared_boundary_length_m', 'centroid_distance_km',
        'source_province', 'target_province', 'cross_province_edge',
        'adjacency_method'
    ]
    edges_df = edges_df[cols_order]
    edges_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved adjacency edges to {output_csv_path}")
    
    # Save summary diagnostics JSON
    elapsed = time.time() - start_time
    summary_data = {
        'total_edges': len(edges_df),
        'rook_edges_count': int((edges_df['edge_type'] == 'rook').sum()),
        'queen_edges_count': int((edges_df['edge_type'] == 'queen').sum()),
        'fallback_edges_count': int((edges_df['edge_type'] == 'nearest_neighbor_fallback').sum()),
        'isolated_csds_before_fallback': len(isolated_uids),
        'isolated_csds_after_fallback': final_isolated_count,
        'cross_province_edges_count': cross_prov_count,
        'self_loops_count': int(self_loops),
        'duplicate_edges_count': int(duplicates),
        'min_degree': int(degrees.min()),
        'max_degree': int(degrees.max()),
        'mean_degree': float(degrees.mean()),
        'median_degree': int(degrees.median()),
        'runtime_seconds': round(elapsed, 2)
    }
    
    with open(output_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary JSON to {output_summary_path}")
    print(f"Total script runtime: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
