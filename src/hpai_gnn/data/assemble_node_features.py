import os
import sys
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
    # Paths setup
    raw_shp_path = os.path.join(base_dir, "data", "raw", "boundaries", "lcsd000b21a_e.shp")
    poultry_proxy_path = os.path.join(base_dir, "data", "processed", "poultry", "ccs_poultry_proxy.csv")
    glw_path = os.path.join(base_dir, "data", "processed", "glw", "csd_glw_poultry_density.csv")
    roads_path = os.path.join(base_dir, "data", "processed", "roads", "csd_road_features.csv")
    graph_path = os.path.join(base_dir, "data", "processed", "graph", "csd_adjacency_edges.csv")
    output_dir = os.path.join(base_dir, "data", "processed", "features")
    output_csv_path = os.path.join(output_dir, "csd_static_structural_features.csv")
    
    print("Loading datasets...")
    
    # Load CSD list from shapefile attributes (discard geometry to keep it fast)
    import geopandas as gpd
    csd_gdf = gpd.read_file(raw_shp_path)
    csd_df = pd.DataFrame({
        'csd_uid': csd_gdf['CSDUID'].str.strip(),
        'csd_name': csd_gdf['CSDNAME'].str.strip()
    })
    
    print(f"Master list has {len(csd_df)} CSDs.")
    
    # Load other static features
    poultry_proxy = pd.read_csv(poultry_proxy_path, dtype={'csd_uid': str, 'ccs_uid': str, 'province_uid': str})
    glw = pd.read_csv(glw_path, dtype={'csd_uid': str})
    roads = pd.read_csv(roads_path, dtype={'csd_uid': str})
    
    # Select specific columns from poultry_proxy
    poultry_cols = [
        'csd_uid', 'ccs_uid', 'ccs_name', 'province_uid', 'province_name',
        'ccs_broilers_roasters_cornish_production_kg', 'ccs_broilers_roasters_cornish_farms_reporting',
        'ccs_turkey_production_kg', 'ccs_turkey_farms_reporting'
    ]
    poultry_proxy_sel = poultry_proxy[poultry_cols]
    
    # Merge datasets sequentially
    print("Merging CSD features...")
    final_df = csd_df.merge(poultry_proxy_sel, on='csd_uid', how='left')
    final_df = final_df.merge(glw, on='csd_uid', how='left')
    final_df = final_df.merge(roads, on='csd_uid', how='left')
    
    # QA Checks
    print("\n--- Feature Assembly QA Validation ---")
    row_count = len(final_df)
    print(f"Output row count: {row_count} (Expected: 5161)")
    if row_count != 5161:
        print("Error: Row count mismatch.")
        sys.exit(1)
        
    duplicates = final_df['csd_uid'].duplicated().sum()
    print(f"Duplicate csd_uids: {duplicates} (Expected: 0)")
    if duplicates > 0:
        print("Error: Duplicates found.")
        sys.exit(1)
        
    # Assert exclusions policy
    excluded_keywords = ['geometry', 'geom', 'latitude', 'longitude', 'lat', 'lon', 'outbreak', 'label']
    found_violations = [c for c in final_df.columns if any(kw in c.lower() for kw in excluded_keywords)]
    print(f"Violations of exclusion policy: {found_violations} (Expected: [])")
    if found_violations:
        print("Error: Exclusion policy violated.")
        sys.exit(1)
        
    # Missingness check
    print("\nMissingness Summary:")
    print(final_df.isna().sum())
    
    # Graph cross-validation
    print("\nCross-validating with graph adjacency...")
    if os.path.exists(graph_path):
        edges = pd.read_csv(graph_path, dtype={'source_csd_uid': str, 'target_csd_uid': str})
        graph_nodes = set(edges['source_csd_uid']).union(set(edges['target_csd_uid']))
        feature_nodes = set(final_df['csd_uid'])
        
        # 1. Confirm all graph nodes have feature rows
        missing_in_features = graph_nodes - feature_nodes
        print(f"Graph nodes missing from feature table: {len(missing_in_features)} (Expected: 0)")
        if missing_in_features:
            print("Error: Graph nodes are missing from the static feature table.")
            sys.exit(1)
            
        # 2. Confirm all feature rows exist in graph node set
        missing_in_graph = feature_nodes - graph_nodes
        print(f"Feature CSDs missing from graph: {len(missing_in_graph)} (Expected: 0)")
        if missing_in_graph:
            print("Error: Static features contain nodes not present in the graph.")
            sys.exit(1)
            
        print("Graph cross-validation passed successfully!")
    else:
        print("Warning: Adjacency graph file not found. Skipping graph cross-validation.")
        
    # Province coverage check
    print("\nCSDs count by Province:")
    print(final_df['province_name'].apply(lambda x: str(x).encode('ascii', errors='ignore').decode('ascii')).value_counts())
    
    # Save features to CSV
    os.makedirs(output_dir, exist_ok=True)
    
    # Define logical column order
    cols_order = [
        'csd_uid', 'csd_name', 'province_uid', 'province_name', 'ccs_uid', 'ccs_name',
        'ccs_broilers_roasters_cornish_production_kg', 'ccs_broilers_roasters_cornish_farms_reporting',
        'ccs_turkey_production_kg', 'ccs_turkey_farms_reporting',
        'glw2020_chicken_density_mean', 'glw2020_chicken_density_max', 'glw2020_chicken_density_p95',
        'glw2015_duck_density_mean', 'glw2015_duck_density_max', 'glw2015_duck_density_p95',
        'csd_area_km2',
        'all_road_length_km', 'all_road_density_km_per_km2',
        'major_road_length_km', 'major_road_density_km_per_km2',
        'highway_length_km', 'highway_density_km_per_km2',
        'distance_to_nearest_major_road_km', 'distance_to_nearest_highway_km',
        'major_road_proportion', 'highway_proportion',
        'glw2020_chicken_extraction_method', 'glw2020_chicken_valid_pixel_count',
        'glw2015_duck_extraction_method', 'glw2015_duck_valid_pixel_count',
        'road_data_coverage', 'small_area_flag'
    ]
    
    # Verify all output columns exist
    missing_cols = [c for c in cols_order if c not in final_df.columns]
    if missing_cols:
        print(f"Error: Missing columns in final DataFrame: {missing_cols}")
        sys.exit(1)
        
    final_df = final_df[cols_order]
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully assembled CSD static node features at {output_csv_path}")

if __name__ == '__main__':
    main()
