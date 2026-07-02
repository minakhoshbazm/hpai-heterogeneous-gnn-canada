import os
import sys
import time
import json
import zipfile
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

# Suppress geometry warnings from GeoPandas/Shapely during masking/overlay operations
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    start_time = time.time()
    
    # Project paths setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    raw_shp_path = os.path.join(base_dir, "data", "raw", "boundaries", "lcsd000b21a_e.shp")
    raw_roads_dir = os.path.join(base_dir, "data", "raw", "roads")
    output_dir = os.path.join(base_dir, "data", "processed", "roads")
    
    output_csv_path = os.path.join(output_dir, "csd_road_features.csv")
    output_summary_path = os.path.join(output_dir, "csd_road_features_summary.json")
    
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
    
    # Define CSD areas in km2
    csd_gdf['csd_area_km2'] = csd_gdf.geometry.area / 1e6
    
    # Define taxonomies based on the unique ROADCLASS values inspected
    HIGHWAY_CLASSES = {'Freeway', 'Expressway / Highway', 'Ramp'}
    MAJOR_CLASSES = {'Freeway', 'Expressway / Highway', 'Ramp', 'Arterial', 'Collector'}
    LOCAL_CLASSES = {'Local / Street', 'Local / Strata', 'Service Lane', 'Alleyway / Lane', 'Local / Unknown'}
    ALL_CLASSES = HIGHWAY_CLASSES | MAJOR_CLASSES | LOCAL_CLASSES | {
        'Resource / Recreation', 'Winter', 'Rapid Transit', 'Unknown'
    }
    
    # Target provinces and their corresponding ZIP files
    PROVINCE_ZIPS = {
        '10': 'nrn_rrn_nl_shp_en.zip',
        '11': 'nrn_rrn_pe_shp_en.zip',
        '12': 'nrn_rrn_ns_shp_en.zip',
        '13': 'nrn_rrn_nb_shp_en.zip',
        '24': 'nrn_rrn_qc_shp_en.zip',
        '35': 'nrn_rrn_on_shp_en.zip',
        '46': 'nrn_rrn_mb_shp_en.zip',
        '47': 'nrn_rrn_sk_shp_en.zip',
        '48': 'nrn_rrn_ab_shp_en.zip',
        '59': 'nrn_rrn_bc_shp_en.zip',
    }
    
    prov_results = []
    prov_sanity_summaries = {}
    
    for prov_code, zip_filename in PROVINCE_ZIPS.items():
        zip_path = os.path.join(raw_roads_dir, zip_filename)
        print(f"\nProcessing Province Code {prov_code} from {zip_filename}...")
        
        if not os.path.exists(zip_path):
            print(f"Warning: ZIP file not found at {zip_path}. Skipping.")
            continue
            
        # Get CSD subset for this province
        csd_prov = csd_gdf[csd_gdf['CSDUID'].str.startswith(prov_code)].copy()
        if csd_prov.empty:
            print(f"No CSDs found starting with code {prov_code}. Skipping.")
            continue
            
        # Locate the ROADSEG shapefile inside the ZIP
        with zipfile.ZipFile(zip_path, 'r') as z:
            matches = [f for f in z.namelist() if f.endswith('ROADSEG.shp')]
            if not matches:
                print(f"Error: ROADSEG.shp not found in {zip_filename}")
                sys.exit(1)
            roadseg_file = matches[0]
            
        # Read the road segments
        print(f"Loading road segments from {roadseg_file}...")
        roads = gpd.read_file(f"zip://{zip_path}!{roadseg_file}")
        
        # Check required columns
        for col in ['ROADCLASS', 'geometry']:
            if col not in roads.columns:
                print(f"Error: Required column '{col}' missing from roads dataset.")
                sys.exit(1)
                
        # Reproject to EPSG:3347
        roads = roads.to_crs("EPSG:3347")
        
        # Calculate original total road length in NRN (for sanity checks)
        total_nrn_length_km = float((roads.geometry.length / 1000.0).sum())
        
        # Filter roads to those in ALL_CLASSES
        roads_filtered = roads[roads['ROADCLASS'].isin(ALL_CLASSES)].copy()
        filtered_nrn_length_km = float((roads_filtered.geometry.length / 1000.0).sum())
        
        # Intersect roads with CSD polygons using optimized CSD-by-CSD loop
        print(f"Intersecting {len(roads_filtered)} roads with {len(csd_prov)} CSDs (optimized loop)...")
        road_sindex = roads_filtered.sindex
        
        all_road_len = {}
        highway_road_len = {}
        major_road_len = {}
        overlay_total_length_km = 0.0
        
        for i, (idx, csd_row) in enumerate(csd_prov.iterrows()):
            csd_uid = csd_row['CSDUID']
            csd_geom = csd_row['geometry']
            
            # Query index to get candidate road indices
            idxs = road_sindex.query(csd_geom)
            if len(idxs) == 0:
                all_road_len[csd_uid] = 0.0
                highway_road_len[csd_uid] = 0.0
                major_road_len[csd_uid] = 0.0
                continue
                
            possible_roads = roads_filtered.iloc[idxs]
            
            # Perform intersection with robust self-healing try-except logic
            try:
                clipped_geoms = possible_roads.geometry.intersection(csd_geom)
                matching_roads = possible_roads
            except Exception as e:
                print(f"\n  [Warning] Intersection failed for CSD {csd_uid}: {e}. Retrying with make_valid()...")
                try:
                    valid_csd_geom = csd_geom.make_valid()
                    clipped_geoms = possible_roads.geometry.intersection(valid_csd_geom)
                    matching_roads = possible_roads
                except Exception as e2:
                    print(f"  [Warning] Retry 1 failed for CSD {csd_uid}: {e2}. Retrying with simplification...")
                    try:
                        simp_csd_geom = csd_geom.simplify(0.5, preserve_topology=True).make_valid()
                        clipped_geoms = possible_roads.geometry.intersection(simp_csd_geom)
                        matching_roads = possible_roads
                    except Exception as e3:
                        print(f"  [Error] Retry 2 failed for CSD {csd_uid}: {e3}. Falling back to unclipped overlay...")
                        try:
                            # Fallback: Find intersecting roads and use unclipped lengths within the CSD
                            intersects_mask = possible_roads.intersects(csd_geom)
                            matching_roads = possible_roads[intersects_mask]
                            clipped_geoms = matching_roads.geometry
                        except Exception as e4:
                            print(f"  [Critical] Fallback failed for CSD {csd_uid}: {e4}. Setting lengths to 0.0.")
                            all_road_len[csd_uid] = 0.0
                            highway_road_len[csd_uid] = 0.0
                            major_road_len[csd_uid] = 0.0
                            continue
            
            # Exclude empty geometries to ignore non-intersecting roads
            valid_mask = ~clipped_geoms.is_empty
            if not valid_mask.any():
                all_road_len[csd_uid] = 0.0
                highway_road_len[csd_uid] = 0.0
                major_road_len[csd_uid] = 0.0
                continue
                
            clipped_geoms = clipped_geoms[valid_mask]
            matching_roads = matching_roads[valid_mask]
            
            # Compute lengths in km
            lengths_km = clipped_geoms.length / 1000.0
            
            # Aggregate lengths
            total_len = float(lengths_km.sum())
            
            h_mask = matching_roads['ROADCLASS'].isin(HIGHWAY_CLASSES)
            h_len = float(lengths_km[h_mask.values].sum())
            
            m_mask = matching_roads['ROADCLASS'].isin(MAJOR_CLASSES)
            m_len = float(lengths_km[m_mask.values].sum())
            
            all_road_len[csd_uid] = total_len
            highway_road_len[csd_uid] = h_len
            major_road_len[csd_uid] = m_len
            overlay_total_length_km += total_len
            
            if (i + 1) % 200 == 0 or (i + 1) == len(csd_prov):
                print(f"  Processed {i + 1}/{len(csd_prov)} CSD overlays...")
        
        prov_sanity_summaries[prov_code] = {
            'nrn_raw_length_km': total_nrn_length_km,
            'nrn_filtered_length_km': filtered_nrn_length_km,
            'overlay_length_km': overlay_total_length_km,
        }
        print(f"Road length check: Raw NRN={total_nrn_length_km:.1f} km, Filtered NRN={filtered_nrn_length_km:.1f} km, Intersected CSD={overlay_total_length_km:.1f} km")
        
        # Compute distances from CSD centroids to nearest roads
        print("Computing nearest road distances from CSD centroids...")
        centroids = csd_prov.geometry.centroid.values
        
        major_roads_only = roads_filtered[roads_filtered['ROADCLASS'].isin(MAJOR_CLASSES)]
        highway_roads_only = roads_filtered[roads_filtered['ROADCLASS'].isin(HIGHWAY_CLASSES)]
        
        dist_to_major = compute_centroid_distances(centroids, major_roads_only)
        dist_to_highway = compute_centroid_distances(centroids, highway_roads_only)
        
        # Build provincial dataframe
        prov_df = pd.DataFrame({
            'csd_uid': csd_prov['CSDUID'],
            'all_road_length_km': csd_prov['CSDUID'].map(all_road_len).fillna(0.0),
            'major_road_length_km': csd_prov['CSDUID'].map(major_road_len).fillna(0.0),
            'highway_length_km': csd_prov['CSDUID'].map(highway_road_len).fillna(0.0),
            'distance_to_nearest_major_road_km': dist_to_major,
            'distance_to_nearest_highway_km': dist_to_highway
        })
        
        prov_results.append(prov_df)
        
    # Combine all provincial dataframes
    if prov_results:
        combined_prov_df = pd.concat(prov_results, ignore_index=True)
    else:
        combined_prov_df = pd.DataFrame(columns=[
            'csd_uid', 'all_road_length_km', 'major_road_length_km', 'highway_length_km',
            'distance_to_nearest_major_road_km', 'distance_to_nearest_highway_km'
        ])
        
    # Left join to master CSD list to ensure all 5,161 CSDs are present
    print("\nMerging with master CSD boundaries list (5,161 rows)...")
    master_df = pd.DataFrame({
        'csd_uid': csd_gdf['CSDUID'],
        'csd_area_km2': csd_gdf['csd_area_km2']
    })
    
    final_df = master_df.merge(combined_prov_df, on='csd_uid', how='left')
    
    # Fill NaN values for road lengths with 0.0
    for col in ['all_road_length_km', 'major_road_length_km', 'highway_length_km']:
        final_df[col] = final_df[col].fillna(0.0)
        
    # Compute densities
    final_df['all_road_density_km_per_km2'] = final_df['all_road_length_km'] / final_df['csd_area_km2']
    final_df['major_road_density_km_per_km2'] = final_df['major_road_length_km'] / final_df['csd_area_km2']
    final_df['highway_density_km_per_km2'] = final_df['highway_length_km'] / final_df['csd_area_km2']
    
    # Compute proportions
    final_df['major_road_proportion'] = (final_df['major_road_length_km'] / final_df['all_road_length_km']).fillna(0.0)
    final_df['highway_proportion'] = (final_df['highway_length_km'] / final_df['all_road_length_km']).fillna(0.0)
    
    # Handle division by zero or NaN areas if any
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    for col in ['all_road_density_km_per_km2', 'major_road_density_km_per_km2', 'highway_density_km_per_km2']:
        final_df[col] = final_df[col].fillna(0.0)
        
    # Add QA Safeguard Columns
    final_df['road_data_coverage'] = np.where(
        final_df['csd_uid'].astype(str).str.startswith(('60', '61', '62')), 
        'not_processed_territory', 
        'processed'
    )
    final_df['small_area_flag'] = final_df['csd_area_km2'] < 1.0
        
    # QA Checks
    print("\n--- QA Verification ---")
    row_count = len(final_df)
    print(f"Output row count: {row_count} (Expected: 5161)")
    
    duplicate_count = final_df['csd_uid'].duplicated().sum()
    print(f"Duplicate CSD UIDs: {duplicate_count} (Expected: 0)")
    
    negative_lengths = (final_df[['all_road_length_km', 'major_road_length_km', 'highway_length_km']] < 0).any().any()
    negative_densities = (final_df[['all_road_density_km_per_km2', 'major_road_density_km_per_km2', 'highway_density_km_per_km2']] < 0).any().any()
    negative_distances = (final_df[['distance_to_nearest_major_road_km', 'distance_to_nearest_highway_km']].dropna() < 0).any().any()
    
    print(f"Has negative lengths: {negative_lengths} (Expected: False)")
    print(f"Has negative densities: {negative_densities} (Expected: False)")
    print(f"Has negative distances: {negative_distances} (Expected: False)")
    
    zero_road_count = (final_df['all_road_length_km'] == 0.0).sum()
    print(f"Number of CSDs with zero road length: {zero_road_count}")
    
    # Print top/bottom density CSDs
    print("\nTop 5 CSDs by All Road Density:")
    top_5 = final_df.sort_values(by='all_road_density_km_per_km2', ascending=False).head(5)
    for idx, row in top_5.iterrows():
        print(f"  CSD {row['csd_uid']}: Density = {row['all_road_density_km_per_km2']:.3f} km/km², Area = {row['csd_area_km2']:.2f} km²")
        
    print("\nSave features to CSV...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Order columns logically
    cols_order = [
        'csd_uid', 'csd_area_km2',
        'all_road_length_km', 'all_road_density_km_per_km2',
        'major_road_length_km', 'major_road_density_km_per_km2',
        'highway_length_km', 'highway_density_km_per_km2',
        'distance_to_nearest_major_road_km', 'distance_to_nearest_highway_km',
        'major_road_proportion', 'highway_proportion',
        'road_data_coverage', 'small_area_flag'
    ]
    final_df = final_df[cols_order]
    
    final_df.to_csv(output_csv_path, index=False)
    print(f"Saved road features to {output_csv_path}")
    
    # Save JSON summary
    elapsed = time.time() - start_time
    summary_data = {
        'row_count': row_count,
        'duplicate_csd_uids': int(duplicate_count),
        'zero_road_csd_count': int(zero_road_count),
        'negative_values_found': bool(negative_lengths or negative_densities or negative_distances),
        'prov_sanity_summaries': prov_sanity_summaries,
        'runtime_seconds': round(elapsed, 2)
    }
    
    with open(output_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary JSON to {output_summary_path}")
    print(f"Total script runtime: {elapsed:.2f} seconds.")

def compute_centroid_distances(centroids, road_gdf):
    if road_gdf.empty:
        return [np.nan] * len(centroids)
    
    road_geoms = road_gdf.geometry.values
    tree = shapely.STRtree(road_geoms)
    
    valid_pts = []
    valid_indices = []
    for idx, p in enumerate(centroids):
        if p is not None and not p.is_empty:
            valid_pts.append(p)
            valid_indices.append(idx)
            
    if not valid_pts:
        return [np.nan] * len(centroids)
        
    idxs = tree.nearest(valid_pts)
    if isinstance(idxs, np.ndarray):
        idxs = idxs.tolist()
    else:
        idxs = [idxs]
        
    distances = [np.nan] * len(centroids)
    for i, pt_idx in enumerate(valid_indices):
        nearest_geom = road_geoms[idxs[i]]
        distances[pt_idx] = float(valid_pts[i].distance(nearest_geom) / 1000.0)
        
    return distances

if __name__ == "__main__":
    main()
