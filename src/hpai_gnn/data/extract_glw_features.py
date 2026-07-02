import os
import sys
import time
import json
import pandas as pd
import numpy as np
import warnings

try:
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask
except ImportError:
    print("Error: 'geopandas' and 'rasterio' are required to run this script.")
    print("Please install them before running.")
    sys.exit(1)

# Suppress geometry warnings from GeoPandas/Shapely during masking operations
warnings.filterwarnings('ignore', category=UserWarning)

def process_single_raster(src, csd_gdf_wgs84, prefix):
    nodata = src.nodata
    means = []
    maxs = []
    p95s = []
    methods = []
    valid_counts = []
    
    for idx, row in csd_gdf_wgs84.iterrows():
        geom = [row.geometry]
        try:
            # Mask raster with CSD geometry
            out_image, out_transform = mask(src, geom, crop=True)
            data = out_image[0]
            
            # Exclude nodata values
            if nodata is not None:
                valid_mask = (data != nodata) & (data > -3.3e+38)
            else:
                valid_mask = ~np.isnan(data) & (data > -3.3e+38)
                
            valid_data = data[valid_mask]
            pixel_count = int(valid_mask.sum())
            
            if len(valid_data) > 0:
                mean_val = float(valid_data.mean())
                max_val = float(valid_data.max())
                p95_val = float(np.percentile(valid_data, 95))
                method = "zonal"
            else:
                # Fallback 1: Centroid sample
                centroid = row.geometry.centroid
                val = list(src.sample([(centroid.x, centroid.y)]))[0][0]
                pixel_count = 0
                if val != nodata and not np.isnan(val) and val > -3.3e+38:
                    mean_val = float(val)
                    max_val = float(val)
                    p95_val = float(val)
                    method = "centroid_fallback"
                else:
                    mean_val = 0.0
                    max_val = 0.0
                    p95_val = 0.0
                    method = "zero_fallback"
                    
        except ValueError:
            # Fallback 2: Centroid sample (geometry doesn't overlap or has bbox issue)
            pixel_count = 0
            try:
                centroid = row.geometry.centroid
                val = list(src.sample([(centroid.x, centroid.y)]))[0][0]
                if val != nodata and not np.isnan(val) and val > -3.3e+38:
                    mean_val = float(val)
                    max_val = float(val)
                    p95_val = float(val)
                    method = "centroid_fallback"
                else:
                    mean_val = 0.0
                    max_val = 0.0
                    p95_val = 0.0
                    method = "zero_fallback"
            except Exception:
                mean_val = 0.0
                max_val = 0.0
                p95_val = 0.0
                method = "zero_fallback"
        except Exception as e:
            # Catch other unexpected exceptions
            pixel_count = 0
            mean_val = 0.0
            max_val = 0.0
            p95_val = 0.0
            method = "zero_fallback"
            
        means.append(mean_val)
        maxs.append(max_val)
        p95s.append(p95_val)
        methods.append(method)
        valid_counts.append(pixel_count)
        
    return pd.DataFrame({
        'csd_uid': csd_gdf_wgs84['CSDUID'].str.strip(),
        f'{prefix}_mean': means,
        f'{prefix}_max': maxs,
        f'{prefix}_p95': p95s,
        f'{prefix}_extraction_method': methods,
        f'{prefix}_valid_pixel_count': valid_counts
    })

def main():
    start_time = time.time()
    
    # Project paths setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    raw_shp_path = os.path.join(base_dir, "data", "raw", "boundaries", "lcsd000b21a_e.shp")
    raw_glw_dir = os.path.join(base_dir, "data", "raw", "glw")
    output_dir = os.path.join(base_dir, "data", "processed", "glw")
    
    output_v1_path = os.path.join(output_dir, "csd_glw_poultry_density.csv")
    output_sens_path = os.path.join(output_dir, "csd_glw_chicken_2015_sensitivity.csv")
    output_summary_path = os.path.join(output_dir, "csd_glw_poultry_density_summary.json")
    
    rasters = {
        'glw2020_chicken': os.path.join(raw_glw_dir, 'GLW4-2020.D-DA.CHK.tif'),
        'glw2015_duck': os.path.join(raw_glw_dir, '5_Dk_2015_Da.tif'),
        'glw2015_chicken': os.path.join(raw_glw_dir, '5_Ch_2015_Da.tif')
    }
    
    print(f"Base Directory: {base_dir}")
    print(f"Loading raw CSD boundaries from: {raw_shp_path}")
    if not os.path.exists(raw_shp_path):
        print(f"Error: Boundary shapefile not found at {raw_shp_path}")
        sys.exit(1)
        
    csd_gdf = gpd.read_file(raw_shp_path)
    total_csds = len(csd_gdf)
    print(f"Loaded {total_csds} CSD boundaries.")
    
    # Ensure CSD boundaries are projected to EPSG:4326 to match raster CRS
    print("Reprojecting CSD boundaries to EPSG:4326...")
    csd_gdf_wgs84 = csd_gdf.to_crs("EPSG:4326")
    
    compiled_dfs = {}
    
    for key, raster_path in rasters.items():
        print(f"\nProcessing raster: {raster_path}...")
        if not os.path.exists(raster_path):
            print(f"Error: Raster file not found at {raster_path}")
            sys.exit(1)
            
        with rasterio.open(raster_path) as src:
            df = process_single_raster(src, csd_gdf_wgs84, key)
            compiled_dfs[key] = df
            
    # Combine dfs into V1 (2020 Chicken, 2015 Duck) and Sensitivity (2015 Chicken)
    v1_df = compiled_dfs['glw2020_chicken'].merge(compiled_dfs['glw2015_duck'], on='csd_uid', how='outer')
    sens_df = compiled_dfs['glw2015_chicken']
    
    # Rename V1 columns to density-explicit ones
    v1_df = v1_df.rename(columns={
        'glw2020_chicken_mean': 'glw2020_chicken_density_mean',
        'glw2020_chicken_max': 'glw2020_chicken_density_max',
        'glw2020_chicken_p95': 'glw2020_chicken_density_p95',
        'glw2015_duck_mean': 'glw2015_duck_density_mean',
        'glw2015_duck_max': 'glw2015_duck_density_max',
        'glw2015_duck_p95': 'glw2015_duck_density_p95'
    })
    
    # Check QA counts for the summary JSON
    method_counts = {}
    for key in rasters.keys():
        col_name = f"{key}_extraction_method"
        method_counts[key] = compiled_dfs[key][col_name].value_counts().to_dict()
        
    # Range stats per numeric feature
    feature_stats = {}
    
    def get_stats(df, col):
        return {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean())
        }
        
    for key in ['glw2020_chicken_density', 'glw2015_duck_density']:
        for stat in ['mean', 'max', 'p95']:
            col = f"{key}_{stat}"
            feature_stats[col] = get_stats(v1_df, col)
            
    for stat in ['mean', 'max', 'p95']:
        col = f"glw2015_chicken_{stat}"
        feature_stats[col] = get_stats(sens_df, col)
        
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving V1 features to: {output_v1_path}")
    v1_df.to_csv(output_v1_path, index=False)
    
    print(f"Saving 2015 chicken sensitivity features to: {output_sens_path}")
    sens_df.to_csv(output_sens_path, index=False)
    
    # Build summary dict
    elapsed = time.time() - start_time
    summary_data = {
        'row_count': total_csds,
        'v1_features': [c for c in v1_df.columns if c != 'csd_uid'],
        'sensitivity_features': [c for c in sens_df.columns if c != 'csd_uid'],
        'method_counts': method_counts,
        'missing_values': {
            'v1_table': v1_df.isna().sum().to_dict(),
            'sensitivity_table': sens_df.isna().sum().to_dict()
        },
        'feature_stats': feature_stats,
        'runtime_seconds': round(elapsed, 2)
    }
    
    print(f"Saving JSON summary to: {output_summary_path}")
    with open(output_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
        
    # QA Range Checks in stdout
    print("\n--- QA Verification ---")
    print(f"V1 Output row count: {len(v1_df)} | Columns: {list(v1_df.columns)}")
    print(f"Sensitivity Output row count: {len(sens_df)} | Columns: {list(sens_df.columns)}")
    
    for key in ['glw2020_chicken', 'glw2015_duck']:
        m_col = f"{key}_extraction_method"
        print(f"Method counts for {key}: {v1_df[m_col].value_counts().to_dict()}")
        
    print(f"2015 chicken removed from V1 features table: {'glw2015_chicken_mean' not in v1_df.columns}")
    print(f"Pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
