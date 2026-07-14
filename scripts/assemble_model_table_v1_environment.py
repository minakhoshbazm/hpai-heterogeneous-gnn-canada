import os
import sys
import pandas as pd
import numpy as np

def assemble_environment_table(labels_path, static_path, climate_path, dw_path, output_path):
    print("Starting V1 Environment-Only Modeling Table Assembly...")
    print("=========================================================")
    
    # Check if files exist
    for path in [labels_path, static_path, climate_path, dw_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input file not found: {path}")
            
    # 1-2. Load labels
    print(f"Loading CFIA labels from: {labels_path}")
    labels = pd.read_csv(labels_path, dtype={'csd_uid': str})
    
    # Extract year and month from labels
    labels['year'] = labels['month'].str.split('-').str[0].astype(int)
    labels['month'] = labels['month'].str.split('-').str[1].astype(int)
    
    # Keep 2022-2025
    labels = labels[(labels['year'] >= 2022) & (labels['year'] <= 2025)]
    
    # Exclude current-month outbreak labels as features
    cols_to_drop = [
        'outbreak_count_current_month',
        'outbreak_binary_current_month',
        'outbreak_count_next_month'
    ]
    labels = labels.drop(columns=[col for col in cols_to_drop if col in labels.columns])
    
    # 3. Load static features
    print(f"Loading static features from: {static_path}")
    static = pd.read_csv(static_path, dtype={'csd_uid': str})
    static_drop = ['csd_name', 'province_uid', 'province_name', 'ccs_name']
    static_clean = static.drop(columns=[col for col in static_drop if col in static.columns])
    
    # 4. Load climate features
    print(f"Loading climate features from: {climate_path}")
    climate = pd.read_csv(climate_path, dtype={'csd_uid': str})
    climate_cols = [
        'csd_uid', 'year', 'month', 'climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c',
        'climate_precip_mm', 'climate_missing_flag', 'climate_extraction_method',
        'climate_fallback_distance_km'
    ]
    climate_clean = climate[[col for col in climate_cols if col in climate.columns]]
    
    # 5. Load Dynamic World filled features
    print(f"Loading Dynamic World filled features from: {dw_path}")
    dw = pd.read_csv(dw_path, dtype={'csd_uid': str})
    dw_cols = [
        'csd_uid', 'year', 'month', 'dw_water_mean', 'dw_water_p95', 'dw_flooded_vegetation_mean',
        'dw_crops_mean', 'dw_grass_mean', 'dw_trees_mean', 'dw_built_mean', 'dw_bare_mean',
        'dw_seasonal_pond_proxy', 'dw_image_count', 'dw_missing_flag', 'dw_missing_flag_original',
        'dw_missing_flag_final', 'dw_fill_method'
    ]
    dw_clean = dw[[col for col in dw_cols if col in dw.columns]]
    
    # 6. Join dynamic features first by csd_uid/year/month
    # Inner join on dw restricts to the 5,054 provincial CSDs covered by Dynamic World
    print("Merging features...")
    df = pd.merge(labels, dw_clean, on=['csd_uid', 'year', 'month'], how='inner')
    df = pd.merge(df, climate_clean, on=['csd_uid', 'year', 'month'], how='left')
    
    # 7. Join static features by csd_uid
    df = pd.merge(df, static_clean, on='csd_uid', how='left')
    
    # Ensure standard metadata columns exist at start
    meta_cols = ['csd_uid', 'csd_name', 'province_code', 'province_name', 'year', 'month']
    # If province_code isn't in Labels, try province_u
    if 'province_code' not in df.columns and 'province_u' in df.columns:
        df['province_code'] = df['province_u']
        
    # Drop coordinates or geometry columns if any somehow ended up in columns
    geometry_cols = ['geometry', '.geo', 'geo', 'decimalLatitude', 'decimalLongitude']
    df = df.drop(columns=[col for col in geometry_cols if col in df.columns], errors='ignore')
    
    # QA validation
    errors = []
    
    # Check duplicates
    dup_count = df.duplicated(subset=['csd_uid', 'year', 'month']).sum()
    if dup_count > 0:
        errors.append(f"Duplicate CSD-month rows found: {dup_count}")
        
    # Expected row count: 5054 CSDs * 48 months = 242592
    expected_rows = 242592
    actual_rows = len(df)
    if actual_rows != expected_rows:
        errors.append(f"Row count mismatch: expected {expected_rows}, got {actual_rows}")
        
    # Unique CSDs
    unique_csds = df['csd_uid'].nunique()
    expected_csds = 5054
    if unique_csds != expected_csds:
        errors.append(f"Unique CSD count mismatch: expected {expected_csds}, got {unique_csds}")
        
    # Check target values
    if 'outbreak_next_month' not in df.columns:
        errors.append("Target column 'outbreak_next_month' is missing")
    else:
        # Check values
        unique_targets = df['outbreak_next_month'].unique()
        for val in unique_targets:
            if val not in [0.0, 1.0, 0, 1] and not np.isnan(val):
                errors.append(f"Target contains invalid value: {val}")
                
    # Group features by type
    dw_feature_names = [col for col in df.columns if col.startswith('dw_')]
    climate_feature_names = [col for col in df.columns if col.startswith('climate_')]
    static_feature_names = [col for col in df.columns if col.startswith('ccs_')]
    
    # Print QA Summary
    print("\n================== QA Summary ================== ")
    print(f"Total row count: {actual_rows} (Expected: {expected_rows})")
    print(f"Unique CSD count: {unique_csds} (Expected: {expected_csds})")
    print(f"Duplicate rows: {dup_count}")
    print(f"Years represented: {sorted(df['year'].unique())}")
    print(f"Months represented: {sorted(df['month'].unique())}")
    
    if 'outbreak_next_month' in df.columns:
        pos_count = int(df['outbreak_next_month'].sum())
        pos_pct = pos_count / actual_rows * 100
        print(f"Target (outbreak_next_month):")
        print(f"  - Positive cases: {pos_count} ({pos_pct:.4f}%)")
        print(f"  - Negative cases: {actual_rows - pos_count} ({100 - pos_pct:.4f}%)")
        
    print("\nMissing values by feature group:")
    print(f"  - Metadata group: {df[meta_cols].isna().sum().sum()} NaNs")
    print(f"  - Climate group: {df[climate_feature_names].isna().sum().sum()} NaNs")
    print(f"  - Dynamic World group: {df[dw_feature_names].isna().sum().sum()} NaNs")
    print(f"  - Static group: {df[static_feature_names].isna().sum().sum()} NaNs")
    
    print(f"\nFinal column count: {len(df.columns)}")
    print("First 5 column names by group:")
    print(f"  - Metadata: {meta_cols[:5]}")
    print(f"  - Climate: {climate_feature_names[:5]}")
    print(f"  - Dynamic World: {dw_feature_names[:5]}")
    print(f"  - Static: {static_feature_names[:5]}")
    print(f"  - Target: ['outbreak_next_month']")
    print("================================================\n")
    
    if len(errors) > 0:
        print("Validation errors occurred:")
        for err in errors:
            print(f"  [ERROR] {err}")
        raise ValueError("Model table validation failed. Output was not saved.")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved modeling table to: {output_path}")

if __name__ == '__main__':
    default_labels = 'data/processed/cfia_mapping/csd_month_labels.csv'
    default_static = 'data/processed/features/csd_static_structural_features.csv'
    default_climate = 'data/processed/features/csd_monthly_climate_2022_2025_with_fallback.csv'
    default_dw = 'data/processed/features/csd_monthly_dynamic_world_2022_2025_filled.csv'
    default_output = 'data/processed/modeling/model_table_v1_environment_only.csv'
    
    labels_file = sys.argv[1] if len(sys.argv) > 1 else default_labels
    static_file = sys.argv[2] if len(sys.argv) > 2 else default_static
    climate_file = sys.argv[3] if len(sys.argv) > 3 else default_climate
    dw_file = sys.argv[4] if len(sys.argv) > 4 else default_dw
    output_file = sys.argv[5] if len(sys.argv) > 5 else default_output
    
    try:
        assemble_environment_table(labels_file, static_file, climate_file, dw_file, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
