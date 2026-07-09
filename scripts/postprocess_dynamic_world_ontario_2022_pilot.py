import os
import sys
import pandas as pd
import numpy as np

def postprocess_dw_ontario_pilot(input_path, output_path):
    print(f"Reading Dynamic World Ontario pilot CSV from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # 1. Read CSV, preserving csd_uid as string
    df = pd.read_csv(input_path, dtype={'csd_uid': str})
    
    # 2. Drop GEE metadata columns
    for col in ['system:index', '.geo', 'geo']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # 3. Ensure exact columns are kept in correct order
    target_cols = [
        'csd_uid', 'csd_name', 'province_u', 'year', 'month',
        'dw_water_mean', 'dw_water_p95', 'dw_flooded_vegetation_mean',
        'dw_crops_mean', 'dw_grass_mean', 'dw_trees_mean', 'dw_built_mean',
        'dw_bare_mean', 'dw_seasonal_pond_proxy', 'dw_image_count', 'dw_missing_flag'
    ]
    for col in target_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[target_cols]
    
    # 4. Convert numeric columns
    numeric_cols = [
        'province_u', 'year', 'month',
        'dw_water_mean', 'dw_water_p95', 'dw_flooded_vegetation_mean',
        'dw_crops_mean', 'dw_grass_mean', 'dw_trees_mean', 'dw_built_mean',
        'dw_bare_mean', 'dw_seasonal_pond_proxy', 'dw_image_count', 'dw_missing_flag'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 5. Run Validation Checks
    errors = []
    
    # Months are 1-12
    expected_months = set(range(1, 13))
    actual_months = set(df['month'].dropna().unique())
    if not actual_months.issubset(expected_months):
        errors.append(f"Months validation failed: must be subset of {expected_months}, got {actual_months}")
        
    # No duplicate csd_uid/year/month rows
    dup_mask = df.duplicated(subset=['csd_uid', 'year', 'month'])
    dup_count = dup_mask.sum()
    if dup_count > 0:
        errors.append(f"Duplicates check failed: found {dup_count} duplicate CSD-month rows")
        
    # province_u should be 35
    provs = df['province_u'].dropna().unique()
    if len(provs) != 1 or provs[0] != 35:
        errors.append(f"province_u validation failed: expected only 35 (Ontario), got {provs}")
        
    # Probability values between 0 and 1 where present
    prob_cols = [
        'dw_water_mean', 'dw_water_p95', 'dw_flooded_vegetation_mean',
        'dw_crops_mean', 'dw_grass_mean', 'dw_trees_mean', 'dw_built_mean', 'dw_bare_mean'
    ]
    for col in prob_cols:
        col_vals = df[col].dropna()
        if not col_vals.empty:
            # We allow slight float tolerances (e.g., -1e-6 to 1.000001) due to reduction rounding
            if (col_vals < -1e-5).any() or (col_vals > 1.00001).any():
                errors.append(f"Probability range validation failed for {col}: values must be in [0, 1]")
                
    # dw_seasonal_pond_proxy between 0 and 1 where present
    if df['dw_seasonal_pond_proxy'].notna().any():
        proxy_vals = df['dw_seasonal_pond_proxy'].dropna()
        if (proxy_vals < -1e-5).any() or (proxy_vals > 1.00001).any():
            errors.append(f"dw_seasonal_pond_proxy range validation failed: values must be in [0, 1]")
            
    # dw_image_count >= 0
    if df['dw_image_count'].notna().any():
        if (df['dw_image_count'] < 0).any():
            errors.append("dw_image_count validation failed: counts cannot be negative")
            
    # dw_missing_flag only 0 or 1
    flags = df['dw_missing_flag'].dropna().unique()
    invalid_flags = [f for f in flags if f not in [0, 1]]
    if len(invalid_flags) > 0:
        errors.append(f"dw_missing_flag validation failed: flags must be 0 or 1, found {invalid_flags}")
        
    # Row and CSD Counts
    unique_csds = df['csd_uid'].nunique()
    expected_rows = unique_csds * 12
    if len(df) != expected_rows:
        errors.append(f"Row count validation failed: expected {expected_rows} (unique CSDs {unique_csds} * 12), got {len(df)}")
        
    # 6. Print QA summary
    print("\n================== QA Summary ==================")
    print(f"Row count: {len(df)} (Expected: {expected_rows})")
    print(f"Unique CSD count: {unique_csds}")
    print(f"Duplicate rows: {dup_count}")
    
    if df['dw_image_count'].notna().any():
        print(f"Image count range: min={int(df['dw_image_count'].min())}, max={int(df['dw_image_count'].max())}")
    else:
        print("Image count: all values missing")
        
    # Missing flag counts
    flag_counts = df['dw_missing_flag'].value_counts(dropna=False)
    print("\ndw_missing_flag counts:")
    for flag, count in flag_counts.items():
        print(f"  - {flag}: {count}")
        
    # Min/max for Dynamic World features
    print("\nDynamic World Features Ranges:")
    all_dw_features = prob_cols + ['dw_seasonal_pond_proxy']
    for var in all_dw_features:
        if df[var].notna().any():
            print(f"  - {var}: min={df[var].min():.4f}, max={df[var].max():.4f}")
        else:
            print(f"  - {var}: All values missing/null")
            
    # Top 10 seasonal pond proxy rows
    print("\nTop 10 highest seasonal pond proxy values:")
    if df['dw_seasonal_pond_proxy'].notna().any():
        highest_proxy = df.sort_values(by='dw_seasonal_pond_proxy', ascending=False).head(10)
        for _, row in highest_proxy.iterrows():
            print(f"  - CSD {row['csd_uid']} ({row['csd_name']}), Month {int(row['month'])}: proxy={row['dw_seasonal_pond_proxy']:.4f} (water_mean={row['dw_water_mean']:.4f}, water_p95={row['dw_water_p95']:.4f})")
    else:
        print("  - No valid proxy values found.")
    print("================================================\n")
    
    # 7. Check for validation errors and write
    if len(errors) > 0:
        print("Validation errors occurred:")
        for err in errors:
            print(f"  [ERROR] {err}")
        raise ValueError("CSV validation failed. Output was not saved.")
        
    print(f"All validation checks passed! Saving cleaned CSV to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Cleaned CSV saved successfully.")

if __name__ == '__main__':
    default_input = 'data/raw/gee_exports/dynamic_world_ontario_2022_pilot.csv'
    default_output = 'data/processed/features/dynamic_world_ontario_2022_pilot_clean.csv'
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    try:
        postprocess_dw_ontario_pilot(input_file, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
