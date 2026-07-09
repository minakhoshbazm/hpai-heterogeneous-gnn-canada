import os
import sys
import pandas as pd
import numpy as np

def postprocess_fallback_csv(input_path, output_path):
    print(f"Reading GEE export CSV from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # 1. Read CSV, preserving csd_uid as string
    df = pd.read_csv(input_path, dtype={'csd_uid': str})
    
    # 2. Drop GEE metadata columns if present
    for col in ['system:index', '.geo', 'geo']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # 3. Ensure exact columns are kept in correct order
    target_cols = [
        'csd_uid', 'csd_name', 'province_u', 'province_n', 'year', 'month',
        'climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm',
        'climate_missing_flag', 'climate_extraction_method', 'climate_fallback_distance_km'
    ]
    for col in target_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[target_cols]
    
    # 4. Convert numeric columns
    numeric_cols = [
        'province_u', 'year', 'month',
        'climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm',
        'climate_missing_flag', 'climate_fallback_distance_km'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 5. Run Validation Checks
    errors = []
    
    # Row count = 61932
    expected_rows = 61932
    if len(df) != expected_rows:
        errors.append(f"Row count validation failed: expected {expected_rows}, got {len(df)}")
        
    # Unique CSDs = 5161
    expected_csds = 5161
    unique_csds = df['csd_uid'].nunique()
    if unique_csds != expected_csds:
        errors.append(f"Unique CSD count validation failed: expected {expected_csds}, got {unique_csds}")
        
    # Months = 1-12
    months_in_df = df['month'].dropna().unique()
    expected_months = set(range(1, 13))
    if set(months_in_df) != expected_months:
        errors.append(f"Months validation failed: expected 1-12, got {sorted(months_in_df)}")
        
    # No duplicate csd_uid/year/month rows
    dup_mask = df.duplicated(subset=['csd_uid', 'year', 'month'])
    dup_count = dup_mask.sum()
    if dup_count > 0:
        errors.append(f"Duplicates check failed: found {dup_count} duplicate CSD-month rows")
        
    # climate_missing_flag only 0 or 1
    flags = df['climate_missing_flag'].dropna().unique()
    invalid_flags = [f for f in flags if f not in [0, 1]]
    if len(invalid_flags) > 0:
        errors.append(f"climate_missing_flag validation failed: flags must be 0 or 1, found {invalid_flags}")
        
    # No negative precipitation
    if (df['climate_precip_mm'] < 0).any():
        neg_precip_count = (df['climate_precip_mm'] < 0).sum()
        errors.append(f"Precipitation check failed: found {neg_precip_count} negative precipitation values")
        
    # tmin <= tmean <= tmax where values exist
    valid_temp_mask = df[['climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c']].notna().all(axis=1)
    temp_check = df[valid_temp_mask]
    invalid_temp_order = temp_check[
        (temp_check['climate_tmin_c'] > temp_check['climate_tmean_c']) |
        (temp_check['climate_tmean_c'] > temp_check['climate_tmax_c'])
    ]
    if len(invalid_temp_order) > 0:
        errors.append(f"Temperature ordering check failed: found {len(invalid_temp_order)} rows where tmin <= tmean <= tmax was violated")
        
    # No missing climate values if climate_missing_flag = 0
    climate_vars = ['climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm']
    flag_zero_df = df[df['climate_missing_flag'] == 0]
    missing_in_flag_zero = flag_zero_df[climate_vars].isna().any(axis=1)
    if missing_in_flag_zero.any():
        missing_count = missing_in_flag_zero.sum()
        errors.append(f"Flag zero logic failed: found {missing_count} rows with missing climate values where climate_missing_flag = 0")
        
    # 6. Print QA summary
    print("\n================== QA Summary ==================")
    print(f"Row count: {len(df)} (Expected: {expected_rows})")
    print(f"Unique CSDs: {unique_csds} (Expected: {expected_csds})")
    print(f"Duplicate rows: {dup_count}")
    
    # Method counts
    method_counts = df['climate_extraction_method'].value_counts(dropna=False)
    print("\nMethod Counts:")
    for method, count in method_counts.items():
        print(f"  - {method}: {count}")
        
    # Missing flag counts
    flag_counts = df['climate_missing_flag'].value_counts(dropna=False)
    print("\nclimate_missing_flag Counts:")
    for flag, count in flag_counts.items():
        print(f"  - {flag}: {count}")
        
    # Min/max climate values
    print("\nClimate Variables Ranges:")
    for var in climate_vars:
        if df[var].notna().any():
            print(f"  - {var}: min={df[var].min():.2f}, max={df[var].max():.2f}")
        else:
            print(f"  - {var}: All values missing")
            
    if df['climate_fallback_distance_km'].notna().any():
        print(f"  - climate_fallback_distance_km: min={df['climate_fallback_distance_km'].min():.2f}, max={df['climate_fallback_distance_km'].max():.2f}")
    else:
        print("  - climate_fallback_distance_km: All values missing/null")
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
    default_input = 'data/raw/gee_exports/era5_land_climate_national_2022_with_fallback.csv'
    default_output = 'data/processed/features/csd_monthly_climate_2022_with_fallback.csv'
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    try:
        postprocess_fallback_csv(input_file, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
