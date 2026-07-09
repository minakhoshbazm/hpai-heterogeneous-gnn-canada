import os
import sys
import pandas as pd
import numpy as np

def postprocess_multiyear(input_files, output_path):
    print("Reading and validating GEE exports for 2022-2025...")
    
    target_cols = [
        'csd_uid', 'csd_name', 'province_u', 'province_n', 'year', 'month',
        'climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm',
        'climate_missing_flag', 'climate_extraction_method', 'climate_fallback_distance_km'
    ]
    numeric_cols = [
        'province_u', 'year', 'month',
        'climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm',
        'climate_missing_flag', 'climate_fallback_distance_km'
    ]
    
    dfs = []
    errors = []
    
    # 1-6. Read and validate each year's file individually
    for filepath in input_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required input file not found: {filepath}")
            
        print(f"  - Loading and checking: {filepath}")
        df = pd.read_csv(filepath, dtype={'csd_uid': str})
        
        # Drop GEE metadata
        for col in ['system:index', '.geo', 'geo']:
            if col in df.columns:
                df = df.drop(columns=[col])
                
        # Ensure exact columns exist
        for col in target_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[target_cols]
        
        # Convert numeric
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Get year for checking
        years_present = df['year'].dropna().unique()
        if len(years_present) == 1:
            y = int(years_present[0])
        else:
            y = f"mixed_{years_present}"
            errors.append(f"[{filepath}] File contains mixed or missing years: {years_present}")
            
        # Validate individual year
        expected_rows = 61932
        if len(df) != expected_rows:
            errors.append(f"[{filepath}] Row count validation failed: expected {expected_rows}, got {len(df)}")
            
        expected_csds = 5161
        unique_csds = df['csd_uid'].nunique()
        if unique_csds != expected_csds:
            errors.append(f"[{filepath}] Unique CSD count validation failed: expected {expected_csds}, got {unique_csds}")
            
        months_in_df = df['month'].dropna().unique()
        expected_months = set(range(1, 13))
        if set(months_in_df) != expected_months:
            errors.append(f"[{filepath}] Months validation failed: expected 1-12, got {sorted(months_in_df)}")
            
        dup_mask = df.duplicated(subset=['csd_uid', 'year', 'month'])
        dup_count = dup_mask.sum()
        if dup_count > 0:
            errors.append(f"[{filepath}] Duplicates check failed: found {dup_count} duplicate CSD-month rows")
            
        dfs.append(df)
        
    # Combine dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 7. Validate combined dataframe
    print("Validating combined dataset...")
    
    expected_total_rows = 247728
    if len(combined_df) != expected_total_rows:
        errors.append(f"[Combined] Total row count failed: expected {expected_total_rows}, got {len(combined_df)}")
        
    expected_years = {2022, 2023, 2024, 2025}
    actual_years = set(combined_df['year'].dropna().unique())
    if actual_years != expected_years:
        errors.append(f"[Combined] Years check failed: expected {expected_years}, got {actual_years}")
        
    comb_dup_mask = combined_df.duplicated(subset=['csd_uid', 'year', 'month'])
    comb_dup_count = comb_dup_mask.sum()
    if comb_dup_count > 0:
        errors.append(f"[Combined] Duplicates check failed: found {comb_dup_count} duplicate CSD-year-month rows")
        
    flags = combined_df['climate_missing_flag'].dropna().unique()
    invalid_flags = [f for f in flags if f not in [0, 1]]
    if len(invalid_flags) > 0:
        errors.append(f"[Combined] climate_missing_flag validation failed: flags must be 0 or 1, found {invalid_flags}")
        
    if (combined_df['climate_precip_mm'] < 0).any():
        neg_precip_count = (combined_df['climate_precip_mm'] < 0).sum()
        errors.append(f"[Combined] Precipitation check failed: found {neg_precip_count} negative precipitation values")
        
    valid_temp_mask = combined_df[['climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c']].notna().all(axis=1)
    temp_check = combined_df[valid_temp_mask]
    invalid_temp_order = temp_check[
        (temp_check['climate_tmin_c'] > temp_check['climate_tmean_c']) |
        (temp_check['climate_tmean_c'] > temp_check['climate_tmax_c'])
    ]
    if len(invalid_temp_order) > 0:
        errors.append(f"[Combined] Temperature ordering check failed: found {len(invalid_temp_order)} rows where tmin <= tmean <= tmax was violated")
        
    climate_vars = ['climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm']
    flag_zero_df = combined_df[combined_df['climate_missing_flag'] == 0]
    missing_in_flag_zero = flag_zero_df[climate_vars].isna().any(axis=1)
    if missing_in_flag_zero.any():
        missing_count = missing_in_flag_zero.sum()
        errors.append(f"[Combined] Flag zero logic failed: found {missing_count} rows with missing climate values where climate_missing_flag = 0")
        
    # 8. Print QA summary
    print("\n================== QA Summary ==================")
    print(f"Total row count: {len(combined_df)} (Expected: {expected_total_rows})")
    
    # Row count by year
    print("\nRow count by year:")
    row_by_year = combined_df['year'].value_counts().sort_index()
    for yr, cnt in row_by_year.items():
        print(f"  - {int(yr)}: {cnt}")
        
    # Method counts by year
    print("\nMethod counts by year:")
    method_by_yr = combined_df.groupby(['year', 'climate_extraction_method'], dropna=False).size().unstack(fill_value=0)
    for yr in sorted(method_by_yr.index):
        print(f"  - {int(yr)}:")
        for method in method_by_yr.columns:
            print(f"    * {method}: {method_by_yr.loc[yr, method]}")
            
    # Missing flag counts by year
    print("\nclimate_missing_flag counts by year:")
    flag_by_yr = combined_df.groupby(['year', 'climate_missing_flag'], dropna=False).size().unstack(fill_value=0)
    for yr in sorted(flag_by_yr.index):
        print(f"  - {int(yr)}:")
        for flag in flag_by_yr.columns:
            print(f"    * {int(flag)}: {flag_by_yr.loc[yr, flag]}")
            
    # Min/max climate values
    print("\nClimate Variables Ranges (Combined):")
    for var in climate_vars:
        if combined_df[var].notna().any():
            print(f"  - {var}: min={combined_df[var].min():.2f}, max={combined_df[var].max():.2f}")
        else:
            print(f"  - {var}: All values missing")
            
    if combined_df['climate_fallback_distance_km'].notna().any():
        print(f"  - climate_fallback_distance_km: min={combined_df['climate_fallback_distance_km'].min():.2f}, max={combined_df['climate_fallback_distance_km'].max():.2f}")
    else:
        print("  - climate_fallback_distance_km: All values missing/null")
    print("================================================\n")
    
    # 9. Save output only if checks pass
    if len(errors) > 0:
        print("Validation errors occurred:")
        for err in errors:
            print(f"  [ERROR] {err}")
        raise ValueError("Combined CSV validation failed. Output was not saved.")
        
    print(f"All validation checks passed! Saving combined CSV to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print("Combined CSV saved successfully.")

if __name__ == '__main__':
    default_inputs = [
        'data/raw/gee_exports/era5_land_climate_national_2022_with_fallback.csv',
        'data/raw/gee_exports/era5_land_climate_national_2023_with_fallback.csv',
        'data/raw/gee_exports/era5_land_climate_national_2024_with_fallback.csv',
        'data/raw/gee_exports/era5_land_climate_national_2025_with_fallback.csv'
    ]
    default_output = 'data/processed/features/csd_monthly_climate_2022_2025_with_fallback.csv'
    
    # Allow override input files via CLI arguments, otherwise use default
    if len(sys.argv) > 1:
        # If files are specified, let last arg be output, others be inputs
        input_files = sys.argv[1:-1]
        output_file = sys.argv[-1]
    else:
        input_files = default_inputs
        output_file = default_output
        
    try:
        postprocess_multiyear(input_files, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
