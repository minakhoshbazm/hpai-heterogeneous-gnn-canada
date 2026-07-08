import os
import sys
import pandas as pd
import numpy as np

def postprocess_national_csv(input_path, output_path, expected_rows=61932, expected_csds=5161):
    print(f"Reading raw national GEE export from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # 1. Read input CSV, preserving csd_uid as string
    df = pd.read_csv(input_path, dtype={'csd_uid': str})
    
    # 2. Drop GEE columns if present
    for col in ['system:index', '.geo', 'geo']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # 3. Ensure all desired columns exist (if some are completely omitted, create them as NaN)
    required_cols = [
        'csd_uid', 'csd_name', 'province_u', 'province_n', 'year', 'month',
        'climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm',
        'climate_missing_flag'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    # Keep only target columns in correct order
    df = df[required_cols]
    
    # 4. Convert numeric columns
    numeric_cols = ['year', 'month', 'climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm', 'climate_missing_flag']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
        
    # 5. Run Validation
    errors = []
    
    # Check row count
    if len(df) != expected_rows:
        errors.append(f"Row count validation failed: expected {expected_rows} rows, got {len(df)}")
        
    # Unique CSD count
    unique_csds = df['csd_uid'].nunique()
    if unique_csds != expected_csds:
        errors.append(f"Unique CSD count validation failed: expected {expected_csds}, got {unique_csds}")
        
    # Months are 1-12
    months_in_df = df['month'].dropna().unique()
    expected_months = set(range(1, 13))
    if set(months_in_df) != expected_months:
        errors.append(f"Months validation failed: expected months 1-12, got {sorted(months_in_df)}")
        
    # Duplicate csd_uid/year/month rows
    dup_mask = df.duplicated(subset=['csd_uid', 'year', 'month'])
    dup_count = dup_mask.sum()
    if dup_count > 0:
        errors.append(f"Duplicates check failed: found {dup_count} duplicate CSD-month rows")
        
    # climate_missing_flag validation
    flags = df['climate_missing_flag'].dropna().unique()
    invalid_flags = [f for f in flags if f not in [0, 1]]
    if len(invalid_flags) > 0:
        errors.append(f"climate_missing_flag validation failed: flags must be 0 or 1, found {invalid_flags}")
        
    # flag equals 1 if any climate value is missing
    climate_vars = ['climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm']
    for idx, row in df.iterrows():
        any_nan = row[climate_vars].isna().any()
        flag = row['climate_missing_flag']
        if any_nan and flag != 1:
            errors.append(f"Missing flag logic failed at row {idx} for CSD {row['csd_uid']}: variables are missing but flag is not 1")
        elif not any_nan and flag == 1:
            errors.append(f"Missing flag logic failed at row {idx} for CSD {row['csd_uid']}: variables are populated but flag is 1")
            
    # precipitation never negative
    if (df['climate_precip_mm'] < 0).any():
        neg_count = (df['climate_precip_mm'] < 0).sum()
        errors.append(f"Precipitation check failed: found {neg_count} negative precipitation values")
        
    # temperature ordering: tmin <= tmean <= tmax
    valid_temp_mask = df[climate_vars].notna().all(axis=1)
    temp_check_df = df[valid_temp_mask]
    invalid_temp_order = temp_check_df[
        (temp_check_df['climate_tmin_c'] > temp_check_df['climate_tmean_c']) | 
        (temp_check_df['climate_tmean_c'] > temp_check_df['climate_tmax_c'])
    ]
    if len(invalid_temp_order) > 0:
        errors.append(f"Temperature ordering check failed: found {len(invalid_temp_order)} rows where tmin <= tmean <= tmax was violated")
        
    # 6. Print QA summary
    print("\n================== QA Summary ==================")
    print(f"Row count: {len(df)}")
    print(f"Unique CSD count: {unique_csds}")
    print(f"Duplicate CSD-month rows: {dup_count}")
    
    missing_count = (df['climate_missing_flag'] == 1).sum()
    complete_count = (df['climate_missing_flag'] == 0).sum()
    print(f"Complete rows (flag = 0): {complete_count}")
    print(f"Missing rows (flag = 1): {missing_count}")
    
    missing_csds_series = df[df['climate_missing_flag'] == 1]['csd_uid'].unique()
    num_missing_csds = len(missing_csds_series)
    print(f"Number of CSDs with any missing climate values: {num_missing_csds}")
    
    if num_missing_csds > 0:
        print("\nMissing CSDs (showing up to 20):")
        missing_by_csd = df[df['climate_missing_flag'] == 1].groupby('csd_uid').size()
        for idx, (csd, cnt) in enumerate(missing_by_csd.items()):
            if idx >= 20:
                print(f"  - ... and {num_missing_csds - 20} more CSDs")
                break
            csd_name = df[df['csd_uid'] == csd]['csd_name'].iloc[0]
            print(f"  - CSD {csd} ({csd_name}): {cnt} missing months")
            
    print("\nClimate Variables Ranges:")
    for var in climate_vars:
        if df[var].notna().any():
            print(f"  - {var}: min={df[var].min():.2f}, max={df[var].max():.2f}")
        else:
            print(f"  - {var}: All values missing")
    print("================================================\n")
    
    # 7. Check for validation errors and write
    if len(errors) > 0:
        print("Validation errors occurred:")
        for err in errors:
            print(f"  [ERROR] {err}")
        raise ValueError("CSV post-processing failed validation checks. Output was not saved.")
        
    print(f"All validation checks passed! Saving cleaned CSV to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Cleaned CSV saved successfully.")

if __name__ == '__main__':
    default_input = "data/raw/gee_exports/era5_land_climate_national_2022.csv"
    default_output = "data/processed/features/pilot_csd_monthly_climate_features_2022_national.csv"
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    try:
        postprocess_national_csv(input_file, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
