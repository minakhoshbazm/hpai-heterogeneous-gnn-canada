import os
import sys
import pandas as pd
import numpy as np

def fill_missing_values(input_path, output_path):
    print(f"Reading combined Dynamic World feature table from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # Read CSV, preserving csd_uid as string
    df = pd.read_csv(input_path, dtype={'csd_uid': str})
    
    # Validation checks on input row and CSD count
    expected_rows = 242592
    if len(df) != expected_rows:
        raise ValueError(f"Input row count mismatch: expected {expected_rows}, got {len(df)}")
        
    expected_csds = 5054
    unique_csds = df['csd_uid'].nunique()
    if unique_csds != expected_csds:
        raise ValueError(f"Input unique CSD count mismatch: expected {expected_csds}, got {unique_csds}")
        
    # Sort chronologically per CSD
    df['time_index'] = (df['year'] - 2022) * 12 + df['month']
    df = df.sort_values(by=['csd_uid', 'time_index']).reset_index(drop=True)
    
    cols_to_fill = [
        'dw_water_mean', 'dw_water_p95', 'dw_flooded_vegetation_mean',
        'dw_crops_mean', 'dw_grass_mean', 'dw_trees_mean', 'dw_built_mean',
        'dw_bare_mean', 'dw_seasonal_pond_proxy'
    ]
    
    # Initialize tracking columns
    df['dw_missing_flag_original'] = df['dw_missing_flag'].astype(int)
    df['dw_missing_flag_final'] = df['dw_missing_flag'].astype(int)
    df['dw_fill_method'] = 'observed'
    
    # Mark rows that are originally missing
    df.loc[df['dw_missing_flag_original'] == 1, 'dw_fill_method'] = 'missing'
    
    # Tier 2: Temporal 3-Month Median (rolling window 3, centered, per CSD)
    print("Applying Tier 2: Temporal 3-Month Median...")
    for col in cols_to_fill:
        # Groupby rolling requires sorting, which we did. 
        # min_periods=1 allows calculation if at least one month in window is valid
        df[f'rolled_{col}'] = df.groupby('csd_uid')[col].rolling(
            window=3, center=True, min_periods=1
        ).median().reset_index(level=0, drop=True)
        
    is_missing = (df['dw_fill_method'] == 'missing')
    rolled_not_null = df[[f'rolled_{col}' for col in cols_to_fill]].notna().all(axis=1)
    tier2_mask = is_missing & rolled_not_null
    
    for col in cols_to_fill:
        df.loc[tier2_mask, col] = df.loc[tier2_mask, f'rolled_{col}']
    df.loc[tier2_mask, 'dw_fill_method'] = 'temporal_3month_median'
    
    # Tier 3: Annual CSD Median
    print("Applying Tier 3: Annual CSD Median...")
    annual_medians = df.groupby(['csd_uid', 'year'])[cols_to_fill].transform('median')
    annual_not_null = annual_medians.notna().all(axis=1)
    
    is_missing = (df['dw_fill_method'] == 'missing')
    tier3_mask = is_missing & annual_not_null
    
    for col in cols_to_fill:
        df.loc[tier3_mask, col] = annual_medians.loc[tier3_mask, col]
    df.loc[tier3_mask, 'dw_fill_method'] = 'annual_csd_median'
    
    # Tier 4: Province-Month Median
    print("Applying Tier 4: Province-Month Median...")
    prov_month_medians = df.groupby(['province_u', 'month'])[cols_to_fill].transform('median')
    prov_month_not_null = prov_month_medians.notna().all(axis=1)
    
    is_missing = (df['dw_fill_method'] == 'missing')
    tier4_mask = is_missing & prov_month_not_null
    
    for col in cols_to_fill:
        df.loc[tier4_mask, col] = prov_month_medians.loc[tier4_mask, col]
    df.loc[tier4_mask, 'dw_fill_method'] = 'province_month_median'
    
    # Tier 5: Set final missing flag based on unresolved rows
    df['dw_missing_flag_final'] = (df['dw_fill_method'] == 'missing').astype(int)
    # Also update the primary missing flag column to reflect post-imputation status
    df['dw_missing_flag'] = df['dw_missing_flag_final']
    
    # Clean up temporary columns
    temp_cols = ['time_index'] + [f'rolled_{col}' for col in cols_to_fill]
    df = df.drop(columns=temp_cols)
    
    # Clip probability values to [0.0, 1.0]
    for col in cols_to_fill:
        df[col] = df[col].clip(0.0, 1.0)
        
    # QA Checks
    errors = []
    
    # Row count stays 242,592
    if len(df) != expected_rows:
        errors.append(f"Row count changed: expected {expected_rows}, got {len(df)}")
        
    # Unique CSD count stays 5,054
    final_csds = df['csd_uid'].nunique()
    if final_csds != expected_csds:
        errors.append(f"Unique CSD count changed: expected {expected_csds}, got {final_csds}")
        
    # No duplicate csd_uid/year/month rows
    dup_count = df.duplicated(subset=['csd_uid', 'year', 'month']).sum()
    if dup_count > 0:
        errors.append(f"Duplicates check failed: found {dup_count} duplicate CSD-month rows")
        
    # Verify feature ranges remain [0, 1]
    for col in cols_to_fill:
        col_vals = df[col].dropna()
        if (col_vals < 0).any() or (col_vals > 1).any():
            errors.append(f"Feature range validation failed for {col} after clipping")
            
    # Print QA Summary
    print("\n================== Imputation QA Summary ==================")
    print(f"Row count: {len(df)} (Expected: {expected_rows})")
    print(f"Unique CSD count: {final_csds} (Expected: {expected_csds})")
    print(f"Duplicate rows: {dup_count}")
    
    orig_missing = df['dw_missing_flag_original'].sum()
    final_missing = df['dw_missing_flag_final'].sum()
    print(f"\nMissing rows summary:")
    print(f"  - Original missing count: {orig_missing} ({orig_missing / len(df) * 100:.2f}%)")
    print(f"  - Final missing count: {final_missing} ({final_missing / len(df) * 100:.2f}%)")
    
    print("\nFill method counts:")
    method_counts = df['dw_fill_method'].value_counts()
    for method, count in method_counts.items():
        print(f"  - {method}: {count} ({count / len(df) * 100:.2f}%)")
        
    print("\nFeature Ranges (Post-Imputation):")
    for var in cols_to_fill:
        if df[var].notna().any():
            print(f"  - {var}: min={df[var].min():.4f}, max={df[var].max():.4f}")
        else:
            print(f"  - {var}: All values missing")
            
    # Print top seasonal pond proxy rows after filling
    print("\nTop 10 highest seasonal pond proxy values:")
    if df['dw_seasonal_pond_proxy'].notna().any():
        highest_proxy = df.sort_values(by='dw_seasonal_pond_proxy', ascending=False).head(10)
        for _, row in highest_proxy.iterrows():
            print(f"  - CSD {row['csd_uid']} ({row['csd_name']}), Month {int(row['month'])}, Yr {int(row['year'])}: proxy={row['dw_seasonal_pond_proxy']:.4f} (method={row['dw_fill_method']})")
    else:
        print("  - No valid proxy values found.")
    print("===========================================================\n")
    
    if len(errors) > 0:
        print("Imputation validation errors occurred:")
        for err in errors:
            print(f"  [ERROR] {err}")
        raise ValueError("Imputed CSV validation failed. Output was not saved.")
        
    print(f"All validation checks passed! Saving filled CSV to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Filled CSV saved successfully.")

if __name__ == '__main__':
    default_input = 'data/processed/features/csd_monthly_dynamic_world_2022_2025.csv'
    default_output = 'data/processed/features/csd_monthly_dynamic_world_2022_2025_filled.csv'
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    try:
        fill_missing_values(input_file, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
