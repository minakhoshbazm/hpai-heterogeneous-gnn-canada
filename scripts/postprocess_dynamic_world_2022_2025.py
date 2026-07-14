import os
import sys
import pandas as pd
import numpy as np

def postprocess_dw_multiyear(input_dir, output_path):
    expected_provinces = [10, 11, 12, 13, 24, 35, 46, 47, 48, 59]
    expected_years = [2022, 2023, 2024, 2025]
    
    target_cols = [
        'csd_uid', 'csd_name', 'province_u', 'year', 'month',
        'dw_water_mean', 'dw_water_p95', 'dw_flooded_vegetation_mean',
        'dw_crops_mean', 'dw_grass_mean', 'dw_trees_mean', 'dw_built_mean',
        'dw_bare_mean', 'dw_seasonal_pond_proxy', 'dw_image_count', 'dw_missing_flag'
    ]
    
    numeric_cols = [
        'province_u', 'year', 'month',
        'dw_water_mean', 'dw_water_p95', 'dw_flooded_vegetation_mean',
        'dw_crops_mean', 'dw_grass_mean', 'dw_trees_mean', 'dw_built_mean',
        'dw_bare_mean', 'dw_seasonal_pond_proxy', 'dw_image_count', 'dw_missing_flag'
    ]
    
    errors = []
    missing_files = []
    found_files = []
    
    # 1. Validate files exist for all expected combinations
    for prov in expected_provinces:
        for year in expected_years:
            filename = f"dynamic_world_province_{prov}_{year}.csv"
            filepath = os.path.join(input_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filepath)
            else:
                found_files.append(filepath)
                
    if len(missing_files) > 0:
        errors.append(f"Missing {len(missing_files)} expected province-year files: {missing_files}")
        print("Validation failed: Missing raw files.")
        for f in missing_files:
            print(f"  [MISSING] {f}")
            
    # Load and combine all found files (even if validation fails, so we can report diagnostics for found files)
    dfs = []
    for filepath in found_files:
        try:
            df = pd.read_csv(filepath, dtype={'csd_uid': str})
            # GEE exports empty files with just headers if no features match, skip or keep?
            # We should keep it as long as schema matches, but pandas handles empty rows fine.
            if len(df) == 0:
                continue
                
            # Drop GEE metadata columns
            for col in ['system:index', '.geo', 'geo']:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    
            # Extract year from filename to override potential GEE export labeling bugs
            filename_part = os.path.basename(filepath)
            year_from_file = int(filename_part.split('_')[-1].split('.')[0])
            df['year'] = year_from_file
            
            # Keep exactly target columns
            for col in target_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[target_cols]
            
            # Convert numeric columns
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            dfs.append(df)
        except Exception as e:
            errors.append(f"Error reading file {filepath}: {e}")
            
    if len(dfs) == 0:
        if len(errors) > 0:
            raise ValueError(f"No valid files loaded. Errors: {errors}")
        else:
            raise ValueError("No files found or loaded.")
            
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 2. Validation checks on combined data
    # Months are 1-12
    actual_months = set(combined_df['month'].dropna().unique())
    expected_months = set(range(1, 13))
    if not actual_months.issubset(expected_months):
        errors.append(f"Months validation failed: must be 1-12, got {actual_months}")
        
    # No duplicate csd_uid/year/month rows
    dup_mask = combined_df.duplicated(subset=['csd_uid', 'year', 'month'])
    dup_count = dup_mask.sum()
    if dup_count > 0:
        errors.append(f"Duplicates check failed: found {dup_count} duplicate CSD-year-month rows")
        
    # Probability values between 0 and 1 where present (allowing slight float tolerance)
    prob_cols = [
        'dw_water_mean', 'dw_water_p95', 'dw_flooded_vegetation_mean',
        'dw_crops_mean', 'dw_grass_mean', 'dw_trees_mean', 'dw_built_mean', 'dw_bare_mean'
    ]
    for col in prob_cols:
        col_vals = combined_df[col].dropna()
        if not col_vals.empty:
            if (col_vals < -1e-5).any() or (col_vals > 1.00001).any():
                errors.append(f"Probability range validation failed for {col}: values must be in [0, 1]")
                
    # dw_seasonal_pond_proxy between 0 and 1 where present
    if combined_df['dw_seasonal_pond_proxy'].notna().any():
        proxy_vals = combined_df['dw_seasonal_pond_proxy'].dropna()
        if (proxy_vals < -1e-5).any() or (proxy_vals > 1.00001).any():
            errors.append(f"dw_seasonal_pond_proxy range validation failed: values must be in [0, 1]")
            
    # dw_missing_flag only 0 or 1
    flags = combined_df['dw_missing_flag'].dropna().unique()
    invalid_flags = [f for f in flags if f not in [0, 1]]
    if len(invalid_flags) > 0:
        errors.append(f"dw_missing_flag validation failed: flags must be 0 or 1, found {invalid_flags}")
        
    # 3. Print QA summary
    print("\n================== QA Summary ==================")
    print(f"Number of expected files found: {len(found_files)} / {len(expected_provinces) * len(expected_years)}")
    print(f"Total row count of combined data: {len(combined_df)}")
    print(f"Unique CSD count: {combined_df['csd_uid'].nunique()}")
    print(f"Duplicate rows: {dup_count}")
    
    # Row count by year
    print("\nRow count by year:")
    row_by_yr = combined_df['year'].value_counts().sort_index()
    for yr, cnt in row_by_yr.items():
        print(f"  - {int(yr)}: {cnt}")
        
    # Row count by province
    print("\nRow count by province:")
    row_by_prov = combined_df['province_u'].value_counts().sort_index()
    for prov, cnt in row_by_prov.items():
        print(f"  - Province {int(prov)}: {cnt}")
        
    # Missing flag counts by year
    print("\ndw_missing_flag counts by year:")
    if 'dw_missing_flag' in combined_df.columns:
        flag_by_yr = combined_df.groupby(['year', 'dw_missing_flag'], dropna=False).size().unstack(fill_value=0)
        for yr in sorted(flag_by_yr.index):
            print(f"  - {int(yr)}:")
            for flag in flag_by_yr.columns:
                flag_str = 'NaN' if pd.isna(flag) else str(int(flag))
                print(f"    * Flag {flag_str}: {flag_by_yr.loc[yr, flag]}")
                
    # Missing flag counts by province
    print("\ndw_missing_flag counts by province:")
    if 'dw_missing_flag' in combined_df.columns:
        flag_by_prov = combined_df.groupby(['province_u', 'dw_missing_flag'], dropna=False).size().unstack(fill_value=0)
        for prov in sorted(flag_by_prov.index):
            print(f"  - Province {int(prov)}:")
            for flag in flag_by_prov.columns:
                flag_str = 'NaN' if pd.isna(flag) else str(int(flag))
                print(f"    * Flag {flag_str}: {flag_by_prov.loc[prov, flag]}")
                
    # Min/max for Dynamic World features
    print("\nDynamic World Features Ranges:")
    all_dw_features = prob_cols + ['dw_seasonal_pond_proxy']
    for var in all_dw_features:
        if combined_df[var].notna().any():
            print(f"  - {var}: min={combined_df[var].min():.4f}, max={combined_df[var].max():.4f}")
        else:
            print(f"  - {var}: All values missing/null")
            
    # Top 20 seasonal pond proxy rows
    print("\nTop 20 highest seasonal pond proxy values:")
    if combined_df['dw_seasonal_pond_proxy'].notna().any():
        highest_proxy = combined_df.sort_values(by='dw_seasonal_pond_proxy', ascending=False).head(20)
        for _, row in highest_proxy.iterrows():
            print(f"  - CSD {row['csd_uid']} ({row['csd_name']}), Prov {int(row['province_u'])}, Yr {int(row['year'])}, M {int(row['month'])}: proxy={row['dw_seasonal_pond_proxy']:.4f} (water_mean={row['dw_water_mean']:.4f}, water_p95={row['dw_water_p95']:.4f})")
    else:
        print("  - No valid proxy values found.")
    print("================================================\n")
    
    # 4. Check errors
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
    default_input_dir = 'data/raw/gee_exports'
    default_output = 'data/processed/features/csd_monthly_dynamic_world_2022_2025.csv'
    
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    try:
        postprocess_dw_multiyear(input_dir, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
