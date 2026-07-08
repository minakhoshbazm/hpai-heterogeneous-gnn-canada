import os
import sys
import pandas as pd
import numpy as np

def analyze_fallback_smoke_test(input_path, output_path=None, expected_rows=110, expected_methods=None, expected_months=None):
    print(f"Reading fallback smoke test CSV from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # 1. Read the CSV, preserving csd_uid as string
    df = pd.read_csv(input_path, dtype={'csd_uid': str})
    
    # 2. Drop unwanted GEE columns
    for col in ['system:index', '.geo', 'geo']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # 3. Ensure expected columns are present
    required_cols = [
        'csd_uid', 'csd_name', 'province_u', 'year', 'month', 'method_name',
        'climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm',
        'method_missing_flag'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[required_cols]
    
    # Convert numeric columns
    numeric_cols = ['province_u', 'year', 'month', 'climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm', 'method_missing_flag']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
        
    errors = []
    
    # 4. Validate row count
    if len(df) != expected_rows:
        errors.append(f"Row count validation failed: expected {expected_rows} rows, got {len(df)}")
        
    # Set default expected methods and months if None
    if expected_methods is None:
        expected_methods = {'polygon_mean', 'centroid_sample', 'buffer_25km', 'buffer_50km', 'buffer_100km'}
    else:
        expected_methods = set(expected_methods)
        
    if expected_months is None:
        expected_months = {1, 7}
    else:
        expected_months = set(expected_months)
        
    # 5. Validate expected methods
    actual_methods = set(df['method_name'].dropna().unique())
    if not actual_methods.issubset(expected_methods) or not expected_methods.issubset(actual_methods):
        errors.append(f"Expected methods validation failed. Expected {expected_methods}, got {actual_methods}")
        
    # 6. Validate expected months
    actual_months = set(df['month'].dropna().unique())
    if not actual_months.issubset(expected_months) or not expected_months.issubset(actual_months):
        errors.append(f"Expected months validation failed. Expected {expected_months}, got {actual_months}")
        
    # 7. Validate no duplicate csd_uid/year/month/method_name rows
    dup_mask = df.duplicated(subset=['csd_uid', 'year', 'month', 'method_name'])
    dup_count = dup_mask.sum()
    if dup_count > 0:
        errors.append(f"Duplicates check failed: found {dup_count} duplicate rows")
        
    # 8. Validate method_missing_flag is only 0 or 1
    flags = df['method_missing_flag'].dropna().unique()
    invalid_flags = [f for f in flags if f not in [0, 1]]
    if len(invalid_flags) > 0:
        errors.append(f"method_missing_flag validation failed: flags must be 0 or 1, found {invalid_flags}")
        
    # 9. Validate method_missing_flag matches missing values
    climate_vars = ['climate_tmean_c', 'climate_tmin_c', 'climate_tmax_c', 'climate_precip_mm']
    for idx, row in df.iterrows():
        any_nan = row[climate_vars].isna().any()
        flag = row['method_missing_flag']
        if any_nan and flag != 1:
            errors.append(f"Missing flag logic failed at row {idx}: variables are missing but flag is not 1")
        elif not any_nan and flag == 1:
            errors.append(f"Missing flag logic failed at row {idx}: variables are populated but flag is 1")
            
    # 10. Validate precipitation is never negative
    if (df['climate_precip_mm'] < 0).any():
        neg_count = (df['climate_precip_mm'] < 0).sum()
        errors.append(f"Precipitation check failed: found {neg_count} negative precipitation values")
        
    # 11. Validate temperature ordering
    valid_temp_mask = df[climate_vars].notna().all(axis=1)
    temp_check_df = df[valid_temp_mask]
    invalid_temp_order = temp_check_df[
        (temp_check_df['climate_tmin_c'] > temp_check_df['climate_tmean_c']) | 
        (temp_check_df['climate_tmean_c'] > temp_check_df['climate_tmax_c'])
    ]
    if len(invalid_temp_order) > 0:
        errors.append(f"Temperature ordering check failed: found {len(invalid_temp_order)} rows where tmin <= tmean <= tmax was violated")
        
    # Stop if validation failed
    if len(errors) > 0:
        print("Validation errors occurred:")
        for err in errors:
            print(f"  [ERROR] {err}")
        raise ValueError("CSV failed validation checks.")
        
    print("All validation checks passed!")
    
    # --- Analysis Part ---
    method_order = ['polygon_mean', 'centroid_sample', 'buffer_25km', 'buffer_50km', 'buffer_100km']
    
    results = []
    # Group by csd_uid, CSD name, province_u, year, month
    grouped = df.groupby(['csd_uid', 'csd_name', 'province_u', 'year', 'month'])
    
    for (csd_uid, csd_name, province_u, year, month), group in grouped:
        resolved_method = "unresolved"
        climate_vals = {}
        
        # Iterate in preferred order
        for m in method_order:
            sub = group[group['method_name'] == m]
            if not sub.empty:
                if sub['method_missing_flag'].iloc[0] == 0:
                    resolved_method = m
                    climate_vals = {
                        'climate_tmean_c': sub['climate_tmean_c'].iloc[0],
                        'climate_tmin_c': sub['climate_tmin_c'].iloc[0],
                        'climate_tmax_c': sub['climate_tmax_c'].iloc[0],
                        'climate_precip_mm': sub['climate_precip_mm'].iloc[0]
                    }
                    break
        
        results.append({
            'csd_uid': csd_uid,
            'csd_name': csd_name,
            'province_u': province_u,
            'year': year,
            'month': month,
            'first_successful_method': resolved_method,
            **climate_vals
        })
        
    summary_df = pd.DataFrame(results)
    
    # Print resolution summary counts
    print("\n================== Resolution Summary ==================")
    print("Counts by first successful method:")
    method_counts = summary_df['first_successful_method'].value_counts()
    for m in method_order + ['unresolved']:
        cnt = method_counts.get(m, 0)
        print(f"  - {m}: {cnt} CSD-months")
        
    unresolved_df = summary_df[summary_df['first_successful_method'] == 'unresolved']
    if len(unresolved_df) > 0:
        print("\nUnresolved CSD-months after 100 km buffer:")
        for _, row in unresolved_df.iterrows():
            print(f"  - CSD {row['csd_uid']} ({row['csd_name']}), Month {row['month']}")
    else:
        print("\nAll CSD-months successfully resolved at or before 100 km buffer.")
        
    # Control CSD comparison summary
    # Control CSD IDs are: 1001101, 3501005, 4801003
    control_ids = {'1001101', '3501005', '4801003'}
    control_df = df[df['csd_uid'].isin(control_ids)]
    
    if len(control_df) > 0:
        print("\n================ Control CSD Deviation Summary ================")
        print("Comparing fallback methods against polygon_mean:")
        for (csd_uid, month), grp in control_df.groupby(['csd_uid', 'month']):
            csd_name = grp['csd_name'].iloc[0]
            poly_row = grp[grp['method_name'] == 'polygon_mean']
            if poly_row.empty or poly_row['method_missing_flag'].iloc[0] == 1:
                continue
                
            p_tmean = poly_row['climate_tmean_c'].iloc[0]
            p_precip = poly_row['climate_precip_mm'].iloc[0]
            
            print(f"\nCSD {csd_uid} ({csd_name}) Month {month} (Polygon tmean={p_tmean:.2f}C, precip={p_precip:.2f}mm):")
            for m in ['centroid_sample', 'buffer_25km', 'buffer_50km', 'buffer_100km']:
                m_row = grp[grp['method_name'] == m]
                if not m_row.empty and m_row['method_missing_flag'].iloc[0] == 0:
                    t_diff = m_row['climate_tmean_c'].iloc[0] - p_tmean
                    p_diff = m_row['climate_precip_mm'].iloc[0] - p_precip
                    print(f"  - {m:16s}: tmean diff = {t_diff:+5.2f}C, precip diff = {p_diff:+6.2f}mm")
        print("===============================================================\n")
        
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"Summary saved to: {output_path}")

if __name__ == '__main__':
    default_input = "data/raw/gee_exports/era5_fallback_smoke_test.csv"
    default_output = "data/processed/features/era5_fallback_smoke_test_summary.csv"
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    try:
        analyze_fallback_smoke_test(input_file, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
