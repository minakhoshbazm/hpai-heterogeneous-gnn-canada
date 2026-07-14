import os
import sys
import pandas as pd

def inspect_missingness(input_path):
    print(f"Loading modeling table from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path, dtype={'csd_uid': str})
    total_rows = len(df)
    
    # Identify columns with missing values
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    print("\nColumns with missing values:")
    if len(cols_with_missing) == 0:
        print("  - None! 0 NaNs across all columns.")
    else:
        for col, count in cols_with_missing.items():
            pct = count / total_rows * 100
            print(f"  - {col}: {count} NaNs ({pct:.2f}%)")
            
    # Group columns by source
    poultry_agriculture_cols = [c for c in df.columns if c.startswith('ccs_')]
    climate_cols = [c for c in df.columns if c.startswith('climate_')]
    dw_cols = [c for c in df.columns if c.startswith('dw_')]
    metadata_target_cols = ['csd_uid', 'csd_name', 'province_code', 'province_name', 'year', 'month', 'outbreak_next_month']
    
    other_cols = [c for c in df.columns if c not in poultry_agriculture_cols + climate_cols + dw_cols + metadata_target_cols]
    
    # Calculate missingness by group
    print("\nMissingness summary by group:")
    groups = {
        "Poultry/Agriculture Static Features": poultry_agriculture_cols,
        "Road/Static Geography Features": other_cols,
        "Climate Features": climate_cols,
        "Dynamic World Features": dw_cols,
        "Metadata/Target Features": metadata_target_cols
    }
    
    only_agriculture = True
    for g_name, g_cols in groups.items():
        g_missing = df[g_cols].isna().sum().sum()
        g_pct = (g_missing / (total_rows * len(g_cols))) * 100 if len(g_cols) > 0 else 0.0
        print(f"  - {g_name} (columns: {len(g_cols)}): {g_missing} NaNs ({g_pct:.2f}% of group values)")
        
        # Check if missingness occurs outside static agriculture columns
        if g_name != "Poultry/Agriculture Static Features" and g_missing > 0:
            only_agriculture = False
            
    print(f"\nIs missingness strictly restricted to static poultry/agriculture columns? {only_agriculture}")
    
    # Print target positive count and class imbalance
    if 'outbreak_next_month' in df.columns:
        pos_count = int(df['outbreak_next_month'].sum())
        pos_pct = pos_count / total_rows * 100
        print(f"\nTarget Variable Diagnostics (outbreak_next_month):")
        print(f"  - Positive Cases: {pos_count} ({pos_pct:.4f}%)")
        print(f"  - Negative Cases: {total_rows - pos_count} ({100 - pos_pct:.4f}%)")
        
    # Recommendations
    print("\nRecommended Fill Strategy:")
    print("  1. Agricultural Production & Count Features (e.g. ccs_*_production_kg, ccs_*_farms_reporting):")
    print("     - Recommend filling NaNs with 0. CSDs without agricultural census records typically have 0 poultry production.")
    print("  2. Status/Code/Identifier Columns (e.g. ccs_uid):")
    print("     - Recommend filling with 'missing' as a string token to treat as a distinct categorical level if used.")
    print("  3. Target/Metadata (e.g. outbreak_next_month, csd_uid, year, month):")
    print("     - Do not fill. These columns must remain clean and intact.")
    print("=========================================================\n")

if __name__ == '__main__':
    default_input = 'data/processed/modeling/model_table_v1_environment_only.csv'
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    try:
        inspect_missingness(input_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
