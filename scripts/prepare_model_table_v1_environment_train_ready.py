import os
import sys
import pandas as pd
import numpy as np

def prepare_train_ready_table(input_path, output_path):
    print("Preparing Train-Ready V1 Environment-Only Modeling Table...")
    print("==========================================================")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path, dtype={'csd_uid': str})
    original_rows = len(df)
    
    # 2. Drop rows where target outbreak_next_month is missing
    df_clean = df.dropna(subset=['outbreak_next_month']).copy()
    final_rows = len(df_clean)
    rows_dropped = original_rows - final_rows
    
    # Ensure they are Dec 2025 rows
    dropped_df = df[df['outbreak_next_month'].isna()]
    dropped_months = dropped_df.groupby(['year', 'month']).size()
    print(f"Dropped target-missing rows: {rows_dropped}")
    for (y, m), count in dropped_months.items():
        print(f"  - Dropped {count} rows for Year {y}, Month {m}")
        
    # 3. Fill agricultural static NaNs with 0
    cols_to_fill = [
        'ccs_broilers_roasters_cornish_production_kg',
        'ccs_broilers_roasters_cornish_farms_reporting',
        'ccs_turkey_production_kg',
        'ccs_turkey_farms_reporting'
    ]
    for col in cols_to_fill:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
            
    # 4. Fill ccs_uid status column with 'missing' to treat as category if present
    if 'ccs_uid' in df_clean.columns:
        df_clean['ccs_uid'] = df_clean['ccs_uid'].fillna('missing')
        
    # 5. Validation Checks
    errors = []
    
    # No missing target
    if df_clean['outbreak_next_month'].isna().any():
        errors.append("Target 'outbreak_next_month' still contains NaNs")
        
    # No duplicate csd_uid/year/month
    dup_count = df_clean.duplicated(subset=['csd_uid', 'year', 'month']).sum()
    if dup_count > 0:
        errors.append(f"Duplicate CSD-month rows found: {dup_count}")
        
    # Target is binary 0/1
    unique_targets = df_clean['outbreak_next_month'].unique()
    for val in unique_targets:
        if val not in [0, 1, 0.0, 1.0]:
            errors.append(f"Target contains invalid non-binary value: {val}")
            
    # Identify numeric features for modeling
    dw_features = [c for c in df_clean.columns if c.startswith('dw_') and c not in ['dw_fill_method']]
    climate_features = [c for c in df_clean.columns if c.startswith('climate_') and c not in ['climate_extraction_method']]
    static_features = [c for c in df_clean.columns if c.startswith('ccs_') and c != 'ccs_uid']
    
    numeric_modeling_features = dw_features + climate_features + static_features
    
    # Check no NaNs remaining in numeric feature columns used for modeling
    for col in numeric_modeling_features:
        nan_count = df_clean[col].isna().sum()
        if nan_count > 0:
            errors.append(f"Feature column '{col}' contains {nan_count} NaNs")
            
    # Print QA Summary
    print("\n================== QA Summary ==================")
    print(f"Original row count: {original_rows}")
    print(f"Final row count: {final_rows}")
    print(f"Rows dropped: {rows_dropped} (Expected: 5054)")
    print(f"Unique CSD count: {df_clean['csd_uid'].nunique()}")
    print(f"Years represented: {sorted(df_clean['year'].unique())}")
    print(f"Months represented: {sorted(df_clean['month'].unique())}")
    
    pos_count = int(df_clean['outbreak_next_month'].sum())
    pos_rate = pos_count / final_rows * 100
    imbalance_ratio = (final_rows - pos_count) / pos_count if pos_count > 0 else float('inf')
    print(f"Target outbreak_next_month:")
    print(f"  - Positive cases: {pos_count}")
    print(f"  - Positive rate: {pos_rate:.4f}%")
    print(f"  - Class imbalance ratio (Neg/Pos): {imbalance_ratio:.2f}:1")
    
    print("\nFeatures columns verified (0 NaNs):")
    print(f"  - {len(numeric_modeling_features)} numeric features check passed.")
    print(f"Columns filled with 0: {cols_to_fill}")
    print("================================================")
    
    if len(errors) > 0:
        print("Validation errors occurred:")
        for err in errors:
            print(f"  [ERROR] {err}")
        raise ValueError("Train-ready table validation failed. Output was not saved.")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved train-ready modeling table to: {output_path}")

if __name__ == '__main__':
    default_input = 'data/processed/modeling/model_table_v1_environment_only.csv'
    default_output = 'data/processed/modeling/model_table_v1_environment_train_ready.csv'
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    try:
        prepare_train_ready_table(input_file, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
