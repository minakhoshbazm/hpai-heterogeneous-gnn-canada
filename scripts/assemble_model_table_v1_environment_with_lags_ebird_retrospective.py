"""
Assemble Retrospective eBird-Enhanced Modeling Table.

Input Files:
  1. data/processed/modeling/model_table_v1_environment_with_lags.csv
  2. data/processed/features/csd_monthly_ebird_status_2022_2025_retrospective.csv

Output File:
  data/processed/modeling/model_table_v1_environment_with_lags_ebird_retrospective.csv
"""

import os
import gc
import pandas as pd
import numpy as np

MODEL_TABLE_PATH = "data/processed/modeling/model_table_v1_environment_with_lags.csv"
EBIRD_RETRO_PATH = "data/processed/features/csd_monthly_ebird_status_2022_2025_retrospective.csv"
OUTPUT_PATH = "data/processed/modeling/model_table_v1_environment_with_lags_ebird_retrospective.csv"

ADDED_EBIRD_COLUMNS = [
    # Provenance metadata
    "ebird_status_product_year",
    "ebird_retrospective_only_flag",
    # Abundance features
    "mallard_relative_abundance",
    "american_black_duck_relative_abundance",
    "northern_pintail_relative_abundance",
    "green-winged_teal_relative_abundance",
    "blue-winged_teal_relative_abundance",
    "canvasback_relative_abundance",
    "redhead_relative_abundance",
    "snow_goose_relative_abundance",
    "canada_goose_relative_abundance",
    "ebird_duck_relative_abundance",
    "ebird_goose_relative_abundance",
    "ebird_selected_waterfowl_relative_abundance",
    # Missingness flags & counts
    "ebird_missing_flag",
    "ebird_missing_species_count",
    "ebird_all_species_missing_flag",
    "ebird_duck_missing_flag",
    "ebird_goose_missing_flag",
]

def main():
    print(f"Loading source modeling table from {MODEL_TABLE_PATH}...")
    df_mod = pd.read_csv(MODEL_TABLE_PATH)
    source_row_count = len(df_mod)
    print(f"Source modeling table shape: {df_mod.shape}")

    # Downcast floats in df_mod for memory efficiency during merge
    float_cols_mod = df_mod.select_dtypes(include=['float64']).columns
    df_mod[float_cols_mod] = df_mod[float_cols_mod].astype('float32')

    print(f"Loading retrospective eBird table from {EBIRD_RETRO_PATH}...")
    ebird_cols_to_load = ['csd_uid', 'year', 'month'] + ADDED_EBIRD_COLUMNS
    df_ebird = pd.read_csv(EBIRD_RETRO_PATH, usecols=ebird_cols_to_load)
    print(f"Retrospective eBird table shape: {df_ebird.shape}")

    float_cols_ebird = df_ebird.select_dtypes(include=['float64']).columns
    df_ebird[float_cols_ebird] = df_ebird[float_cols_ebird].astype('float32')

    # Verify keys
    mod_keys = set(zip(df_mod['csd_uid'], df_mod['year'], df_mod['month']))
    ebird_keys = set(zip(df_ebird['csd_uid'], df_ebird['year'], df_ebird['month']))
    unmatched_mod_keys = len(mod_keys - ebird_keys)
    unmatched_ebird_keys = len(ebird_keys - mod_keys)

    print(f"Unmatched source modeling keys count: {unmatched_mod_keys}")
    print(f"Unmatched retrospective eBird keys count: {unmatched_ebird_keys}")

    # Perform left join to preserve every row from source modeling table
    print("Performing left join on csd_uid, year, month...")
    df_merged = pd.merge(
        df_mod,
        df_ebird,
        on=['csd_uid', 'year', 'month'],
        how='left'
    )
    output_row_count = len(df_merged)

    # Free memory of df_ebird
    del df_ebird
    gc.collect()

    # ------------------ ASSERTS ------------------
    print("\nExecuting assertions...")

    # 1. Output row count equals source modeling table row count
    assert output_row_count == source_row_count, (
        f"Row count mismatch! Source: {source_row_count}, Output: {output_row_count}"
    )

    # 2. No duplicate csd_uid/year/month keys
    dup_count = df_merged.duplicated(subset=['csd_uid', 'year', 'month']).sum()
    assert dup_count == 0, f"Duplicate keys found: {dup_count}"

    # 3. Every source row matched exactly one eBird row (no missingness in added eBird cols)
    for col in ADDED_EBIRD_COLUMNS:
        nans = df_merged[col].isna().sum()
        assert nans == 0, f"eBird column {col} contains {nans} NaNs after join!"

    # 4. No numeric NaNs introduced across the entire merged table
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()
    total_numeric_nans = df_merged[numeric_cols].isna().sum().sum()
    assert total_numeric_nans == 0, f"Numeric NaNs found in output table: {total_numeric_nans}"

    # 5. outbreak_next_month values are unchanged
    assert (df_merged['outbreak_next_month'] == df_mod['outbreak_next_month']).all(), (
        "outbreak_next_month target values were altered!"
    )

    # 6. All eBird abundance and flag columns have expected ranges
    assert (df_merged['ebird_status_product_year'] == 2023).all()
    assert (df_merged['ebird_retrospective_only_flag'] == 1).all()
    assert (df_merged['ebird_missing_species_count'].between(0, 9)).all()
    assert df_merged['ebird_missing_flag'].isin([0, 1]).all()
    assert df_merged['ebird_all_species_missing_flag'].isin([0, 1]).all()
    assert df_merged['ebird_duck_missing_flag'].isin([0, 1]).all()
    assert df_merged['ebird_goose_missing_flag'].isin([0, 1]).all()

    abundance_cols = [c for c in ADDED_EBIRD_COLUMNS if 'abundance' in c]
    for col in abundance_cols:
        assert (df_merged[col] >= 0.0).all(), f"Negative values found in {col}"

    print("All assertions passed successfully!")

    # Save output table
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved retrospective eBird-enhanced modeling table to {OUTPUT_PATH}")

    # ------------------ PRINT REPORT ------------------
    print("\n================ REPORT ================")
    print(f"Source Modeling Table Row Count: {source_row_count}")
    print(f"Output Retrospective Table Row Count: {output_row_count}")
    print(f"Unmatched Source Modeling Keys: {unmatched_mod_keys}")
    print(f"Unmatched eBird Retrospective Keys: {unmatched_ebird_keys}")
    print(f"\nList of Added eBird Columns ({len(ADDED_EBIRD_COLUMNS)}):")
    for col in ADDED_EBIRD_COLUMNS:
        print(f"  - {col}")

    print("\nMissingness Counts in Added eBird Columns:")
    print(df_merged[ADDED_EBIRD_COLUMNS].isna().sum().to_string())

    print("\nFive Sample Rows:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_merged[['csd_uid', 'year', 'month', 'outbreak_next_month'] + ADDED_EBIRD_COLUMNS].head(5))

    print("\nCONFIRMATION:")
    print("This output dataset is strictly a retrospective sensitivity table and has NOT been merged into the primary forecasting modeling table.")

if __name__ == "__main__":
    main()
