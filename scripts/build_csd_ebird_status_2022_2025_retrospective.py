"""
Generate retrospective eBird Status feature dataset (2022-2025).

Purpose:
Creates an exploratory sensitivity dataset by repeating the 2023 monthly CSD eBird Status patterns
across years 2022, 2023, 2024, and 2025. This table is kept strictly separate from the primary modeling table.

Input:
  data/processed/features/csd_monthly_ebird_status_2023_9km.csv

Output:
  data/processed/features/csd_monthly_ebird_status_2022_2025_retrospective.csv
"""

import os
import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/features/csd_monthly_ebird_status_2023_9km.csv"
OUTPUT_PATH = "data/processed/features/csd_monthly_ebird_status_2022_2025_retrospective.csv"

DUCK_SPECIES_COLS = [
    "mallard_relative_abundance",
    "american_black_duck_relative_abundance",
    "northern_pintail_relative_abundance",
    "green-winged_teal_relative_abundance",
    "blue-winged_teal_relative_abundance",
    "canvasback_relative_abundance",
    "redhead_relative_abundance"
]

GOOSE_SPECIES_COLS = [
    "snow_goose_relative_abundance",
    "canada_goose_relative_abundance"
]

ALL_SPECIES_COLS = DUCK_SPECIES_COLS + GOOSE_SPECIES_COLS

def main():
    print(f"Loading raw 2023 eBird status dataset from {INPUT_PATH}...")
    df_raw = pd.read_csv(INPUT_PATH)
    print(f"Raw 2023 shape: {df_raw.shape}")

    # 1. Compute missingness flags on raw species data prior to filling
    print("Computing missingness counts and flags...")
    missing_species_count = df_raw[ALL_SPECIES_COLS].isna().sum(axis=1)
    all_species_missing_flag = (missing_species_count == len(ALL_SPECIES_COLS)).astype(int)
    duck_missing_flag = (df_raw[DUCK_SPECIES_COLS].isna().sum(axis=1) == len(DUCK_SPECIES_COLS)).astype(int)
    goose_missing_flag = (df_raw[GOOSE_SPECIES_COLS].isna().sum(axis=1) == len(GOOSE_SPECIES_COLS)).astype(int)

    # 2. Fill individual species NaNs with 0.0 placeholder
    df_processed = df_raw.copy()
    df_processed[ALL_SPECIES_COLS] = df_processed[ALL_SPECIES_COLS].fillna(0.0)

    # 3. Recompute aggregate group sums after 0.0 fill
    df_processed["ebird_duck_relative_abundance"] = df_processed[DUCK_SPECIES_COLS].sum(axis=1)
    df_processed["ebird_goose_relative_abundance"] = df_processed[GOOSE_SPECIES_COLS].sum(axis=1)
    df_processed["ebird_selected_waterfowl_relative_abundance"] = (
        df_processed["ebird_duck_relative_abundance"] + df_processed["ebird_goose_relative_abundance"]
    )

    # 4. Attach metadata and missingness flag columns
    df_processed["ebird_status_product_year"] = 2023
    df_processed["ebird_retrospective_only_flag"] = 1
    df_processed["ebird_missing_species_count"] = missing_species_count
    df_processed["ebird_all_species_missing_flag"] = all_species_missing_flag
    df_processed["ebird_duck_missing_flag"] = duck_missing_flag
    df_processed["ebird_goose_missing_flag"] = goose_missing_flag

    # 5. Repeat monthly CSD pattern across years 2022, 2023, 2024, 2025
    years = [2022, 2023, 2024, 2025]
    yearly_dfs = []
    for y in years:
        df_year = df_processed.copy()
        df_year["year"] = y
        yearly_dfs.append(df_year)

    df_retrospective = pd.concat(yearly_dfs, ignore_index=True)

    # Reorder columns logically
    first_cols = [
        "csd_uid",
        "year",
        "month",
        "ebird_status_product_year",
        "ebird_retrospective_only_flag",
        "ebird_missing_flag",
        "ebird_missing_species_count",
        "ebird_all_species_missing_flag",
        "ebird_duck_missing_flag",
        "ebird_goose_missing_flag"
    ]
    other_cols = [c for c in df_retrospective.columns if c not in first_cols]
    df_retrospective = df_retrospective[first_cols + other_cols]

    # Save retrospective table
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_retrospective.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved retrospective eBird table to {OUTPUT_PATH}")

    # 6. Comprehensive QA Checks
    print("\n================ QA RESULTS ================")
    total_rows = len(df_retrospective)
    expected_rows = 5054 * 12 * 4
    print(f"Total Rows: {total_rows} (Expected: {expected_rows})")

    yearly_counts = df_retrospective["year"].value_counts().to_dict()
    print(f"Yearly Row Counts: {yearly_counts}")

    unique_csds = df_retrospective["csd_uid"].nunique()
    print(f"Unique CSDs: {unique_csds}")

    dup_count = df_retrospective.duplicated(subset=["csd_uid", "year", "month"]).sum()
    print(f"Duplicate Key Count (csd_uid, year, month): {dup_count}")

    print("\nMissingness-flag Counts by Year:")
    flag_cols = [
        "ebird_missing_flag",
        "ebird_missing_species_count",
        "ebird_all_species_missing_flag",
        "ebird_duck_missing_flag",
        "ebird_goose_missing_flag"
    ]
    for col in flag_cols:
        print(f"--- {col} ---")
        print(df_retrospective.groupby(["year", col]).size().unstack(fill_value=0))

    numeric_cols = df_retrospective.select_dtypes(include=[np.number]).columns.tolist()
    nan_counts = df_retrospective[numeric_cols].isna().sum()
    total_nans = nan_counts.sum()
    print(f"\nTotal NaNs in Numeric Columns: {total_nans}")

    # Check 4-year pattern identity per CSD-month
    print("\nVerifying identical values across all 4 years per CSD-month...")
    pivot = df_retrospective.pivot(index=["csd_uid", "month"], columns="year", values="ebird_selected_waterfowl_relative_abundance")
    diff1 = (pivot[2022] != pivot[2023]).sum()
    diff2 = (pivot[2023] != pivot[2024]).sum()
    diff3 = (pivot[2024] != pivot[2025]).sum()
    print(f"Differences 2022 vs 2023: {diff1}")
    print(f"Differences 2023 vs 2024: {diff2}")
    print(f"Differences 2024 vs 2025: {diff3}")

    print("\nFive Sample Rows:")
    print(df_retrospective.head(5).to_string())

if __name__ == "__main__":
    main()
