#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add leakage-safe lag/history outbreak features.

Input:  data/processed/modeling/model_table_v1_environment_train_ready.csv
Output: data/processed/modeling/model_table_v1_environment_with_lags.csv

Features added:
- same_csd_outbreak_current_month
- same_csd_outbreak_prev_1_month
- same_csd_outbreak_count_past_3_months
- same_csd_outbreak_count_past_6_months
- same_csd_outbreak_count_past_12_months
- province_outbreak_count_current_month
- province_outbreak_count_past_3_months
"""

import sys
import pandas as pd

def main():
    input_path = "data/processed/modeling/model_table_v1_environment_train_ready.csv"
    output_path = "data/processed/modeling/model_table_v1_environment_with_lags.csv"

    # Load data
    df = pd.read_csv(input_path, dtype=str)
    # Ensure numeric columns are proper types
    numeric_cols = ["year", "month", "outbreak_next_month"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create a proper date column for sorting
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
    # Sort by csd_uid and date
    df = df.sort_values(["csd_uid", "date"]).reset_index(drop=True)

    # Reconstruct outbreak_current_month if missing
    if "outbreak_current_month" not in df.columns:
        # shift outbreak_next_month within each CSD by one month backward
        df["outbreak_current_month"] = (
            df.groupby("csd_uid")["outbreak_next_month"].shift(1).fillna(0).astype(int)
        )
    else:
        df["outbreak_current_month"] = pd.to_numeric(df["outbreak_current_month"], errors="coerce").fillna(0).astype(int)

    # Helper series for per‑CSD rolling sums
    csd_grp = df.groupby("csd_uid")
    # same_csd_outbreak_current_month (already present)
    df["same_csd_outbreak_current_month"] = df["outbreak_current_month"]
    # same_csd_outbreak_prev_1_month
    df["same_csd_outbreak_prev_1_month"] = csd_grp["outbreak_current_month"].shift(1).fillna(0).astype(int)
    # Rolling counts for past 3, 6, 12 months (including current month)
    df["same_csd_outbreak_count_past_3_months"] = csd_grp["outbreak_current_month"].transform(lambda s: s.rolling(window=3, min_periods=1).sum()).astype(int)
    df["same_csd_outbreak_count_past_6_months"] = csd_grp["outbreak_current_month"].transform(lambda s: s.rolling(window=6, min_periods=1).sum()).astype(int)
    df["same_csd_outbreak_count_past_12_months"] = csd_grp["outbreak_current_month"].transform(lambda s: s.rolling(window=12, min_periods=1).sum()).astype(int)

    # Province level aggregates
    prov_current = df.groupby(["province_code", "date"], sort=False)["outbreak_current_month"].sum().reset_index(name="province_outbreak_count_current_month")
    df = df.merge(prov_current, on=["province_code", "date"], how="left")
    # rolling sum for past 3 months at province level using transform
    df["province_outbreak_count_past_3_months"] = df.groupby("province_code")["outbreak_current_month"].transform(lambda s: s.rolling(window=3, min_periods=1).sum()).astype(int)

    # QA checks
    print("=== QA SUMMARY ===")
    print(f"Row count unchanged: {len(df)}")
    dup = df.duplicated(subset=["csd_uid", "year", "month"]).sum()
    print(f"Duplicate csd_uid/year/month rows: {dup}")
    lag_cols = [
        "same_csd_outbreak_current_month",
        "same_csd_outbreak_prev_1_month",
        "same_csd_outbreak_count_past_3_months",
        "same_csd_outbreak_count_past_6_months",
        "same_csd_outbreak_count_past_12_months",
        "province_outbreak_count_current_month",
        "province_outbreak_count_past_3_months",
    ]
    nan_counts = df[lag_cols].isna().sum()
    print("NaN counts in lag columns:")
    print(nan_counts.to_string())
    negative_counts = (df[lag_cols] < 0).sum()
    print("Negative values in lag columns:")
    print(negative_counts.to_string())
    print("Summary statistics (mean, std) for lag columns:")
    print(df[lag_cols].agg(["mean", "std"]).to_string())
    # Correlation with target
    if "outbreak_next_month" in df.columns:
        corrs = df[lag_cols].apply(lambda s: s.corr(df["outbreak_next_month"]))
        print("Correlation of lag columns with outbreak_next_month:")
        print(corrs.to_string())

    # Save output CSV (exclude the temporary date column)
    df.drop(columns=["date"], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Saved enriched table to {output_path}")

if __name__ == "__main__":
    main()
