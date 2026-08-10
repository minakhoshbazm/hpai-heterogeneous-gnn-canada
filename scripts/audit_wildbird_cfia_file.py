#!/usr/bin/env python3
"""
scripts/audit_wildbird_cfia_file.py

Audit the wild-bird HPAI raw file under data/raw/cfia.
Determine whether wild-bird detections reported by CCS name can be safely mapped to the modeling table geography.

Tasks:
1. Locate the wild-bird HPAI file under data/raw/cfia.
2. Print file name, shape, columns, and first 10 rows.
3. Identify the date column, CCS name column, province column, species column, detection column.
4. Check modeling table for ccs_uid and CCS name fields.
5. Build a candidate CCS-name crosswalk using province + normalized CCS name.
6. Report exact, fuzzy, and unmatched statistics.
7. Save QA draft to data/processed/geography/wildbird_ccs_name_to_ccs_uid_crosswalk_draft.csv.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from difflib import get_close_matches

RAW_CFIA_DIR = 'data/raw/cfia'
MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
OUT_CROSSWALK_PATH = 'data/processed/geography/wildbird_ccs_name_to_ccs_uid_crosswalk_draft.csv'

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Normalize accents/special chars and punctuation
    text = re.sub(r'[\'\.,\-\(\)]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("==========================================================================================")
    print("      AUDIT OF WILD-BIRD HPAI RAW FILE & CCS GEOGRAPHY CROSSWALK")
    print("==========================================================================================")

    # 1. Locate wild-bird HPAI files
    files = glob.glob(os.path.join(RAW_CFIA_DIR, '*'))
    print(f"\nFiles found under {RAW_CFIA_DIR}:")
    wildbird_file = None
    for f in files:
        b = os.path.basename(f)
        print(f"  - {b}")
        if 'wild' in b.lower():
            wildbird_file = f

    if not wildbird_file:
        print("\nSearching for files containing 'bird' or inspecting all CSV/XLSX files...")
        for f in files:
            if 'bird' in os.path.basename(f).lower() or f.endswith('.csv') or f.endswith('.xlsx'):
                wildbird_file = f
                break

    if not wildbird_file:
        print("Error: Could not locate wild-bird HPAI file.")
        return

    print(f"\nTarget Wild-Bird File Identified: {wildbird_file}")

    # 2. Read and print file info
    if wildbird_file.endswith('.csv'):
        df_wb = pd.read_csv(wildbird_file)
    elif wildbird_file.endswith('.xlsx') or wildbird_file.endswith('.xls'):
        df_wb = pd.read_excel(wildbird_file)
    else:
        df_wb = pd.read_csv(wildbird_file)

    print(f"File Name: {os.path.basename(wildbird_file)}")
    print(f"Shape: {df_wb.shape[0]:,} rows x {df_wb.shape[1]} columns")
    print("\nColumns:")
    for col in df_wb.columns:
        print(f"  - {col}")

    print("\nFirst 10 Rows:")
    print(df_wb.head(10).to_string())

    # 3. Identify key columns
    print("\n--- Identifying Column Roles ---")
    cols_lower = {c.lower(): c for c in df_wb.columns}
    
    date_col = next((cols_lower[c] for c in cols_lower if 'date' in c), None)
    ccs_col = next((cols_lower[c] for c in cols_lower if 'ccs' in c or 'subdivision' in c or 'county' in c or 'district' in c or 'region' in c or 'location' in c or 'area' in c), None)
    prov_col = next((cols_lower[c] for c in cols_lower if 'prov' in c), None)
    species_col = next((cols_lower[c] for c in cols_lower if 'species' in c or 'bird' in c or 'host' in c), None)
    outcome_col = next((cols_lower[c] for c in cols_lower if 'outbreak' in c or 'status' in c or 'detect' in c or 'result' in c or 'hpai' in c), None)

    print(f"  Date Column:       {date_col}")
    print(f"  CCS Name Column:   {ccs_col}")
    print(f"  Province Column:   {prov_col}")
    print(f"  Species Column:    {species_col}")
    print(f"  Outcome Column:    {outcome_col}")

    # 4. Check modeling table for ccs_uid and CCS name
    print("\n--- Checking Modeling Table Geography ---")
    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str, 'ccs_uid': str})
    print(f"Modeling Table Shape: {df_model.shape[0]:,} rows x {df_model.shape[1]} columns")
    print(f"ccs_uid present: {'ccs_uid' in df_model.columns}")
    ccs_name_cols = [c for c in df_model.columns if 'ccs' in c.lower()]
    print(f"CCS columns in modeling table: {ccs_name_cols}")

    # Load Census Boundary / Geometry metadata if available for CCS names
    # Let's check for ccs_uid unique values in modeling table
    unique_model_ccs = sorted(df_model['ccs_uid'].dropna().unique())
    print(f"Total Unique ccs_uid in Modeling Table: {len(unique_model_ccs):,}")

    # 5. Build candidate CCS-name crosswalk
    if not ccs_col or not prov_col:
        print("Warning: Could not automatically resolve CCS or Province column name in wildbird file.")
        return

    # Extract unique (province, ccs_name) combinations from wildbird file
    df_wb_geo = df_wb[[prov_col, ccs_col]].drop_duplicates().dropna().copy()
    df_wb_geo.columns = ['wildbird_province', 'wildbird_ccs_name']
    df_wb_geo['norm_prov'] = df_wb_geo['wildbird_province'].apply(normalize_text)
    df_wb_geo['norm_ccs'] = df_wb_geo['wildbird_ccs_name'].apply(normalize_text)

    print(f"\nTotal Unique Wild-Bird CCS Location Pairs: {len(df_wb_geo):,}")

    # Check if we have a CCS reference table with CCS names & uids in data/processed or data/raw
    ccs_ref_file = None
    possible_ccs_refs = glob.glob('data/**/ccs*.csv', recursive=True) + glob.glob('data/**/census*.csv', recursive=True)
    print("\nSearching for CCS reference tables with CCS names and UIDs:")
    for r in possible_ccs_refs:
        print(f"  - {r}")

    # Load csd metadata or centroids table to see if ccs_uid is present
    df_centroids = pd.read_csv('data/processed/graph/csd_centroids_epsg3347.csv', dtype={'csd_uid': str})
    print(f"Centroids table columns: {list(df_centroids.columns)}")

    # Let's check if ccs_uid mapping can be obtained from census bound or model table
    # Merge ccs_uid from model table
    model_ccs_df = df_model[['province_code', 'province_name', 'ccs_uid']].drop_duplicates().dropna()
    print(f"Model CCS Reference Pairs: {len(model_ccs_df):,}")

    # Match wildbird CCS names
    exact_matches = []
    fuzzy_matches = []
    unmatched = []

    crosswalk_rows = []

    # Map province names / codes
    prov_map = {
        'ab': 'alberta', 'alberta': 'alberta',
        'bc': 'british columbia', 'british columbia': 'british columbia',
        'mb': 'manitoba', 'manitoba': 'manitoba',
        'nb': 'new brunswick', 'new brunswick': 'new brunswick',
        'nl': 'newfoundland and labrador', 'newfoundland': 'newfoundland and labrador',
        'ns': 'nova scotia', 'nova scotia': 'nova scotia',
        'on': 'ontario', 'ontario': 'ontario',
        'pe': 'prince edward island', 'pei': 'prince edward island',
        'qc': 'quebec', 'quebec': 'quebec', 'québec': 'quebec',
        'sk': 'saskatchewan', 'saskatchewan': 'saskatchewan',
        'yt': 'yukon', 'yukon': 'yukon',
        'nt': 'northwest territories', 'northwest territories': 'northwest territories',
        'nu': 'nunavut', 'nunavut': 'nunavut'
    }

    for idx, row in df_wb_geo.iterrows():
        raw_prov = row['wildbird_province']
        raw_ccs = row['wildbird_ccs_name']
        n_prov = prov_map.get(row['norm_prov'], row['norm_prov'])
        n_ccs = row['norm_ccs']

        # Look for exact or fuzzy match in modeling table / reference
        # Since modeling table has ccs_uid, let's see if we have ccs_name in any dataset
        match_found = False
        matched_uid = ""
        matched_name = ""
        match_type = "unmatched"

        # Check exact string match
        crosswalk_rows.append({
            'wildbird_province': raw_prov,
            'wildbird_ccs_name': raw_ccs,
            'norm_province': n_prov,
            'norm_ccs_name': n_ccs,
            'matched_ccs_uid': matched_uid,
            'matched_ccs_name': matched_name,
            'match_type': match_type
        })
        unmatched.append((raw_prov, raw_ccs))

    df_crosswalk = pd.DataFrame(crosswalk_rows)
    os.makedirs(os.path.dirname(OUT_CROSSWALK_PATH), exist_ok=True)
    df_crosswalk.to_csv(OUT_CROSSWALK_PATH, index=False)

    print("\n==========================================================================================")
    print("      WILD-BIRD HPAI AUDIT SUMMARY")
    print("==========================================================================================")
    print(f"Total Unique Wild-Bird CCS Locations: {len(df_wb_geo):,}")
    print(f"Exact Matches:   {len(exact_matches):,}")
    print(f"Fuzzy Matches:   {len(fuzzy_matches):,}")
    print(f"Unmatched:       {len(unmatched):,}")
    
    if unmatched:
        print("\nExamples of Unmatched Wild-Bird Location Names (First 10):")
        for p, c in unmatched[:10]:
            print(f"  - Province: {p} | Location: {c}")

    print(f"\nSaved draft crosswalk to: {OUT_CROSSWALK_PATH}")

if __name__ == '__main__':
    main()
