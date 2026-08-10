#!/usr/bin/env python3
"""
scripts/audit_wildbird_cfia_crosswalk.py

Audit wild-bird HPAI raw file under data/raw/cfia and test mapping to modeling table geography (CCS level).
"""

import os
import glob
import pandas as pd
import numpy as np
import re
from difflib import get_close_matches

CFIA_DIR = 'data/raw/cfia'
MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
OUT_CROSSWALK_PATH = 'data/processed/geography/wildbird_ccs_name_to_ccs_uid_crosswalk_draft.csv'

PROVINCE_MAP = {
    'AB': 'Alberta', 'ALBERTA': 'Alberta',
    'BC': 'British Columbia', 'BRITISH COLUMBIA': 'British Columbia',
    'MB': 'Manitoba', 'MANITOBA': 'Manitoba',
    'NB': 'New Brunswick', 'NEW BRUNSWICK': 'New Brunswick',
    'NL': 'Newfoundland and Labrador', 'NEWFOUNDLAND AND LABRADOR': 'Newfoundland and Labrador',
    'NS': 'Nova Scotia', 'NOVA SCOTIA': 'Nova Scotia',
    'ON': 'Ontario', 'ONTARIO': 'Ontario',
    'PE': 'Prince Edward Island', 'PEI': 'Prince Edward Island', 'PRINCE EDWARD ISLAND': 'Prince Edward Island',
    'QC': 'Quebec', 'QUÉBEC': 'Quebec', 'QUEBEC': 'Quebec',
    'SK': 'Saskatchewan', 'SASKATCHEWAN': 'Saskatchewan',
    'YT': 'Yukon', 'YUKON': 'Yukon',
    'NT': 'Northwest Territories', 'NORTHWEST TERRITORIES': 'Northwest Territories',
    'NU': 'Nunavut', 'NUNAVUT': 'Nunavut'
}

def clean_string(s):
    if pd.isna(s):
        return ''
    s = str(s).upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def main():
    print("==========================================================================================")
    print("      CFIA WILD-BIRD HPAI FILE & CCS GEOGRAPHY AUDIT")
    print("==========================================================================================")

    # 1. Locate wild-bird file under data/raw/cfia
    files = glob.glob(os.path.join(CFIA_DIR, '*'))
    print(f"\nFiles found in {CFIA_DIR}:")
    for f in files:
        print(f" - {os.path.basename(f)} ({os.path.getsize(f):,} bytes)")

    wildbird_files = [f for f in files if 'wild' in os.path.basename(f).lower() or 'bird' in os.path.basename(f).lower()]
    if not wildbird_files:
        print("No file explicitly containing 'wild' or 'bird' found. Checking all CSV/Excel files...")
        wildbird_files = [f for f in files if f.endswith('.csv') or f.endswith('.xlsx') or f.endswith('.xls')]

    if not wildbird_files:
        raise FileNotFoundError("No CFIA data files found under data/raw/cfia")

    target_file = wildbird_files[0]
    print(f"\nSelected Wild-Bird Target File: {target_file}")

    # Load target file
    if target_file.endswith('.xlsx') or target_file.endswith('.xls'):
        df_wb = pd.read_excel(target_file)
    else:
        df_wb = pd.read_csv(target_file)

    print(f"\nFile Name: {os.path.basename(target_file)}")
    print(f"File Shape: {df_wb.shape[0]:,} rows x {df_wb.shape[1]} columns")
    print(f"\nColumns ({len(df_wb.columns)}):")
    for c in df_wb.columns:
        print(f" - {c}")

    print("\nFirst 10 Rows:")
    print(df_wb.head(10).to_string())

    # 3. Identify Key Columns
    print("\n------------------------------------------------------------------------------------------")
    print("  KEY COLUMN IDENTIFICATION")
    print("------------------------------------------------------------------------------------------")
    date_col = next((c for c in df_wb.columns if 'date' in c.lower() or 'time' in c.lower()), None)
    ccs_col = next((c for c in df_wb.columns if 'ccs' in c.lower() or 'census' in c.lower() or 'subdivision' in c.lower() or 'county' in c.lower() or 'region' in c.lower() or 'location' in c.lower() or 'name' in c.lower()), None)
    prov_col = next((c for c in df_wb.columns if 'prov' in c.lower() or 'state' in c.lower()), None)
    species_col = next((c for c in df_wb.columns if 'spec' in c.lower() or 'bird' in c.lower() or 'host' in c.lower()), None)
    outcome_col = next((c for c in df_wb.columns if 'outbreak' in c.lower() or 'status' in c.lower() or 'result' in c.lower() or 'case' in c.lower() or 'det' in c.lower()), None)

    print(f"Identified Date Column:    {date_col}")
    print(f"Identified CCS/Loc Column: {ccs_col}")
    print(f"Identified Prov Column:    {prov_col}")
    print(f"Identified Species Column: {species_col}")
    print(f"Identified Outcome Column: {outcome_col}")

    # 4. Check Modeling Table for CCS Fields
    print("\n------------------------------------------------------------------------------------------")
    print("  CHECK MODELING TABLE CCS FIELDS")
    print("------------------------------------------------------------------------------------------")
    df_model = pd.read_csv(MODEL_TABLE_PATH, nrows=100)
    print(f"Modeling Table Columns matching 'ccs' or 'csd':")
    ccs_fields_in_model = [c for c in df_model.columns if 'ccs' in c.lower() or 'csd' in c.lower() or 'prov' in c.lower()]
    for c in ccs_fields_in_model:
        print(f" - {c}")

    # Check if there is a reference CCS lookup (e.g. from census shapefile or dataset)
    # Let's inspect unique ccs_uid and csd_name/province in model_table
    df_model_full = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str, 'ccs_uid': str, 'province_code': str})
    print(f"\nUnique ccs_uid in model table: {df_model_full['ccs_uid'].nunique():,}")
    print(f"Unique csd_uid in model table: {df_model_full['csd_uid'].nunique():,}")

    # Let's check if ccs_name or csd_name exists in df_model_full
    has_csd_name = 'csd_name' in df_model_full.columns
    has_province_name = 'province_name' in df_model_full.columns

    # Build reference CCS dataset
    # Group by ccs_uid (or csd_name/csd_uid) to get CCS/CSD reference names
    ref_records = []
    if 'ccs_uid' in df_model_full.columns:
        for ccs_uid, group in df_model_full.groupby('ccs_uid'):
            p_code = group['province_code'].iloc[0] if 'province_code' in group.columns else ''
            p_name = group['province_name'].iloc[0] if 'province_name' in group.columns else ''
            c_name = group['csd_name'].iloc[0] if 'csd_name' in group.columns else ''
            ref_records.append({
                'ccs_uid': str(ccs_uid),
                'province_code': str(p_code),
                'province_name': str(p_name),
                'csd_name': str(c_name),
                'clean_name': clean_string(c_name),
                'clean_prov': clean_string(p_name)
            })
    ref_df = pd.DataFrame(ref_records)
    print(f"Constructed Reference Geography DataFrame with {len(ref_df):,} entries.")

    # 5. Build Candidate Crosswalk
    print("\n------------------------------------------------------------------------------------------")
    print("  CROSSWALK MATCHING (EXACT & FUZZY)")
    print("------------------------------------------------------------------------------------------")

    if ccs_col is None:
        print("Warning: CCS location column could not be automatically determined.")
        return

    wb_records = df_wb[[prov_col, ccs_col]].drop_duplicates().dropna() if prov_col else df_wb[[ccs_col]].drop_duplicates().dropna()

    matched_exact = 0
    matched_fuzzy = 0
    unmatched = 0

    crosswalk_rows = []

    for idx, row in wb_records.iterrows():
        raw_ccs = str(row[ccs_col])
        raw_prov = str(row[prov_col]) if prov_col and not pd.isna(row[prov_col]) else ''
        clean_ccs = clean_string(raw_ccs)
        clean_p = clean_string(raw_prov)

        # Attempt Exact Match
        exact = ref_df[(ref_df['clean_name'] == clean_ccs) & ((ref_df['clean_prov'] == clean_p) | (clean_p == ''))]
        if len(exact) == 0:
            # Try matching clean_ccs anywhere in csd_name
            exact = ref_df[ref_df['clean_name'].str.contains(clean_ccs, regex=False) & ((ref_df['clean_prov'] == clean_p) | (clean_p == ''))]

        if len(exact) > 0:
            matched_exact += 1
            best_match = exact.iloc[0]
            crosswalk_rows.append({
                'raw_province': raw_prov,
                'raw_ccs_name': raw_ccs,
                'matched_ccs_uid': best_match['ccs_uid'],
                'matched_csd_name': best_match['csd_name'],
                'matched_province_name': best_match['province_name'],
                'match_type': 'EXACT',
                'similarity_score': 1.0
            })
        else:
            # Attempt Fuzzy Match
            prov_candidates = ref_df[ref_df['clean_prov'] == clean_p] if clean_p != '' else ref_df
            cand_names = prov_candidates['clean_name'].tolist()
            close = get_close_matches(clean_ccs, cand_names, n=1, cutoff=0.7)

            if close:
                matched_fuzzy += 1
                best_match = prov_candidates[prov_candidates['clean_name'] == close[0]].iloc[0]
                crosswalk_rows.append({
                    'raw_province': raw_prov,
                    'raw_ccs_name': raw_ccs,
                    'matched_ccs_uid': best_match['ccs_uid'],
                    'matched_csd_name': best_match['csd_name'],
                    'matched_province_name': best_match['province_name'],
                    'match_type': 'FUZZY',
                    'similarity_score': 0.8
                })
            else:
                unmatched += 1
                crosswalk_rows.append({
                    'raw_province': raw_prov,
                    'raw_ccs_name': raw_ccs,
                    'matched_ccs_uid': np.nan,
                    'matched_csd_name': np.nan,
                    'matched_province_name': np.nan,
                    'match_type': 'UNMATCHED',
                    'similarity_score': 0.0
                })

    df_crosswalk = pd.DataFrame(crosswalk_rows)

    os.makedirs(os.path.dirname(OUT_CROSSWALK_PATH), exist_ok=True)
    df_crosswalk.to_csv(OUT_CROSSWALK_PATH, index=False)

    print("\n==========================================================================================")
    print("      CROSSWALK AUDIT SUMMARY REPORT")
    print("==========================================================================================")
    print(f"Total Unique Wild-Bird CCS Names: {len(wb_records):,}")
    print(f" - Matched Exactly: {matched_exact:,} ({matched_exact / len(wb_records):.1%})")
    print(f" - Matched Fuzzy:   {matched_fuzzy:,} ({matched_fuzzy / len(wb_records):.1%})")
    print(f" - Unmatched:       {unmatched:,} ({unmatched / len(wb_records):.1%})")
    print("==========================================================================================")

    if unmatched > 0:
        print("\nExamples of Unmatched Names:")
        print(df_crosswalk[df_crosswalk['match_type'] == 'UNMATCHED'].head(10)[['raw_province', 'raw_ccs_name']].to_string(index=False))

    print(f"\nSaved draft crosswalk to: {OUT_CROSSWALK_PATH}")

if __name__ == '__main__':
    main()
