#!/usr/bin/env python3
"""
scripts/audit_wildbird_cfia.py

Audit wild-bird HPAI raw file under data/raw/cfia.

Tasks:
1. Locate the wild-bird / CFIA HPAI file under data/raw/cfia.
2. Print file name, shape, columns, and first 10 rows.
3. Identify date column, CCS/location name column, province column, species column, and detection/outcome column.
4. Check whether the modeling table contains ccs_uid and any CCS/CSD name fields.
5. Build candidate CCS-name crosswalk using province + normalized location name.
6. Report:
   - number of unique location/CCS names
   - number matched exactly
   - number matched by fuzzy matching
   - number unmatched
   - examples of unmatched names
7. Save QA crosswalk draft:
   data/processed/geography/wildbird_ccs_name_to_ccs_uid_crosswalk_draft.csv
"""

import os
import glob
import re
import pandas as pd
import numpy as np

CFIA_RAW_PATH = 'data/raw/cfia/CFIA_outbreaks.csv'
MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
CSD_LOOKUP_PATH = 'data/processed/lookups/csd_to_ccs_lookup.csv'
CROSSWALK_OUT = 'data/processed/geography/wildbird_ccs_name_to_ccs_uid_crosswalk_draft.csv'

PROV_CODE_MAP = {
    'NL': '10', 'NEWFOUNDLAND AND LABRADOR': '10', 'NEWFOUNDLAND': '10',
    'PE': '11', 'PRINCE EDWARD ISLAND': '11',
    'NS': '12', 'NOVA SCOTIA': '12',
    'NB': '13', 'NEW BRUNSWICK': '13',
    'QC': '24', 'QUEBEC': '24', 'QUÉBEC': '24',
    'ON': '35', 'ONTARIO': '35',
    'MB': '46', 'MANITOBA': '46',
    'SK': '47', 'SASKATCHEWAN': '47',
    'AB': '48', 'ALBERTA': '48',
    'BC': '59', 'BRITISH COLUMBIA': '59',
    'YT': '60', 'YUKON': '60',
    'NT': '61', 'NORTHWEST TERRITORIES': '61',
    'NU': '62', 'NUNAVUT': '62'
}

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = text.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('ë', 'e')
    text = text.replace('à', 'a').replace('â', 'a').replace('ç', 'c')
    for char in [',', '.', '-', "'", '"', '/', '\\', '(', ')', '_']:
        text = text.replace(char, ' ')
    words = text.split()
    return " ".join(words)

def main():
    print("==========================================================================================")
    print("      AUDIT OF RAW CFIA / WILD-BIRD HPAI FILE UNDER DATA/RAW/CFIA")
    print("==========================================================================================")

    # 1. Locate file under data/raw/cfia
    print(f"Target raw file: {CFIA_RAW_PATH}")
    assert os.path.exists(CFIA_RAW_PATH), f"File not found: {CFIA_RAW_PATH}"

    # 2. Load and print metadata, shape, columns, first 10 rows
    df_raw = pd.read_csv(CFIA_RAW_PATH, encoding='latin1')
    file_name = os.path.basename(CFIA_RAW_PATH)

    print(f"\n1. FILE METADATA:")
    print(f"   File Name: {file_name}")
    print(f"   Shape:     {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    print(f"\n2. COLUMNS ({len(df_raw.columns)} total):")
    for i, col in enumerate(df_raw.columns):
        print(f"   [{i:2d}] {col}")

    print("\n3. FIRST 10 ROWS:")
    print(df_raw.head(10).to_string())

    # 3. Identify key columns
    print("\n==========================================================================================")
    print("      FIELD IDENTIFICATION")
    print("==========================================================================================")

    date_col = "Date of First Clinical Signs" if "Date of First Clinical Signs" in df_raw.columns else "CFIA Notification Date"
    ccs_name_col = "City" if "City" in df_raw.columns else "Control Zone"
    prov_col = "Province" if "Province" in df_raw.columns else None
    species_col = "Species - Production Type" if "Species - Production Type" in df_raw.columns else "WOAH Classification"
    outcome_col = "Repeat Detection" if "Repeat Detection" in df_raw.columns else "Pathogenicity"

    print(f"   - Date Column:                {date_col}")
    print(f"   - CCS / Location Name Column: {ccs_name_col}")
    print(f"   - Province Column:            {prov_col}")
    print(f"   - Species / Host Column:      {species_col}")
    print(f"   - Detection / Outcome Column: {outcome_col}")

    # 4. Check modeling table for ccs_uid and geography
    print("\n==========================================================================================")
    print("      CHECKING MODELING TABLE GEOGRAPHY")
    print("==========================================================================================")

    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype=str)
    print(f"Modeling Table: {df_model.shape[0]:,} rows x {df_model.shape[1]} columns")
    print(f"Contains 'ccs_uid': {'ccs_uid' in df_model.columns}")
    print(f"Contains 'csd_uid': {'csd_uid' in df_model.columns}")
    print(f"Contains 'csd_name': {'csd_name' in df_model.columns}")
    print(f"Contains 'province_code': {'province_code' in df_model.columns}")
    print(f"Contains 'province_name': {'province_name' in df_model.columns}")

    # Build reference lookup from csd_to_ccs_lookup.csv & model table
    df_lookup = pd.read_csv(CSD_LOOKUP_PATH, dtype=str)
    print(f"Loaded CSD-to-CCS lookup table: {len(df_lookup):,} entries")

    # Map (province_code, normalized_name) -> ccs_uid from both CCSNAME and CSDNAME
    ccs_geo_map = {}
    prov_refs_dict = {}

    for _, row in df_lookup.iterrows():
        p_code = str(row['PRUID_PRIDU'])
        ccs_uid = str(row['CCSUID_SRUIDU'])
        csd_name = str(row['CSDNAME_SDRNOM'])
        ccs_name = str(row['CCSNAME_SRUNOM'])

        norm_csd = normalize_text(csd_name)
        norm_ccs = normalize_text(ccs_name)

        if p_code and ccs_uid:
            if norm_csd:
                ccs_geo_map[(p_code, norm_csd)] = ccs_uid
                prov_refs_dict.setdefault(p_code, []).append((norm_csd, ccs_uid))
            if norm_ccs:
                ccs_geo_map[(p_code, norm_ccs)] = ccs_uid
                prov_refs_dict.setdefault(p_code, []).append((norm_ccs, ccs_uid))

    # Also add from df_model csd_name & ccs_uid
    for _, row in df_model[['province_code', 'csd_name', 'ccs_uid']].drop_duplicates().iterrows():
        p_code = str(row['province_code'])
        norm_c = normalize_text(row['csd_name'])
        ccs_uid = str(row['ccs_uid'])
        if p_code and norm_c and ccs_uid:
            ccs_geo_map[(p_code, norm_c)] = ccs_uid
            prov_refs_dict.setdefault(p_code, []).append((norm_c, ccs_uid))

    print(f"Total Reference (province_code, location_name) pairs: {len(ccs_geo_map):,}")

    # 5. Build candidate CCS-name crosswalk
    print("\n==========================================================================================")
    print("      BUILDING CCS-NAME CROSSWALK DRAFT")
    print("==========================================================================================")

    unique_locations = df_raw[[prov_col, ccs_name_col]].dropna().drop_duplicates()
    print(f"Total Unique Location-Province Pairs in Raw Data: {len(unique_locations):,}")

    exact_matches = 0
    fuzzy_matches = 0
    unmatched = 0
    unmatched_examples = []
    crosswalk_rows = []

    for _, row in unique_locations.iterrows():
        raw_prov = str(row[prov_col]).strip()
        raw_loc = str(row[ccs_name_col]).strip()
        
        prov_code = PROV_CODE_MAP.get(raw_prov.upper(), "")
        norm_loc = normalize_text(raw_loc)

        matched_ccs_uid = None
        match_type = "unmatched"

        # Step A: Exact match on (prov_code, normalized_name)
        if (prov_code, norm_loc) in ccs_geo_map:
            matched_ccs_uid = ccs_geo_map[(prov_code, norm_loc)]
            match_type = "exact"
            exact_matches += 1
        else:
            # Step B: Fuzzy match on substring within same province
            prov_refs = prov_refs_dict.get(prov_code, [])
            best_match_uid = None
            
            for ref_name, ref_uid in prov_refs:
                if norm_loc == ref_name or norm_loc in ref_name or ref_name in norm_loc:
                    best_match_uid = ref_uid
                    break
            
            if best_match_uid:
                matched_ccs_uid = best_match_uid
                match_type = "fuzzy"
                fuzzy_matches += 1
            else:
                match_type = "unmatched"
                unmatched += 1
                if len(unmatched_examples) < 15:
                    unmatched_examples.append(f"{raw_prov} (code {prov_code}) | {raw_loc}")

        crosswalk_rows.append({
            'raw_province': raw_prov,
            'province_code': prov_code,
            'raw_location_name': raw_loc,
            'normalized_location_name': norm_loc,
            'matched_ccs_uid': matched_ccs_uid,
            'match_type': match_type
        })

    df_crosswalk = pd.DataFrame(crosswalk_rows)

    # Save draft crosswalk
    os.makedirs(os.path.dirname(CROSSWALK_OUT), exist_ok=True)
    df_crosswalk.to_csv(CROSSWALK_OUT, index=False)

    print(f"\n==========================================================================================")
    print("      CROSSWALK AUDIT SUMMARY REPORT")
    print("==========================================================================================")
    print(f"Total Unique Location Names Evaluated: {len(df_crosswalk):,}")
    print(f"  1. Exact Matches:   {exact_matches:4d} ({exact_matches / len(df_crosswalk):6.1%})")
    print(f"  2. Fuzzy Matches:   {fuzzy_matches:4d} ({fuzzy_matches / len(df_crosswalk):6.1%})")
    print(f"  3. Unmatched:       {unmatched:4d} ({unmatched / len(df_crosswalk):6.1%})")
    
    print(f"\nExamples of Unmatched Location Names:")
    for ex in unmatched_examples:
        print(f"  - {ex}")

    print(f"\nSaved draft QA crosswalk to: {CROSSWALK_OUT}")

if __name__ == '__main__':
    main()
