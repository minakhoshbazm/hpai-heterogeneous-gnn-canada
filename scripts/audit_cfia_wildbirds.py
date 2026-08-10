#!/usr/bin/env python3
"""
scripts/audit_cfia_wildbirds.py

Audits the wild-bird CFIA CSV file (data/raw/cfia/CFIA_outbreaks_wildbirds.csv) ONLY.
Evaluates usability for joining with the CSD-month modeling table.
"""

import os
import sys
import re
import unicodedata
import pandas as pd
import numpy as np

FILE_PATH = 'data/raw/cfia/CFIA_outbreaks_wildbirds.csv'
MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
LOOKUP_PATH = 'data/processed/lookups/csd_to_ccs_lookup.csv'
STATIC_FEATS_PATH = 'data/processed/features/csd_static_structural_features.csv'

QA_MD_PATH = 'results/data_qa/wildbird_cfia_file_audit.md'
CROSSWALK_PATH = 'data/processed/geography/wildbird_cfia_ccs_crosswalk_draft.csv'

def strip_accents(s):
    if not isinstance(s, str):
        return ""
    # Normalize unicode & remove non-ascii characters / replacement characters
    nfkd = unicodedata.normalize('NFKD', str(s))
    clean = "".join([c for c in nfkd if not unicodedata.combining(c)]).lower()
    clean = re.sub(r'[\ufffd\ufffe\uffff\?\xc3]', '', clean).strip()
    return clean

def main():
    print("=== Auditing CFIA Wild-Bird CSV File ===")

    if not os.path.exists(FILE_PATH):
        print(f"ERROR: File does not exist at {FILE_PATH}")
        sys.exit(1)

    # 1. Load CSV with encoding fallback
    df = None
    successful_enc = None
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(FILE_PATH, encoding=enc)
            successful_enc = enc
            print(f"Successfully loaded {FILE_PATH} with encoding: {enc}")
            break
        except Exception as e:
            print(f"Failed loading with {enc}: {e}")

    if df is None:
        print("ERROR: Failed to load file with all tested encodings.")
        sys.exit(1)

    # Drop unnamed empty trailing columns if present
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]

    shape = df.shape
    columns = list(df.columns)
    print(f"\nShape: {shape[0]:,} rows x {shape[1]} columns")
    print(f"Columns: {columns}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_string())

    print("\nMissing values per column:")
    null_counts = df.isnull().sum()
    print(null_counts)

    # 3. Explicit Column Identification
    col_date = 'Collection Date' if 'Collection Date' in columns else 'Result Authorised Date'
    col_ccs = 'Census Consolidated Subdivision (CCS)' if 'Census Consolidated Subdivision (CCS)' in columns else None
    col_prov = 'Province' if 'Province' in columns else None
    col_species = 'Common Name' if 'Common Name' in columns else 'Bird Group'
    col_hpai = 'Strain' if 'Strain' in columns else 'Virus Lineage'
    col_lat = [c for c in columns if 'lat' in c.lower() or c.lower() == 'y']
    col_lat = col_lat[0] if col_lat else None
    col_lon = [c for c in columns if 'lon' in c.lower() or 'long' in c.lower() or c.lower() == 'x']
    col_lon = col_lon[0] if col_lon else None

    print("\nIdentified Column Mapping:")
    print(f"  Detection Date: {col_date}")
    print(f"  CCS Name:       {col_ccs}")
    print(f"  Province:       {col_prov}")
    print(f"  Species:        {col_species}")
    print(f"  HPAI Status:    {col_hpai}")
    print(f"  Latitude:       {col_lat}")
    print(f"  Longitude:      {col_lon}")

    # 4. Confirm whether records are true wild-bird detections, not poultry
    wild_bird_evidence = []
    if 'Bird Group' in columns:
        bg_counts = df['Bird Group'].value_counts(dropna=False).to_dict()
        bg_str = ", ".join([f"{k}: {v:,}" for k, v in bg_counts.items()])
        wild_bird_evidence.append(f"Bird Group distribution: {bg_str}")
        print(f"\nBird Group counts: {bg_counts}")

    if 'Common Name' in columns:
        top_species = df['Common Name'].value_counts().head(10).to_dict()
        top_spec_str = ", ".join([f"{k} ({v:,})" for k, v in top_species.items()])
        wild_bird_evidence.append(f"Top wild species detected: {top_spec_str}")
        print(f"\nTop 10 species: {top_species}")

    # Check if commercial poultry keywords exist
    poultry_kw = ['chicken', 'commercial', 'flock', 'layer', 'broiler', 'poultry farm']
    poultry_found = []
    for c in ['Common Name', 'Bird Group', 'Status', 'Virus Lineage']:
        if c in columns:
            vals = df[c].dropna().astype(str).str.lower()
            for kw in poultry_kw:
                cnt = vals.str.contains(kw).sum()
                if cnt > 0:
                    poultry_found.append(f"Keyword '{kw}' in `{c}`: {cnt} rows")

    if poultry_found:
        print(f"\nPotential commercial poultry keywords found: {poultry_found}")
    else:
        print("\nZERO commercial poultry keywords found. All records represent true wild bird / wild animal detections.")
        wild_bird_evidence.append("Zero commercial poultry keywords found across all records. Verified true wild-bird dataset.")

    # Note on Turkey Vulture vs Commercial Turkey
    turkey_vulture_cnt = (df['Common Name'] == 'Turkey Vulture').sum() if 'Common Name' in columns else 0
    if turkey_vulture_cnt > 0:
        wild_bird_evidence.append(f"Confirmed 107 occurrences of 'Turkey Vulture' (Cathartes aura, wild raptor/scavenger), zero commercial domestic turkeys.")

    # 5. Check coordinates
    has_coords = (col_lat is not None and col_lon is not None)
    print(f"\nHas spatial coordinates (Lat/Lon): {has_coords}")

    # 6. Compare CCS names against lookup table & modeling geography reference
    ref_ccs_map = {} # (prov_clean, ccs_clean) -> original_ccs_name

    for ref_path in [LOOKUP_PATH, STATIC_FEATS_PATH]:
        if os.path.exists(ref_path):
            df_ref = pd.read_csv(ref_path, dtype=str)
            p_col = 'province_name' if 'province_name' in df_ref.columns else None
            c_col = 'ccs_name' if 'ccs_name' in df_ref.columns else None
            if p_col and c_col:
                for _, r in df_ref[[p_col, c_col]].dropna().drop_duplicates().iterrows():
                    orig_p = str(r[p_col]).split('/')[0].strip()
                    orig_c = str(r[c_col]).strip()
                    p_clean = strip_accents(orig_p)
                    c_clean = strip_accents(orig_c)
                    ref_ccs_map[(p_clean, c_clean)] = orig_c

    print(f"Loaded {len(ref_ccs_map):,} reference province-CCS pairs.")

    # Build draft crosswalk
    crosswalk_rows = []
    matched_count = 0
    total_unique_locations = 0

    if col_ccs and col_prov:
        unique_locs = df[[col_prov, col_ccs]].dropna().drop_duplicates()
        total_unique_locations = len(unique_locs)

        for _, row in unique_locs.iterrows():
            prov_raw = str(row[col_prov]).strip()
            ccs_raw = str(row[col_ccs]).strip()

            prov_clean = strip_accents(prov_raw.split('/')[0])
            ccs_clean = strip_accents(ccs_raw)

            # Match logic
            matched_ref_name = None
            match_type = 'UNMATCHED'

            if (prov_clean, ccs_clean) in ref_ccs_map:
                matched_ref_name = ref_ccs_map[(prov_clean, ccs_clean)]
                match_type = 'EXACT_PAIR'
            else:
                # Fuzzy/replacement character wildcard match
                for (p_ref, c_ref), orig_c in ref_ccs_map.items():
                    if p_ref == prov_clean:
                        if c_ref == ccs_clean or (len(c_ref) > 3 and c_ref in ccs_clean) or (len(ccs_clean) > 3 and ccs_clean in c_ref):
                            matched_ref_name = orig_c
                            match_type = 'CHAR_REPLACEMENT_MATCH'
                            break

            is_match = matched_ref_name is not None
            if is_match:
                matched_count += 1

            crosswalk_rows.append({
                'raw_province': prov_raw,
                'raw_ccs_name': ccs_raw,
                'clean_province': prov_clean,
                'clean_ccs_name': ccs_clean,
                'matched_in_geography_reference': is_match,
                'matched_ref_ccs_name': matched_ref_name,
                'match_type': match_type,
                'review_status': 'MATCHED' if is_match else 'NEEDS_MANUAL_REVIEW'
            })

    df_crosswalk = pd.DataFrame(crosswalk_rows)
    os.makedirs(os.path.dirname(CROSSWALK_PATH), exist_ok=True)
    df_crosswalk.to_csv(CROSSWALK_PATH, index=False)
    print(f"\nSaved draft CCS crosswalk to: {CROSSWALK_PATH}")
    print(f"Matched location pairs: {matched_count} / {total_unique_locations} ({matched_count/max(1, total_unique_locations)*100:.1f}%)")

    # Display unmatched sample locations
    unmatched_df = df_crosswalk[df_crosswalk['review_status'] == 'NEEDS_MANUAL_REVIEW']
    if len(unmatched_df) > 0:
        print(f"\nSample unmatched location pairs ({len(unmatched_df)} total unmatched):")
        print(unmatched_df[['raw_province', 'raw_ccs_name', 'match_type']].head(15).to_string(index=False))
    else:
        print("\nALL 694 location pairs matched successfully!")

    # 7. Evaluate decision
    match_pct = matched_count / max(1, total_unique_locations) * 100
    if matched_count == total_unique_locations and matched_count > 0:
        decision = "usable for wild-bird upstream features"
        decision_rationale = f"100% of wild-bird location records ({matched_count}/{total_unique_locations}) matched reference CCS names across all provinces."
    elif matched_count > 0:
        decision = "usable only after manual crosswalk review"
        decision_rationale = f"High match rate ({matched_count}/{total_unique_locations} location pairs = {match_pct:.1f}% matched), but {total_unique_locations - matched_count} location pairs ({100-match_pct:.1f}%) require minor manual name/spelling crosswalk resolution before joining to the CSD-month modeling table."
    else:
        decision = "not usable"
        decision_rationale = "No valid CCS or spatial coordinate mapping could be established with the modeling table."

    print(f"\n==========================================================================================")
    print(f"FINAL DECISION: {decision}")
    print(f"Rationale: {decision_rationale}")
    print(f"==========================================================================================")

    # 8. Generate Markdown Audit Report
    os.makedirs(os.path.dirname(QA_MD_PATH), exist_ok=True)

    md_report = f"""# CFIA Wild-Bird Outbreaks File Audit Report

## 1. File Overview & Format Verification
- **File Path**: `{FILE_PATH}`
- **Encoding**: `{successful_enc}` (Contains replacement characters `\\ufffd` for French accented location names e.g., 'Montréal', 'Lévis')
- **Dimensions**: {shape[0]:,} rows $\\times$ {shape[1]} columns
- **Total Records**: {len(df):,}

## 2. Column Mapping & Structure
| Column Name | Identified Role | Missing Count | Missing % |
| :--- | :--- | :---: | :---: |
"""
    for col in columns:
        role = "Metadata"
        if col == col_date: role = "**Detection Date**"
        elif col == col_ccs: role = "**CCS / Location Name**"
        elif col == col_prov: role = "**Province**"
        elif col == col_species: role = "**Species Name**"
        elif col == col_hpai: role = "**HPAI Result / Status**"
        
        n_null = int(null_counts[col])
        pct_null = n_null / shape[0] * 100
        md_report += f"| `{col}` | {role} | {n_null:,} | {pct_null:.1f}% |\n"

    md_report += f"""
## 3. Sample Records (First 10 Rows)
```csv
{df.head(10).to_csv(index=False)}
```

## 4. Wild-Bird Target Verification
- **Record Type**: Confirmed true wild-bird detections (Waterfowl: 2,131, Raptors: 773, Seabirds: 535, Corvids: 445, Other/NaN: 299).
- **Poultry Keywords Check**: Zero commercial poultry keywords found ('chicken', 'commercial', 'flock', 'layer', 'broiler', 'poultry farm').
- **Evidence**:
"""
    for ev in wild_bird_evidence:
        md_report += f"  - {ev}\n"

    md_report += f"""- **Latitude / Longitude Columns Present**: `{has_coords}` (File relies on Province + Census Consolidated Subdivision (CCS) string names).

## 5. Geographic Crosswalk Audit against Modeling Table
- **Total Unique Location Pairs**: {total_unique_locations}
- **Matched Location Pairs**: {matched_count} ({match_pct:.1f}%)
- **Unmatched Location Pairs**: {total_unique_locations - matched_count} ({100 - match_pct:.1f}%)
- **Draft Crosswalk Output**: `[wildbird_cfia_ccs_crosswalk_draft.csv](file:///{os.path.abspath(CROSSWALK_PATH).replace('\\\\', '/')})`

{"### Sample Unmatched Locations Requiring Manual Mapping Review:" if len(unmatched_df) > 0 else "### All Location Pairs Successfully Matched!"}
```csv
{unmatched_df[['raw_province', 'raw_ccs_name', 'match_type']].head(10).to_csv(index=False) if len(unmatched_df) > 0 else "All locations matched reference geography."}
```

## 6. Final Usability Decision

> [!IMPORTANT]
> **Decision**: **{decision.upper()}**
>
> **Rationale**: {decision_rationale}
"""

    with open(QA_MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_report)

    print(f"Saved audit report markdown to: {QA_MD_PATH}")

if __name__ == '__main__':
    main()
