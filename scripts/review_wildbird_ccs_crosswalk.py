#!/usr/bin/env python3
"""
scripts/review_wildbird_ccs_crosswalk.py

Reviews and finalizes the wild-bird CFIA CCS crosswalk:
- Resolves French accent character corruption (e.g. Montréal, Lévis, Château-Richer, Percé) using regex wildcard matching against official StatCan CSD and CCS reference tables.
- Resolves province assignment discrepancies in raw data (e.g. Moosomin No. 121, Britannia No. 502, Compton, Coaticook).
- Marks unorganized territories (e.g. Kivalliq, Unorganized) as manual_review_required.
- Outputs:
  - data/processed/geography/wildbird_cfia_ccs_crosswalk_reviewed.csv
  - results/data_qa/wildbird_cfia_ccs_crosswalk_review.md
"""

import os
import sys
import re
import difflib
import unicodedata
import pandas as pd
import numpy as np

DRAFT_CROSSWALK_PATH = 'data/processed/geography/wildbird_cfia_ccs_crosswalk_draft.csv'
LOOKUP_PATH = 'data/processed/lookups/csd_to_ccs_lookup.csv'
STATIC_FEATS_PATH = 'data/processed/features/csd_static_structural_features.csv'

REVIEWED_CROSSWALK_PATH = 'data/processed/geography/wildbird_cfia_ccs_crosswalk_reviewed.csv'
QA_MD_PATH = 'results/data_qa/wildbird_cfia_ccs_crosswalk_review.md'

def clean_alpha(s):
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize('NFKD', str(s))
    clean = "".join([c for c in nfkd if not unicodedata.combining(c)]).lower()
    clean = re.sub(r'[\?\ufffd\ufffe\uffff\xc3]', '.', clean)
    clean = re.sub(r'[^a-z0-9\.]', '', clean)
    return clean.strip()

def make_regex_pat(s):
    """Converts clean string containing wildcard '.' into anchored regex."""
    clean = clean_alpha(s)
    escaped = []
    for char in clean:
        if char == '.':
            escaped.append('.')
        else:
            escaped.append(re.escape(char))
    return "^" + "".join(escaped) + "$"

def main():
    print("=== Reviewing and Finalizing Wild-Bird CFIA CCS Crosswalk ===")

    if not os.path.exists(DRAFT_CROSSWALK_PATH):
        print(f"ERROR: Draft crosswalk missing at {DRAFT_CROSSWALK_PATH}")
        sys.exit(1)

    df_draft = pd.read_csv(DRAFT_CROSSWALK_PATH)
    total_locations = len(df_draft)
    print(f"Loaded draft crosswalk with {total_locations} unique location pairs.")

    # Load reference geography tables
    ref_ccs_rows = []
    ref_csd_rows = []

    if os.path.exists(LOOKUP_PATH):
        df_lk = pd.read_csv(LOOKUP_PATH, dtype=str)
        for _, r in df_lk.dropna(subset=['PRNAME_PRNOM', 'CCSNAME_SRUNOM']).drop_duplicates().iterrows():
            prov_full = str(r['PRNAME_PRNOM']).split('/')[0].strip()
            ccs_full = str(r['CCSNAME_SRUNOM']).strip()
            ccs_uid = str(r['CCSUID_SRUIDU']).strip() if pd.notnull(r['CCSUID_SRUIDU']) else None
            csd_full = str(r['CSDNAME_SDRNOM']).strip() if pd.notnull(r['CSDNAME_SDRNOM']) else None

            ref_ccs_rows.append({
                'ref_province': prov_full,
                'ref_province_clean': clean_alpha(prov_full).replace('.', ''),
                'ref_ccs_name': ccs_full,
                'ref_ccs_clean': clean_alpha(ccs_full),
                'ref_ccs_uid': ccs_uid
            })

            if csd_full:
                ref_csd_rows.append({
                    'ref_province': prov_full,
                    'ref_province_clean': clean_alpha(prov_full).replace('.', ''),
                    'ref_csd_name': csd_full,
                    'ref_csd_clean': clean_alpha(csd_full),
                    'ref_ccs_name': ccs_full,
                    'ref_ccs_uid': ccs_uid
                })

    if os.path.exists(STATIC_FEATS_PATH):
        df_st = pd.read_csv(STATIC_FEATS_PATH, dtype=str)
        if 'province_name' in df_st.columns and 'ccs_name' in df_st.columns:
            for _, r in df_st.dropna(subset=['province_name', 'ccs_name']).drop_duplicates().iterrows():
                prov_full = str(r['province_name']).split('/')[0].strip()
                ccs_full = str(r['ccs_name']).strip()
                ccs_uid = str(r['ccs_uid']).strip() if 'ccs_uid' in df_st.columns and pd.notnull(r['ccs_uid']) else None
                ref_ccs_rows.append({
                    'ref_province': prov_full,
                    'ref_province_clean': clean_alpha(prov_full).replace('.', ''),
                    'ref_ccs_name': ccs_full,
                    'ref_ccs_clean': clean_alpha(ccs_full),
                    'ref_ccs_uid': ccs_uid
                })

    df_ref_ccs = pd.DataFrame(ref_ccs_rows).drop_duplicates(subset=['ref_province_clean', 'ref_ccs_clean'])
    df_ref_csd = pd.DataFrame(ref_csd_rows).drop_duplicates(subset=['ref_province_clean', 'ref_csd_clean'])
    print(f"Loaded {len(df_ref_ccs):,} reference CCS pairs and {len(df_ref_csd):,} CSD mapping entries.")

    reviewed_rows = []

    for idx, row in df_draft.iterrows():
        raw_p = str(row['raw_province']).strip()
        raw_c = str(row['raw_ccs_name']).strip()
        p_clean = clean_alpha(raw_p.split('/')[0]).replace('.', '')
        c_clean = clean_alpha(raw_c)

        target_ccs_name = None
        target_ccs_uid = None
        match_category = 'UNRESOLVED'
        final_status = 'manual_review_required'
        notes = ""

        raw_regex_pat = make_regex_pat(raw_c)

        # Step 1: Direct exact or wildcard match against CCS reference in same province
        prov_ccs_sub = df_ref_ccs[df_ref_ccs['ref_province_clean'] == p_clean]
        ccs_matches = []

        for _, r_ref in prov_ccs_sub.iterrows():
            ref_clean = r_ref['ref_ccs_clean']
            ref_pat = make_regex_pat(ref_clean)

            # Match bidirectional regex
            if (c_clean == ref_clean) or re.match(ref_pat, c_clean.replace('.', 'a')) or re.match(raw_regex_pat, ref_clean.replace('.', 'a')):
                ccs_matches.append((r_ref['ref_ccs_name'], r_ref['ref_ccs_uid']))

        if len(ccs_matches) >= 1:
            target_ccs_name = ccs_matches[0][0]
            target_ccs_uid = ccs_matches[0][1]
            match_category = 'EXACT_MATCH' if c_clean == df_ref_ccs.iloc[0]['ref_ccs_clean'] else 'FRENCH_ACCENT_RESTORED'
            final_status = 'matched'
            notes = f"Matched CCS reference: '{raw_c}' -> '{target_ccs_name}'"
        else:
            # Step 2: Match against CSD (municipality) in same province to find parent CCS
            prov_csd_sub = df_ref_csd[df_ref_csd['ref_province_clean'] == p_clean]
            csd_matches = []

            for _, r_ref in prov_csd_sub.iterrows():
                ref_clean = r_ref['ref_csd_clean']
                ref_pat = make_regex_pat(ref_clean)

                if (c_clean == ref_clean) or re.match(ref_pat, c_clean.replace('.', 'a')) or re.match(raw_regex_pat, ref_clean.replace('.', 'a')):
                    csd_matches.append((r_ref['ref_ccs_name'], r_ref['ref_ccs_uid'], r_ref['ref_csd_name']))

            if len(csd_matches) >= 1:
                target_ccs_name = csd_matches[0][0]
                target_ccs_uid = csd_matches[0][1]
                csd_matched_name = csd_matches[0][2]
                match_category = 'CSD_MUNICIPALITY_TO_CCS'
                final_status = 'matched'
                notes = f"Mapped CSD municipality '{raw_c}' ({csd_matched_name}) to parent reference CCS '{target_ccs_name}'"
            else:
                # Step 3: Cross-province search (correcting raw data province assignment errors)
                cross_ccs_matches = []
                for _, r_ref in df_ref_ccs.iterrows():
                    ref_clean = r_ref['ref_ccs_clean']
                    ref_pat = make_regex_pat(ref_clean)
                    if (c_clean == ref_clean) or re.match(ref_pat, c_clean.replace('.', 'a')) or re.match(raw_regex_pat, ref_clean.replace('.', 'a')):
                        cross_ccs_matches.append((r_ref['ref_ccs_name'], r_ref['ref_ccs_uid'], r_ref['ref_province']))

                if len(cross_ccs_matches) >= 1:
                    target_ccs_name = cross_ccs_matches[0][0]
                    target_ccs_uid = cross_ccs_matches[0][1]
                    correct_prov = cross_ccs_matches[0][2]
                    match_category = 'PROVINCE_MISMATCH_RESOLVED'
                    final_status = 'matched'
                    notes = f"Resolved province assignment error (listed under {raw_p} in raw data, correctly mapped to {correct_prov} reference CCS '{target_ccs_name}')"
                else:
                    # Step 4: Check cross-province CSD search
                    cross_csd_matches = []
                    for _, r_ref in df_ref_csd.iterrows():
                        ref_clean = r_ref['ref_csd_clean']
                        ref_pat = make_regex_pat(ref_clean)
                        if (c_clean == ref_clean) or re.match(ref_pat, c_clean.replace('.', 'a')) or re.match(raw_regex_pat, ref_clean.replace('.', 'a')):
                            cross_csd_matches.append((r_ref['ref_ccs_name'], r_ref['ref_ccs_uid'], r_ref['ref_province']))

                    if len(cross_csd_matches) >= 1:
                        target_ccs_name = cross_csd_matches[0][0]
                        target_ccs_uid = cross_csd_matches[0][1]
                        correct_prov = cross_csd_matches[0][2]
                        match_category = 'PROVINCE_MISMATCH_RESOLVED'
                        final_status = 'matched'
                        notes = f"Resolved CSD province mismatch (listed under {raw_p} in raw data, mapped to {correct_prov} reference CCS '{target_ccs_name}')"
                    else:
                        # Step 5: Check if unorganized territory / ambiguous
                        c_lower = raw_c.lower()
                        if 'unorganized' in c_lower or 'territory' in c_lower or 'non-organized' in c_lower or 'unorganized' in p_clean:
                            final_status = 'manual_review_required'
                            match_category = 'UNORGANIZED_TERRITORY'
                            notes = "Unorganized territory / region requiring manual spatial assignment."
                        else:
                            final_status = 'manual_review_required'
                            match_category = 'UNRESOLVED'
                            notes = "No confident reference match found."

        reviewed_rows.append({
            'raw_province': raw_p,
            'raw_ccs_name': raw_c,
            'clean_province': p_clean,
            'clean_ccs_name': c_clean,
            'final_status': final_status,
            'match_category': match_category,
            'target_ccs_name': target_ccs_name,
            'target_ccs_uid': target_ccs_uid,
            'review_notes': notes
        })

    df_reviewed = pd.DataFrame(reviewed_rows)
    os.makedirs(os.path.dirname(REVIEWED_CROSSWALK_PATH), exist_ok=True)
    df_reviewed.to_csv(REVIEWED_CROSSWALK_PATH, index=False)
    print(f"\nSaved reviewed crosswalk to: {REVIEWED_CROSSWALK_PATH}")

    # Summary metrics
    matched_df = df_reviewed[df_reviewed['final_status'] == 'matched']
    unresolved_df = df_reviewed[df_reviewed['final_status'] == 'manual_review_required']

    matched_count = len(matched_df)
    unresolved_count = len(unresolved_df)
    coverage_pct = (matched_count / total_locations) * 100.0

    print("\n==========================================================================================")
    print("                      CROSSWALK REVIEW FINAL SUMMARY")
    print("==========================================================================================")
    print(f"Total Location Pairs: {total_locations}")
    print(f"Matched After Review: {matched_count} ({coverage_pct:.2f}%)")
    print(f"Still Unresolved:     {unresolved_count} ({100.0 - coverage_pct:.2f}%)")
    print("==========================================================================================")

    if unresolved_count > 0:
        print(f"\nUnresolved Location Pairs ({unresolved_count}):")
        for _, r in unresolved_df.iterrows():
            p_safe = r['raw_province'].encode('ascii', errors='replace').decode('ascii')
            c_safe = r['raw_ccs_name'].encode('ascii', errors='replace').decode('ascii')
            print(f"  Prov: {p_safe:<25} | Raw CCS: {c_safe:<30} | Category: {r['match_category']:<22} | Notes: {r['review_notes']}")

    # Generate Markdown QA Report
    os.makedirs(os.path.dirname(QA_MD_PATH), exist_ok=True)

    md_report = f"""# CFIA Wild-Bird CCS Crosswalk Review Report

## 1. Overview & Goal
Reviewed and finalized the wild-bird CFIA location crosswalk (`data/processed/geography/wildbird_cfia_ccs_crosswalk_reviewed.csv`) to map wild-bird outbreak records to reference Canadian Census Consolidated Subdivisions (CCS) before feature engineering.

## 2. Summary Statistics & Coverage
- **Total Unique Location Pairs**: **{total_locations}**
- **Successfully Matched After Review**: **{matched_count}** ({coverage_pct:.2f}%)
- **Still Unresolved (`manual_review_required`)**: **{unresolved_count}** ({100.0 - coverage_pct:.2f}%)
- **Percent Coverage**: **{coverage_pct:.2f}%**

## 3. Resolution Categorization
| Match Category | Count | Status | Description |
| :--- | :---: | :---: | :--- |
| **EXACT_MATCH / FRENCH_ACCENT_RESTORED** | {len(df_reviewed[df_reviewed['match_category'].isin(['EXACT_MATCH', 'FRENCH_ACCENT_RESTORED'])]):,} | `matched` | Exact province and CCS name match in reference lookup, or French accent corruption restored via bidirectional regex matching (e.g., *Montréal*, *Lévis*, *Château-Richer*, *Percé*, *Cap-Santé*, *Saint-Jérôme*, *Sainte-Adèle*, *Saint-André-de-Kamouraska*). |
| **CSD_MUNICIPALITY_TO_CCS** | {len(df_reviewed[df_reviewed['match_category'] == 'CSD_MUNICIPALITY_TO_CCS']):,} | `matched` | Municipal CSD names mapped to parent StatCan CCS reference. |
| **PROVINCE_MISMATCH_RESOLVED** | {len(df_reviewed[df_reviewed['match_category'] == 'PROVINCE_MISMATCH_RESOLVED']):,} | `matched` | Raw data province assignment errors corrected (e.g., *Moosomin No. 121* [MB $\\to$ SK], *Britannia No. 502* [AB $\\to$ SK], *Compton* / *Coaticook* [ON $\\to$ QC], *Kivalliq, Unorganized* [MB $\\to$ NU]). |
| **UNORGANIZED_TERRITORY** | {len(df_reviewed[df_reviewed['match_category'] == 'UNORGANIZED_TERRITORY']):,} | `manual_review_required` | Unorganized territories (*Kivalliq, Unorganized*) flagged for manual spatial assignment. |
| **UNRESOLVED** | {len(df_reviewed[df_reviewed['match_category'] == 'UNRESOLVED']):,} | `manual_review_required` | Names without high-confidence candidate match. |

## 4. Sample Restored Location Mappings
```csv
{matched_df[matched_df['match_category'].isin(['FRENCH_ACCENT_RESTORED', 'CSD_MUNICIPALITY_TO_CCS', 'PROVINCE_MISMATCH_RESOLVED'])][['raw_province', 'raw_ccs_name', 'target_ccs_name', 'review_notes']].head(20).to_csv(index=False)}
```

## 5. Remaining Unresolved Location Pairs (`manual_review_required`)
{"```csv\n" + unresolved_df[['raw_province', 'raw_ccs_name', 'match_category', 'review_notes']].to_csv(index=False) + "```" if unresolved_count > 0 else "All location pairs fully resolved."}

## 6. Output Files
- **Reviewed Crosswalk CSV**: `[wildbird_cfia_ccs_crosswalk_reviewed.csv](file:///{os.path.abspath(REVIEWED_CROSSWALK_PATH).replace('\\\\', '/')})`
- **QA Review Report**: `[wildbird_cfia_ccs_crosswalk_review.md](file:///{os.path.abspath(QA_MD_PATH).replace('\\\\', '/')})`
"""

    with open(QA_MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_report)

    print(f"\nSaved QA Markdown report to: {QA_MD_PATH}")

if __name__ == '__main__':
    main()
