import pandas as pd
import unicodedata
import re

df_lk = pd.read_csv('data/processed/lookups/csd_to_ccs_lookup.csv', dtype=str)

def clean_ref(s):
    if not isinstance(s, str): return ""
    nfkd = unicodedata.normalize('NFKD', str(s))
    c = "".join([ch for ch in nfkd if not unicodedata.combining(ch)]).lower().strip()
    c = re.sub(r'[\?\ufffd\ufffe\uffff\xc3]', '.', c)
    c = re.sub(r'[^a-z0-9\.]', '', c)
    return c

df_lk['prov_clean'] = df_lk['PRNAME_PRNOM'].apply(lambda s: clean_ref(str(s).split('/')[0]))
df_lk['ccs_clean'] = df_lk['CCSNAME_SRUNOM'].apply(clean_ref)
df_lk['csd_clean'] = df_lk['CSDNAME_SDRNOM'].apply(clean_ref)

draft = pd.read_csv('data/processed/geography/wildbird_cfia_ccs_crosswalk_draft.csv')
unmatched = draft[draft['review_status'] == 'NEEDS_MANUAL_REVIEW']

matched_count = 0
print("Testing regex wildcard match against lookup table:")

for _, r in unmatched.iterrows():
    raw_p = str(r['raw_province']).strip()
    raw_c = str(r['raw_ccs_name']).strip()
    p_clean = clean_ref(raw_p.split('/')[0])
    c_clean = clean_ref(raw_c)

    # Build regex from ref_clean by replacing '.' with '.'
    pat_str = re.sub(r'[\?\ufffd\ufffe\uffff\xc3\.]', '.', c_clean)

    # Search ref entries where ref pattern matches c_clean OR c_clean pattern matches ref
    matched_name = None
    match_type = "UNMATCHED"

    for _, r_ref in df_lk.iterrows():
        p_ref = r_ref['prov_clean']
        c_ref = r_ref['ccs_clean']
        csd_ref = r_ref['csd_clean']

        # Make ref pattern
        c_ref_pat = "^" + c_ref.replace('.', '.') + "$"
        csd_ref_pat = "^" + csd_ref.replace('.', '.') + "$"

        if p_ref == p_clean:
            if re.match(c_ref_pat, c_clean) or re.match("^" + pat_str + "$", c_ref.replace('.', 'a')):
                matched_name = r_ref['CCSNAME_SRUNOM']
                match_type = "CCS_MATCH"
                break
            elif re.match(csd_ref_pat, c_clean) or re.match("^" + pat_str + "$", csd_ref.replace('.', 'a')):
                matched_name = r_ref['CCSNAME_SRUNOM']
                match_type = "CSD_MATCH"
                break

    matched_name = None
    match_type = "UNMATCHED"

    # Match CCS using regex pattern
    sub_ccs = df_lk[(df_lk['prov_clean'] == p_clean) & (df_lk['ccs_clean'].str.replace('.', '.', regex=False).str.match(pat_str, na=False))]
    sub_csd = df_lk[(df_lk['prov_clean'] == p_clean) & (df_lk['csd_clean'].str.replace('.', '.', regex=False).str.match(pat_str, na=False))]
    cross_ccs = df_lk[df_lk['ccs_clean'].str.replace('.', '.', regex=False).str.match(pat_str, na=False)]
    cross_csd = df_lk[df_lk['csd_clean'].str.replace('.', '.', regex=False).str.match(pat_str, na=False)]

    if len(sub_ccs) > 0:
        matched_name = sub_ccs.iloc[0]['CCSNAME_SRUNOM']
        match_type = "CCS_MATCH"
    elif len(sub_csd) > 0:
        matched_name = sub_csd.iloc[0]['CCSNAME_SRUNOM']
        match_type = "CSD_MATCH"
    elif len(cross_ccs) > 0:
        matched_name = cross_ccs.iloc[0]['CCSNAME_SRUNOM']
        match_type = f"CROSS_CCS ({cross_ccs.iloc[0]['PRNAME_PRNOM'].split('/')[0]})"
    elif len(cross_csd) > 0:
        matched_name = cross_csd.iloc[0]['CCSNAME_SRUNOM']
        match_type = f"CROSS_CSD ({cross_csd.iloc[0]['PRNAME_PRNOM'].split('/')[0]})"

    if matched_name:
        matched_count += 1

    c_safe = raw_c.encode('ascii', errors='replace').decode('ascii')
    m_safe = str(matched_name).encode('ascii', errors='replace').decode('ascii')
    print(f"Raw: {raw_p:<12} | {c_safe:<32} | Type: {match_type:<20} | Target: {m_safe}")

print(f"\nTotal matched out of {len(unmatched)}: {matched_count}/{len(unmatched)}")
