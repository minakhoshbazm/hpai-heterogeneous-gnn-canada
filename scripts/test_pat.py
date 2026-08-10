import pandas as pd
import re
import unicodedata

df_lk = pd.read_csv('data/processed/lookups/csd_to_ccs_lookup.csv', dtype=str)

def sa(s):
    nfkd = unicodedata.normalize('NFKD', str(s))
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()

df_lk['prov_clean'] = df_lk['PRNAME_PRNOM'].apply(lambda s: sa(str(s).split('/')[0]))
df_lk['csd_clean'] = df_lk['CSDNAME_SDRNOM'].apply(sa)
df_lk['ccs_clean'] = df_lk['CCSNAME_SRUNOM'].apply(sa)

draft = pd.read_csv('data/processed/geography/wildbird_cfia_ccs_crosswalk_draft.csv')
unmatched = draft[draft['review_status'] == 'NEEDS_MANUAL_REVIEW']

print("Testing matching logic for draft unmatched rows:")
for _, r in unmatched.iterrows():
    raw_p = str(r['raw_province']).strip()
    raw_c = str(r['raw_ccs_name']).strip()
    p_clean = sa(raw_p.split('/')[0])

    pat_str = re.sub(r'[\?\ufffd\ufffe\uffff\xc3]', '.', raw_c.lower().strip())
    pat = "^" + pat_str + "$"

    # Try CCS match
    ccs_sub = df_lk[(df_lk['prov_clean'] == p_clean) & (df_lk['ccs_clean'].str.match(pat, na=False))]
    # Try CSD match
    csd_sub = df_lk[(df_lk['prov_clean'] == p_clean) & (df_lk['csd_clean'].str.match(pat, na=False))]

    # Try cross-prov CCS/CSD match
    cross_ccs = df_lk[df_lk['ccs_clean'].str.match(pat, na=False)]
    cross_csd = df_lk[df_lk['csd_clean'].str.match(pat, na=False)]

    matched_ccs_name = None
    matched_type = "UNMATCHED"

    if len(ccs_sub) > 0:
        matched_ccs_name = ccs_sub.iloc[0]['CCSNAME_SRUNOM']
        matched_type = "CCS_MATCH"
    elif len(csd_sub) > 0:
        matched_ccs_name = csd_sub.iloc[0]['CCSNAME_SRUNOM']
        matched_type = "CSD_MATCH"
    elif len(cross_ccs) > 0:
        matched_ccs_name = cross_ccs.iloc[0]['CCSNAME_SRUNOM']
        matched_type = f"CROSS_PROV_CCS ({cross_ccs.iloc[0]['PRNAME_PRNOM'].split('/')[0]})"
    elif len(cross_csd) > 0:
        matched_ccs_name = cross_csd.iloc[0]['CCSNAME_SRUNOM']
        matched_type = f"CROSS_PROV_CSD ({cross_csd.iloc[0]['PRNAME_PRNOM'].split('/')[0]})"

print("\nChecking exact CCS names in lookup table:")
for kw in ['Richer', 'Montr', 'Lévis', 'Levis', 'Québec', 'Quebec', 'Trois', 'Pér', 'Perce', 'Tach', 'Madeleine']:
    sub = df_lk[df_lk['CCSNAME_SRUNOM'].str.contains(kw, case=False, na=False)]
    vals = [str(v).encode('ascii', errors='replace').decode('ascii') for v in sub['CCSNAME_SRUNOM'].unique()[:5]]
    print(f"Keyword '{kw}':", vals)
