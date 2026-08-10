import pandas as pd
import unicodedata
import re

df = pd.read_csv('data/processed/lookups/csd_to_ccs_lookup.csv')

def sa(s):
    nfkd = unicodedata.normalize('NFKD', str(s))
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()

df['csd_clean'] = df['CSDNAME_SDRNOM'].apply(sa)
df['prov_clean'] = df['PRNAME_PRNOM'].apply(lambda s: sa(str(s).split('/')[0]))

targets = [
    'Saint-Andr?-de-Kamouraska',
    'Montr?al',
    'Saint-J?r?me',
    'Sainte-Ad?le',
    'Sainte-Genevi?ve-de-Berthier',
    'Saint-Pierre-de-l\'?le-d\'Orl?ans',
    'Sainte-Ang?le-de-Monnoir',
    'Sainte-Anne-de-Beaupr?',
    'Saint-C?saire',
    'Sainte-H?n?dine',
    'Saint-?tienne-des-Gr?s',
    'Moosomin No. 121',
    'Britannia No. 502'
]

for t in targets:
    t_clean = sa(t).replace('?', '.')
    pattern = "^" + re.sub(r'[^a-z0-9\s\-\,\.\']', '.', t_clean) + "$"
    sub = df[df['csd_clean'].str.contains(t_clean, regex=True, na=False)]
    res = sub[['PRNAME_PRNOM', 'CSDNAME_SDRNOM', 'CCSNAME_SRUNOM', 'CCSUID_SRUIDU']].to_dict('records')
    print(f"Target: {t:<35} | Matches found: {len(res)}")
    for r in res[:2]:
        p = r['PRNAME_PRNOM'].split('/')[0].encode('ascii', errors='replace').decode('ascii')
        csd = str(r['CSDNAME_SDRNOM']).encode('ascii', errors='replace').decode('ascii')
        ccs = str(r['CCSNAME_SRUNOM']).encode('ascii', errors='replace').decode('ascii')
        print(f"   Prov: {p} | CSD: {csd} -> CCS: {ccs} ({r['CCSUID_SRUIDU']})")
