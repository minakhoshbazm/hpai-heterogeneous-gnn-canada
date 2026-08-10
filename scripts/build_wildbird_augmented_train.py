"""
Build a wild-bird-augmented training table for HPAI poultry outbreak modeling.

Approach:
- All poultry outbreak positives get sample_weight = 1.0
- Wild-bird HPAI CSD-months (all species) get sample_weight = 0.5
  (treated as softer positive signal, not equal to a confirmed poultry outbreak)
- Pseudo-absences drawn using stratified province-weighted sampling at ratio 5:1
- Test split (2025) is completely untouched

Outputs:
  data/processed/modeling/train_augmented_wildbird_soft_weight.csv
  results/baselines/wildbird_augmented_train_audit.md
"""
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODELING_TABLE = BASE_DIR / 'data' / 'processed' / 'modeling' / \
    'model_table_v1_environment_with_lags_wildbird.csv'
OUT_DIR = BASE_DIR / 'data' / 'processed' / 'modeling'
AUDIT_MD = BASE_DIR / 'results' / 'baselines' / 'wildbird_augmented_train_audit.md'
AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)

TARGET = 'outbreak_next_month'
WB_COUNT_COL = 'wildbird_hpai_count_ccs_current_month'
TRAIN_YEARS = [2022, 2023, 2024]
NEG_RATIO = 5       # 5 pseudo-absences per (poultry + wildbird) positive
WB_WEIGHT = 0.5     # sample weight for wild-bird rows
POULTRY_WEIGHT = 1.0
NEG_WEIGHT = 1.0
SEED = 42

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
print('Loading modeling table…')
df = pd.read_csv(MODELING_TABLE)
df = df[df[TARGET].notna()].copy()
df[TARGET] = df[TARGET].astype(int)

# Training split only
train_df = df[df['year'].isin(TRAIN_YEARS)].copy()
print(f'Training rows (2022-2024): {len(train_df):,}')

# -----------------------------------------------------------------------
# Poultry positives
# -----------------------------------------------------------------------
poultry_pos = train_df[train_df[TARGET] == 1].copy()
poultry_pos['sample_weight'] = POULTRY_WEIGHT
poultry_pos['row_source'] = 'poultry_positive'
print(f'Poultry positives: {len(poultry_pos):,}')

# -----------------------------------------------------------------------
# Wild-bird HPAI rows
#   - CSD-month has at least one wild-bird HPAI detection
#   - NOT already a poultry positive (no double-counting)
# -----------------------------------------------------------------------
if WB_COUNT_COL in train_df.columns:
    wb_mask = (train_df[WB_COUNT_COL] > 0) & (train_df[TARGET] == 0)
    wb_rows = train_df[wb_mask].copy()
    wb_rows['sample_weight'] = WB_WEIGHT
    wb_rows['row_source'] = 'wildbird_positive'
    print(f'Wild-bird HPAI rows (non-poultry-positive): {len(wb_rows):,}')
else:
    wb_rows = pd.DataFrame()
    print('Wild-bird HPAI count column not found; skipping wildbird augmentation.')

# -----------------------------------------------------------------------
# Pseudo-absence negatives (province-weighted)
# -----------------------------------------------------------------------
negatives = train_df[
    (train_df[TARGET] == 0) &
    (~train_df.index.isin(wb_rows.index if not wb_rows.empty else []))
].copy()

# Province outbreak history weight
prov_pos = (train_df[train_df[TARGET] == 1]
            .groupby('province_code')
            .size()
            .rename('prov_pos_count'))
negatives = negatives.merge(prov_pos.reset_index(), on='province_code', how='left')
negatives['prov_pos_count'] = negatives['prov_pos_count'].fillna(0)
neg_weights = negatives['prov_pos_count'] + 1.0
neg_weights = neg_weights / neg_weights.sum()

n_effective_pos = len(poultry_pos) + (len(wb_rows) if not wb_rows.empty else 0)
n_neg_target = min(n_effective_pos * NEG_RATIO, len(negatives))

sampled_neg = negatives.sample(
    n=n_neg_target,
    replace=False,
    weights=neg_weights,
    random_state=SEED
)
sampled_neg = sampled_neg.drop(columns=['prov_pos_count'], errors='ignore')
sampled_neg['sample_weight'] = NEG_WEIGHT
sampled_neg['row_source'] = 'pseudo_absence'
print(f'Pseudo-absences sampled: {len(sampled_neg):,}  '
      f'(ratio {NEG_RATIO}:1 vs {n_effective_pos} effective positives)')

# -----------------------------------------------------------------------
# Combine and save
# -----------------------------------------------------------------------
parts = [poultry_pos]
if not wb_rows.empty:
    parts.append(wb_rows)
parts.append(sampled_neg)

combined = pd.concat(parts, ignore_index=True)
# Shuffle rows so model doesn't see blocks
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

out_path = OUT_DIR / 'train_augmented_wildbird_soft_weight.csv'
combined.to_csv(out_path, index=False)
print(f'\nSaved augmented training table: {out_path}')
print(f'  Total rows : {len(combined):,}')
print(f'  Poultry pos: {(combined["row_source"] == "poultry_positive").sum()}')
print(f'  WB HPAI    : {(combined["row_source"] == "wildbird_positive").sum()}')
print(f'  Negatives  : {(combined["row_source"] == "pseudo_absence").sum()}')

# -----------------------------------------------------------------------
# Audit markdown
# -----------------------------------------------------------------------
with open(AUDIT_MD, 'w', encoding='utf-8') as f:
    f.write('# Wild-Bird Augmented Training Table Audit\n\n')
    f.write(f'**Source table**: {MODELING_TABLE.name}\n\n')
    f.write(f'**Training years**: {TRAIN_YEARS}\n\n')
    f.write('## Row composition\n\n')
    f.write(f'| Source | Count | sample_weight |\n')
    f.write(f'|--------|-------|---------------|\n')
    f.write(f'| Poultry outbreak positives | {len(poultry_pos)} | {POULTRY_WEIGHT} |\n')
    wb_n = len(wb_rows) if not wb_rows.empty else 0
    f.write(f'| Wild-bird HPAI CSD-months (non-poultry) | {wb_n} | {WB_WEIGHT} |\n')
    f.write(f'| Pseudo-absence negatives (province-weighted) | {len(sampled_neg):,} | {NEG_WEIGHT} |\n')
    f.write(f'| **Total** | **{len(combined):,}** | — |\n\n')
    f.write('## Notes\n\n')
    f.write('- Wild-bird HPAI rows use `sample_weight=0.5`; they represent elevated '
            'environmental HPAI risk but are **not** confirmed poultry outbreaks.\n')
    f.write('- Pseudo-absences are province-weighted (more absences from provinces '
            'with historical outbreaks) at ratio 5:1 vs effective positives.\n')
    f.write('- The 2025 test split is completely excluded from this file.\n')
    f.write('- To use sample weights in scikit-learn: '
            '`model.fit(X_train, y_train, sample_weight=train["sample_weight"])`\n')

print(f'Audit written to: {AUDIT_MD}')
print('Done.')
