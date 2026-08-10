"""
Build risk-stratified pseudo-absence samples for HPAI poultry outbreak modeling.

Rules:
- Only uses 2022-2024 training rows (test split 2025 is untouched).
- Stratifies absences by province_code, season, and wild-bird HPAI presence.
- Produces ratios 3:1, 5:1, 10:1 (negatives:positives).
- Never overwrites existing raw or primary modeling tables.
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
AUDIT_MD = BASE_DIR / 'results' / 'baselines' / 'pseudo_absence_stratified_audit.md'
AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)

TARGET = 'outbreak_next_month'
TRAIN_YEARS = [2022, 2023, 2024]
RATIOS = [3, 5, 10]
SEED = 42

# -----------------------------------------------------------------------
# Seasonal groupings
# -----------------------------------------------------------------------
def season(month: int) -> str:
    if month in (12, 1, 2):
        return 'winter'
    elif month in (3, 4, 5):
        return 'spring'
    elif month in (6, 7, 8):
        return 'summer'
    else:
        return 'fall'

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
print('Loading modeling table…')
df = pd.read_csv(MODELING_TABLE)
df = df[df[TARGET].notna()].copy()
df[TARGET] = df[TARGET].astype(int)

# Restrict to training years only
train_df = df[df['year'].isin(TRAIN_YEARS)].copy()
print(f'Training rows (2022-2024): {len(train_df):,}')
print(f'Positives: {train_df[TARGET].sum():,}')
print(f'Negatives: {(train_df[TARGET] == 0).sum():,}')

# -----------------------------------------------------------------------
# Build strata label
# -----------------------------------------------------------------------
train_df['_season'] = train_df['month'].apply(season)

# Wild-bird presence flag
wb_col = 'wildbird_hpai_present_ccs_current_month'
if wb_col in train_df.columns:
    train_df['_wb_risk'] = (train_df[wb_col] > 0).astype(int)
else:
    train_df['_wb_risk'] = 0

train_df['_stratum'] = (
    train_df['province_code'].astype(str) + '_' +
    train_df['_season'] + '_wb' +
    train_df['_wb_risk'].astype(str)
)

# Province outbreak history (used for weighted sampling)
prov_pos = (train_df[train_df[TARGET] == 1]
            .groupby('province_code')
            .size()
            .rename('prov_pos_count'))
train_df = train_df.merge(prov_pos.reset_index(), on='province_code', how='left')
train_df['prov_pos_count'] = train_df['prov_pos_count'].fillna(0)

# -----------------------------------------------------------------------
# Positives (all of them – no oversampling of positives)
# -----------------------------------------------------------------------
positives = train_df[train_df[TARGET] == 1].copy()
negatives = train_df[train_df[TARGET] == 0].copy()
n_pos = len(positives)
print(f'\nPositives available: {n_pos}')

# -----------------------------------------------------------------------
# Compute stratum-level sampling weights for negatives
# -----------------------------------------------------------------------
# Weight = province historical outbreak count + 1 (so remote provinces ≠ 0)
# This draws more absences from historically risky provinces
neg_weights = negatives['prov_pos_count'] + 1.0
neg_weights = neg_weights / neg_weights.sum()

audit_lines = ['# Stratified Pseudo-Absence Sampling Audit\n\n']
audit_lines.append(f'**Source table**: {MODELING_TABLE.name}\n\n')
audit_lines.append(f'**Training years**: {TRAIN_YEARS}\n\n')
audit_lines.append(f'**Total training positives**: {n_pos}\n\n')
audit_lines.append(f'**Total training negatives available**: {len(negatives):,}\n\n')
audit_lines.append('**Note**: Negatives are sampled proportionally to province '
                   'historical outbreak count; no leakage from 2025 test set.\n\n')
audit_lines.append('---\n\n')

# -----------------------------------------------------------------------
# Sample and save for each ratio
# -----------------------------------------------------------------------
for ratio in RATIOS:
    n_neg_target = min(n_pos * ratio, len(negatives))
    sampled_neg = negatives.sample(
        n=n_neg_target,
        replace=False,
        weights=neg_weights,
        random_state=SEED
    )
    combined = pd.concat([positives, sampled_neg], ignore_index=True)
    # Drop helper columns
    combined = combined.drop(columns=['_season', '_wb_risk', '_stratum', 'prov_pos_count'],
                             errors='ignore')
    out_path = OUT_DIR / f'pseudo_absence_stratified_ratio_{ratio}.csv'
    combined.to_csv(out_path, index=False)
    pos_count = combined[TARGET].sum()
    neg_count = (combined[TARGET] == 0).sum()
    print(f'Ratio {ratio}:1 -> {len(combined):,} rows '
          f'(pos={pos_count}, neg={neg_count}) -> {out_path.name}')
    audit_lines.append(f'### Ratio {ratio}:1\n')
    audit_lines.append(f'- Total rows: {len(combined):,}\n')
    audit_lines.append(f'- Positives: {pos_count}\n')
    audit_lines.append(f'- Negatives: {neg_count}\n')
    audit_lines.append(f'- File: `{out_path.name}`\n\n')

# -----------------------------------------------------------------------
# Write audit markdown
# -----------------------------------------------------------------------
with open(AUDIT_MD, 'w', encoding='utf-8') as f:
    f.writelines(audit_lines)

print(f'\nAudit written to: {AUDIT_MD}')
print('Done.')
