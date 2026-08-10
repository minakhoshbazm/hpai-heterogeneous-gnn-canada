import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_OBS_PATH = Path('d:/gdrive/models/data/raw/ebird_api/ebird_observations_raw_2026.csv')
BOUNDARY_DIR = BASE_DIR / 'data' / 'raw' / 'boundaries'
SHAPEFILE = BOUNDARY_DIR / 'lcsd000b21a_e.shp'

# Output directories
OUT_DIR = BASE_DIR / 'data' / 'processed' / 'ebird'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSIGNED_CSV = OUT_DIR / 'ebird_observations_assigned_2026.csv'
UNMATCHED_CSV = OUT_DIR / 'ebird_observations_unmatched_2026.csv'
AUDIT_MD = BASE_DIR / 'results' / 'data_qa' / 'ebird_spatial_audit_2026.md'

print('Loading raw observations...')
obs_df = pd.read_csv(RAW_OBS_PATH)
# Ensure proper column names
required_cols = ['latitude', 'longitude', 'observation_date', 'species_code', 'region', 'observation_id', 'retrieval_timestamp']
missing = set(required_cols) - set(obs_df.columns)
if missing:
    raise RuntimeError(f'Missing required columns in raw obs: {missing}')

# Build GeoDataFrame for observations
geometry = [Point(xy) for xy in zip(obs_df['longitude'], obs_df['latitude'])]
obs_gdf = gpd.GeoDataFrame(obs_df, geometry=geometry, crs='EPSG:4326')

print('Loading CSD boundaries...')
# Load CSD boundaries and add province name
csd_gdf = gpd.read_file(SHAPEFILE)[['CSDUID', 'PRUID', 'geometry']]
# Load province lookup (PRUID -> PRNAME)
prov_lookup = pd.read_csv(BASE_DIR / 'data' / 'lookups' / 'province_lookup.csv', dtype=str)
csd_gdf = csd_gdf.merge(prov_lookup, on='PRUID', how='left')
# Ensure CRS matches
if csd_gdf.crs != obs_gdf.crs:
    csd_gdf = csd_gdf.to_crs(obs_gdf.crs)
# Spatial join (points within polygons)
joined = gpd.sjoin(obs_gdf, csd_gdf[['CSDUID', 'PRUID', 'PRNAME', 'geometry']], how='left', predicate='within')
# renamed columns for clarity
joined = joined.rename(columns={'CSDUID': 'csd_uid', 'PRUID': 'province_uid', 'PRNAME': 'province_name'})

# Separate matched and unmatched
matched = joined[~joined['csd_uid'].isna()].copy()
unmatched = joined[joined['csd_uid'].isna()].copy()

# Save outputs
matched.to_csv(ASSIGNED_CSV, index=False)
unmatched.to_csv(UNMATCHED_CSV, index=False)

# Audit markdown
with open(AUDIT_MD, 'w', encoding='utf-8') as f:
    f.write('# eBird Spatial Assignment Audit (2026)\n\n')
    f.write(f'**Total observations loaded**: {len(obs_df)}\n\n')
    f.write(f'**Matched to a CSD**: {len(matched)}\n\n')
    f.write(f'**Unmatched (outside Canada or missing coordinates)**: {len(unmatched)}\n\n')
    f.write('Observations with missing or invalid coordinates are kept in the unmatched file for transparency.\n')
    f.write('\n')
    f.write('**Note**: No artificial coordinate imputation was performed.\n')

print('Spatial assignment complete.')
print(f'Matched CSV: {ASSIGNED_CSV}')
print(f'Unmatched CSV: {UNMATCHED_CSV}')
print(f'Audit report: {AUDIT_MD}')
